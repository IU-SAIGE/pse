import asteroid.models
import numpy as np
import torch
import torch.nn.functional as tf

from typing import Optional, Set, Tuple
from torch.nn.modules.loss import _Loss
from exp_data import sample_rate


_fft_size: int = 1024
_hop_size: int = 256
_eps: float = 1e-8
_recover_noise: bool = False
_window: torch.Tensor = torch.hann_window(_fft_size)


def _make_2d(x: torch.Tensor):
    """Normalize shape of `x` to two dimensions: [batch, time]."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    elif x.ndim == 3:
        return x.squeeze(1)
    else:
        if x.ndim != 2: raise ValueError('Could not force 2d.')
        return x


def _make_3d(x: torch.Tensor):
    """Normalize shape of `x` to three dimensions: [batch, n_chan, time]."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        if x.ndim != 3: raise ValueError('Could not force 3d.')
        return x


def _pad_x_to_y(x: torch.Tensor, y: torch.Tensor, axis: int = -1):
    """Right-pad or right-trim first argument to have same size as second argument
    Args:
        x (torch.Tensor): Tensor to be padded.
        y (torch.Tensor): Tensor to pad `x` to.
        axis (int): Axis to pad on.
    Returns:
        torch.Tensor, `x` padded to match `y`'s shape.
    """
    if axis != -1:
        raise NotImplementedError
    inp_len = y.shape[axis]
    output_len = x.shape[axis]
    return torch.nn.functional.pad(x, [0, inp_len - output_len])


def _shape_reconstructed(reconstructed: torch.Tensor, size: torch.Tensor):
    """Reshape `reconstructed` to have same size as `size`
    Args:
        reconstructed (torch.Tensor): Reconstructed waveform
        size (torch.Tensor): Size of desired waveform
    Returns:
        torch.Tensor: Reshaped waveform
    """
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed


def _forward_single_mask(self, waveform: torch.Tensor):
    """Custom forward function to do single-mask two-source estimation.
    """
    # Remember shape to shape reconstruction
    shape = torch.tensor(waveform.shape)

    # Reshape to (batch, n_mix, time)
    waveform = _make_3d(waveform)

    # Real forward
    tf_rep = self.forward_encoder(waveform)
    est_masks = self.forward_masker(tf_rep)
    est_masks = est_masks.repeat(1, 2, 1, 1)
    est_masks[:, 1] = 1 - est_masks[:, 1]
    masked_tf_rep = self.apply_masks(tf_rep, est_masks)
    decoded = self.forward_decoder(masked_tf_rep)

    reconstructed = _pad_x_to_y(decoded, waveform)
    return _shape_reconstructed(reconstructed, shape)


def _logistic(v, beta: float = 1., offset: float = 0.):
    return 1 / (1 + torch.exp(-beta * (v - offset)))


def _stft(waveform: torch.Tensor):
    """Calculates the Short-time Fourier transform (STFT)."""

    # perform the short-time Fourier transform
    spectrogram = torch.stft(
        waveform, _fft_size, _hop_size,
        window=_window.to(waveform.device),
        return_complex=False
    )

    # swap seq_len & feature_dim of the spectrogram (for RNN processing)
    spectrogram = spectrogram.permute(0, 2, 1, 3)

    # calculate the magnitude spectrogram
    magnitude_spectrogram = torch.sqrt(spectrogram[..., 0] ** 2 +
                                       spectrogram[..., 1] ** 2)

    return spectrogram, magnitude_spectrogram


def _istft(spectrogram: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """Calculates the inverse Short-time Fourier transform (ISTFT)."""

    # apply a time-frequency mask if provided
    if mask is not None:
        spectrogram[..., 0] *= mask
        spectrogram[..., 1] *= mask

    # swap seq_len & feature_dim of the spectrogram (undo RNN processing)
    spectrogram = spectrogram.permute(0, 2, 1, 3)

    # perform the inverse short-time Fourier transform
    waveform = torch.istft(
        spectrogram, _fft_size, _hop_size,
        window=_window.to(spectrogram.device),
        return_complex=False
    )

    return waveform


class ConvTasNet(asteroid.models.ConvTasNet):
    if _recover_noise:
        forward = _forward_single_mask


class DPRNNTasNet(asteroid.models.DPRNNTasNet):
    if _recover_noise:
        forward = _forward_single_mask


class GRUNet(torch.nn.Module):

    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # create a neural network which predicts a TF binary ratio mask
        self.rnn = torch.nn.GRU(
            input_size=int(_fft_size // 2 + 1),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.hidden_size,
                out_features=int(_fft_size // 2 + 1)
            ),
            torch.nn.Sigmoid()
        )

    def forward(self, waveform: torch.Tensor):
        # convert waveform to spectrogram
        (x, x_magnitude) = _stft(waveform)

        # generate a time-frequency mask
        h = self.rnn(x_magnitude)[0]
        y = self.dnn(h)
        y = y.reshape_as(x_magnitude)

        # convert masked spectrogram back to waveform
        denoised = _istft(x, mask=y)

        return denoised


class SNRPredictor(torch.nn.Module):

    def __init__(self, hidden_size: int = 1024, num_layers: int = 3):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers

        # layers
        self.rnn = torch.nn.GRU(
            input_size=int(_fft_size // 2 + 1),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.dnn = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=1
        )

    def forward(self, waveform: torch.Tensor):

        # convert to time-frequency domain
        (_, X_magnitude) = _stft(waveform)

        # generate frame-by-frame SNR predictions
        predicted_snrs = self.dnn(self.rnn(X_magnitude)[0]).reshape(
            -1, X_magnitude.shape[1])

        return _logistic(predicted_snrs) if self.training else predicted_snrs

    def load(self):
        self.load_state_dict(torch.load('snr_predictor'), strict=False)


class SegmentalLoss(_Loss):
    """Loss function applied to audio segmented frame by frame."""

    def __init__(
        self,
        loss_type: str = 'sisdr',
        reduction: str = 'none',
        segment_size: int = 1024,
        hop_length: int = 256,
        windowing: bool = True,
        centering: bool = True,
        pad_mode: str = 'reflect'
    ):
        super().__init__(reduction=reduction)
        assert loss_type in ('mse', 'snr', 'sisdr', 'sdsdr')
        assert pad_mode in ('constant', 'reflect')
        assert isinstance(centering, bool)
        assert isinstance(windowing, bool)
        assert segment_size > hop_length > 0

        self.loss_type = loss_type
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.pad_mode = pad_mode

        self.centering = centering
        self.windowing = windowing

        self.unfold = torch.nn.Unfold(
            kernel_size=(1, segment_size),
            stride=(1, hop_length)
        )
        self.window = torch.hann_window(self.segment_size).view(1, 1, -1)

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        assert target.size() == estimate.size()
        assert target.ndim == 2
        assert self.segment_size < target.size()[-1]

        # subtract signal means
        target -= torch.mean(target, dim=1, keepdim=True)
        estimate -= torch.mean(estimate, dim=1, keepdim=True)

        # center the signals using padding
        if self.centering:
            signal_dim = target.dim()
            ext_shape = [1] * (3 - signal_dim) + list(target.size())
            p = int(self.segment_size // 2)
            target = tf.pad(target.view(ext_shape), [p, p], self.pad_mode)
            target = target.view(target.shape[-signal_dim:])
            estimate = tf.pad(estimate.view(ext_shape), [p, p], self.pad_mode)
            estimate = estimate.view(estimate.shape[-signal_dim:])

        # use unfold to construct overlapping frames out of inputs
        n_batch = target.size()[0]
        target = self.unfold(target.view(n_batch,1,1,-1)).permute(0,2,1)
        estimate = self.unfold(estimate.view(n_batch,1,1,-1)).permute(0,2,1)
        losses: torch.Tensor

        # window all the frames
        if self.windowing:
            self.window = self.window.to(target.device)
            target = torch.multiply(target, self.window)
            estimate = torch.multiply(estimate, self.window)

        # MSE loss
        if self.loss_type == 'mse':
            losses = ((target - estimate)**2).sum(dim=2)
            losses /= self.segment_size

        # SDR based loss
        else:

            if self.loss_type == 'snr':
                scaled_target = target
            else:
                dot = (estimate * target).sum(dim=2, keepdim=True)
                s_target_energy = (target ** 2).sum(dim=2, keepdim=True) + _eps
                scaled_target = dot * target / s_target_energy

            if self.loss_type == 'sisdr':
                e_noise = estimate - scaled_target
            else:
                e_noise = estimate - target

            losses = (scaled_target ** 2).sum(dim=2)
            losses = losses / ((e_noise ** 2).sum(dim=2) + _eps)
            losses += _eps
            losses = torch.log10(losses)
            losses *= -10

        # apply weighting (if provided)
        if weights is not None:
            assert losses.size() == weights.size()
            weights = weights.detach()
            losses = torch.multiply(losses, weights).mean(dim=1)

        if self.reduction == 'mean':
            losses = losses.mean()

        return losses


def init_ctn(N=512, L=16, B=128, H=512, Sc=128, P=3, X=8, R=3, causal=False):
    model_config = locals()
    return (ConvTasNet(
        n_src=1,
        sample_rate=sample_rate,
        n_filters=N,
        kernel_size=L,
        bn_chan=B,
        hid_chan=H,
        skip_chan=Sc,
        conv_kernel_size=P,
        n_blocks=X,
        n_repeats=R,
        causal=causal
    ), model_config)


def init_dprnn(N=64, L=2, B=128, H=128, R=6, K=250, causal=False):
    model_config = locals()
    return (DPRNNTasNet(
        n_src=1,
        sample_rate=sample_rate,
        n_filters=N,
        kernel_size=L,
        bn_chan=B,
        hid_size=H,
        n_repeats=R,
        chunk_size=K,
        bidirectional=(not causal)
    ), model_config)


def init_model(config: dict) -> Tuple[torch.nn.Module, int, dict]:
    """Instantiates model based on name and size.
    """
    # verify config
    expected_keys: Set[str] = {'model_name', 'model_size'}
    if not expected_keys.issubset(set(config.keys())):
        raise ValueError(f'Expected `config` to contain keys: {expected_keys}')

    # instantiate network
    model: torch.nn.Module
    model_config: dict
    model_name: str = config['model_name']
    if model_name == 'convtasnet':
        model, model_config = init_ctn(**{
            'small': dict(X=1, R=1),
            'medium': dict(X=2, R=1),
            'large': dict(X=2, R=2)
        }.get(config['model_size']))
    elif model_name == 'dprnntasnet':
        model, model_config = init_dprnn(**{
            'small': dict(B=64, H=64, R=1),
            'medium': dict(B=64, H=64, R=2),
            'large': dict(B=128, H=128, R=2)
        }.get(config['model_size']))
    elif model_name == 'grunet':
        model_config = dict(hidden_size={
            'small': 64,
            'medium': 128,
            'large': 256
        }.get(config['model_size']))
        model = GRUNet(**model_config)
    else:
        raise ValueError(f'Unsupported model name: "{model_name}".')
    model_nparams: int = count_parameters(model)

    return model, model_nparams, model_config


def count_parameters(network: torch.nn.Module):
    return sum(p.numel() for p in network.parameters() if p.requires_grad)

