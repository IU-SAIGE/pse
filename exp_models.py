import json
import os
import pathlib
from contextlib import suppress
from typing import Optional, Union, Sequence
from typing import Tuple

import asteroid.models
import torch
import torch.nn.functional as tf
from torch.nn.modules.loss import _Loss

from exp_data import Mixtures, sample_rate, sisdr_improvement, \
    data_te_generalist
from exp_utils import make_2d, make_3d, pad_x_to_y, shape_reconstructed


_fft_size: int = 1024
_hop_size: int = 256
_eps: float = 1e-8
_recover_noise: bool = False
_window: torch.Tensor = torch.hann_window(_fft_size)


def _forward_single_mask(self, waveform: torch.Tensor):
    """Custom forward function to do single-mask two-source estimation.
    """
    # Remember shape to shape reconstruction
    shape = torch.tensor(waveform.shape)

    # Reshape to (batch, n_mix, time)
    waveform = make_3d(waveform)

    # Real forward
    tf_rep = self.forward_encoder(waveform)
    est_masks = self.forward_masker(tf_rep)
    est_masks = est_masks.repeat(1, 2, 1, 1)
    est_masks[:, 1] = 1 - est_masks[:, 1]
    masked_tf_rep = self.apply_masks(tf_rep, est_masks)
    decoded = self.forward_decoder(masked_tf_rep)

    reconstructed = pad_x_to_y(decoded, waveform)
    return shape_reconstructed(reconstructed, shape)


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
            -1, X_magnitude.shape[1]).detach()

        return predicted_snrs if self.training else _logistic(predicted_snrs)

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


def feedforward(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        model: torch.nn.Module,
        weights: Optional[torch.Tensor] = None,
        accumulation: bool = False,
        validation: bool = False,
        test: bool = False
) -> Tuple[torch.Tensor, float]:
    """Runs a feedforward pass through a model by unraveling batched data.
    """
    batch_size = inputs.shape[0]
    context = torch.no_grad() if (validation or test) else suppress()
    sisdri = 0
    loss_tensor = torch.Tensor(0)
    reduction = torch.mean

    with context:
        if accumulation:
            for i in range(batch_size):

                # unravel batch
                x = inputs[i].unsqueeze(0).cuda()
                t = targets[i].unsqueeze(0).cuda()

                # forward pass
                y = make_2d(model(x))

                # backwards pass
                if not test:
                    if weights is not None:
                        w = weights[i].unsqueeze(0)
                        loss_tensor = reduction(
                            loss_wmse(y, t, w))
                    else:
                        loss_tensor = reduction(
                            loss_mse(y, t))
                if not (validation or test):
                    loss_tensor.backward()

                # calculate signal improvement
                sisdri += float(torch.mean(
                    sisdr_improvement(y, t, x)))

            sisdri /= batch_size

        else:

            # forward pass
            inputs, targets = inputs.cuda(), targets.cuda()
            estimates = make_2d(model(inputs))

            # compute loss
            if not test:
                if weights is not None:
                    loss_tensor = reduction(
                        loss_wmse(estimates, targets, weights))
                else:
                    loss_tensor = reduction(
                        loss_mse(estimates, targets))

            # backwards pass
            if not (validation or test):
                loss_tensor.backward()

            # calculate signal improvement
            sisdri = float(reduction(
                sisdr_improvement(estimates, targets, inputs)))

    return loss_tensor, sisdri


def contrastive_feedforward(
        inputs_1: torch.Tensor,
        inputs_2: torch.Tensor,
        targets_1: torch.Tensor,
        targets_2: torch.Tensor,
        labels: torch.BoolTensor,
        lambda_positive: float,
        lambda_negative: float,
        model: torch.nn.Module,
        weights_1: Optional[torch.Tensor] = None,
        weights_2: Optional[torch.Tensor] = None,
        accumulation: bool = False,
        validation: bool = False,
        test: bool = False
) -> Tuple[float, float]:
    """Runs a feedforward pass through a model by unraveling batched data.
    """
    labels = labels.bool()
    batch_size = inputs_1.shape[0]
    context = torch.no_grad() if (validation or test) else suppress()
    sisdri, loss_value = 0, 0
    ratio_pos = float(sum(labels) / batch_size)
    ratio_neg = float(sum(~labels) / batch_size)
    use_dp = bool(weights_1 is not None) and bool(weights_2 is not None)

    with context:
        if accumulation:
            for i in range(batch_size):

                # unravel batch
                x_1 = inputs_1[i].unsqueeze(0).cuda()
                x_2 = inputs_2[i].unsqueeze(0).cuda()
                t_1 = targets_1[i].unsqueeze(0).cuda()
                t_2 = targets_2[i].unsqueeze(0).cuda()

                # forward pass
                y_1 = make_2d(model(x_1))
                y_2 = make_2d(model(x_2))

                # stack for batchwise loss
                x = torch.cat([x_1, x_2], dim=0)
                t = torch.cat([t_1, t_2], dim=0)
                y = torch.cat([y_1, y_2], dim=0)

                # calculate loss
                if not test:
                    if use_dp:
                        w_1 = weights_1[i].unsqueeze(0).cuda()
                        w_2 = weights_2[i].unsqueeze(0).cuda()
                        w_p = w_1 * w_2
                        w = torch.cat([w_1, w_2], dim=0)
                        loss_tensor = torch.mean(loss_wmse(y, t, w))
                        if labels[i]:
                            loss_tensor += lambda_positive * torch.mean(
                                loss_wmse(y_1, y_2, w_1))
                        else:
                            loss_tensor += lambda_negative * torch.mean(
                                torch.pow(loss_wmse(y_1, y_2, w_p) -
                                          loss_wmse(t_1, t_2, w_p), 2))
                    else:
                        loss_tensor = torch.mean(loss_mse(y, t))
                        if labels[i]:
                            loss_tensor += lambda_positive * torch.mean(
                                loss_mse(y_1, y_2))
                        else:
                            loss_tensor += lambda_negative * torch.mean(
                                torch.pow(loss_mse(y_1, y_2) -
                                          loss_mse(t_1, t_2), 2))
                    loss_value += float(loss_tensor)

                    # backwards pass
                    if not validation:
                        loss_tensor.backward()

                # calculate signal improvement
                sisdri += float(sisdr_improvement(y, t, x, 'mean'))

            sisdri /= batch_size
            loss_value /= batch_size

        else:

            # unravel batch
            x_1 = inputs_1.cuda()
            x_2 = inputs_2.cuda()
            t_1 = targets_1.cuda()
            t_2 = targets_2.cuda()

            # forward pass
            y_1 = make_2d(model(x_1))
            y_2 = make_2d(model(x_2))

            # stack for batchwise loss
            x = torch.cat([x_1, x_2], dim=0)
            t = torch.cat([t_1, t_2], dim=0)
            y = torch.cat([y_1, y_2], dim=0)
            l = labels

            # calculate loss
            if not test:
                if use_dp:
                    w_1 = weights_1.cuda()
                    w_2 = weights_2.cuda()
                    w_p = w_1 * w_2
                    w = torch.cat([w_1, w_2], dim=0)
                    loss_tensor = torch.mean(loss_wmse(y, t, w))
                    if ratio_pos:
                        loss_tensor += lambda_positive * ratio_pos * torch.mean(
                            loss_wmse(y_1[l], y_2[l], w_1[l]))
                    if ratio_neg:
                        loss_tensor += lambda_negative * ratio_neg * torch.mean(
                            torch.pow(loss_wmse(t_1[~l], t_2[~l], w_p[~l]) -
                                      loss_wmse(y_1[~l], y_2[~l], w_p[~l]), 2))
                else:
                    loss_tensor = torch.mean(loss_mse(y, t))
                    if ratio_pos:
                        loss_tensor += lambda_positive * ratio_pos * torch.mean(
                            loss_mse(y_1[l], y_2[l]))
                    if ratio_neg:
                        loss_tensor += lambda_negative * ratio_neg * torch.mean(
                            torch.pow(loss_mse(t_1[~l], t_2[~l]) -
                                      loss_mse(y_1[~l], y_2[~l]), 2))
                loss_value += float(loss_tensor)

                # backwards pass
                if not validation:
                    loss_tensor.backward()

            # calculate signal improvement
            sisdri += float(sisdr_improvement(y, t, x, 'mean'))

    return loss_value, sisdri


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


def init_gru(hidden_size=64, num_layers=2):
    model_config = locals()
    return (GRUNet(
        hidden_size=hidden_size,
        num_layers=num_layers
    ), model_config)


def init_model(
        model_name: str,
        model_size: Optional[str] = None,
        model_config: Optional[dict] = None
) -> Tuple[torch.nn.Module, int, dict]:
    """Instantiates model based on name and size.
    """
    # instantiate network
    model: torch.nn.Module
    model_config: dict = model_config or {}
    if not bool(model_size or model_config):
        raise ValueError('Expected either `model_size` or `model_config`.')
    if not (model_size in {'small', 'medium', 'large'}):
        raise ValueError('Size must be either "small", "medium", or "large".')
    if model_name == 'convtasnet':
        if model_config:
            model, model_config = init_ctn(**model_config)
        else:
            model, model_config = init_ctn(**{
                'small': dict(H=64, B=16, X=7, R=2),
                'medium': dict(H=128, B=32, X=7, R=2),
                'large': dict(H=256, B=64, X=7, R=2),
            }.get(model_size))
    elif model_name == 'dprnntasnet':
        if model_config:
            model, model_config = init_dprnn(**model_config)
        else:
            model, model_config = init_dprnn(**{
                'small': dict(B=64, H=64, R=1),
                'medium': dict(B=64, H=128, R=1),
                'large': dict(B=128, H=128, R=2)
            }.get(model_size))
    elif model_name == 'grunet':
        if model_config:
            model, model_config = init_gru(**model_config)
        else:
            model, model_config = init_gru(**{
                'small': dict(hidden_size=64, num_layers=2),
                'medium': dict(hidden_size=128, num_layers=2),
                'large': dict(hidden_size=256, num_layers=2)
            }.get(model_size))
    else:
        raise ValueError(f'Unsupported model name: "{model_name}".')
    model_nparams: int = count_parameters(model)

    return model, model_nparams, model_config


def count_parameters(network: torch.nn.Module):
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


@torch.no_grad()
def test_denoiser_from_module(
        model: torch.nn.Module,
        data_te: Union[Mixtures, Sequence[Mixtures]] = data_te_generalist,
        accumulation: bool = False
):
    """Evaluates speech enhancement model using provided dataset.
    """
    if not isinstance(data_te, (list, tuple)):
        data_te = [data_te]
    results = {}
    for dataset in data_te:
        batch = dataset(100, seed=0)
        key = dataset.speaker_ids_repr
        value = feedforward(batch.inputs, batch.targets,
                            model, None, accumulation, test=True)[1]
        results[key] = value
    num_tests = len(list(results.keys()))
    if num_tests > 1:
        results['mean']: float = sum(list(results.values())) / num_tests
    return results


@torch.no_grad()
def test_denoiser_from_file(
        checkpoint_path: Union[str, os.PathLike],
        data_te: Union[Mixtures, Sequence[Mixtures]] = data_te_generalist,
        accumulation: bool = False
):
    """Evaluates speech enhancement model checkpoint using provided dataset.
    """
    # load a config yaml file which should be in the same location
    config_file = pathlib.Path(checkpoint_path).with_name('config.json')
    if not config_file.exists():
        raise ValueError(f'Could not find {str(config_file)}.')
    with open(config_file, 'r') as fp:
        config: dict = json.load(fp)

    model = init_model(model_name=config.get('model_name'),
                       model_size=config.get('model_size'))[0]
    model.load_state_dict(torch.load(checkpoint_path).get('model_state_dict'),
                          strict=True)
    model.cuda()

    return test_denoiser_from_module(model, data_te, accumulation)


@torch.no_grad()
def test_denoiser_from_folder(
        checkpoint_folder: Union[str, os.PathLike],
        data_te: Union[Mixtures, Sequence[Mixtures]] = data_te_generalist,
        accumulation: bool = False
):
    """Selects speech enhancement model checkpoint from folder, and then
    evaluates using provided dataset.
    """
    # identify the best checkpoint using saved text file
    checkpoint_folder = pathlib.Path(checkpoint_folder)
    best_step_file = checkpoint_folder.joinpath('best_step.txt')
    if not best_step_file.exists():
        raise ValueError(f'Could not find {str(best_step_file)}.')
    with open(best_step_file, 'r') as fp:
        best_step = int(fp.readline())

    checkpoint_path = checkpoint_folder.joinpath(f'ckpt_{best_step:08}.pt')
    if not checkpoint_path.exists():
        raise IOError(f'{str(checkpoint_path)} does not exist.')
    print(f'Using {checkpoint_path}...')

    return test_denoiser_from_file(checkpoint_path, data_te, accumulation)


@torch.no_grad()
def test_denoiser(
        model: Union[str, os.PathLike, torch.nn.Module],
        data_te: Union[Mixtures, Sequence[Mixtures]] = data_te_generalist,
        accumulation: bool = False
):
    if isinstance(model, torch.nn.Module):
        return test_denoiser_from_module(model, data_te, accumulation)
    elif isinstance(model, (str, os.PathLike)):
        path = pathlib.Path(model)
        if path.is_dir():
            return test_denoiser_from_folder(path, data_te, accumulation)
        elif path.is_file():
            return test_denoiser_from_file(path, data_te, accumulation)
        else:
            raise ValueError(f'{str(path)} does not exist.')
    else:
        raise ValueError('Expected input to be PyTorch model or filepath.')


loss_mse = torch.nn.MSELoss(reduction='none')
loss_wmse = SegmentalLoss('mse', reduction='none')