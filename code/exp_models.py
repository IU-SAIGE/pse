import json
import os
import pathlib
from contextlib import suppress
from typing import Any, Optional, Union, Sequence, Tuple, Dict, Callable

import asteroid.models
import torch
import torch.nn.functional as tf
from torch.nn.modules.loss import _Loss

from exp_data import Mixtures, sample_rate, sisdr_improvement, sdr, sisdr, wav_write
from exp_utils import make_2d, make_3d, pad_x_to_y, shape_reconstructed


_fft_size: int = 1024
_hop_size: int = 256
_eps: float = 1e-8
_recover_noise: bool = False
_window: torch.Tensor = torch.hann_window(_fft_size)
try:
    from pesq import pesq
except ImportError:
    pesq = lambda *a, **k: 0
    print('Module `pesq` not installed, this metric will be a no-op.')
try:
    from pystoi import stoi
except ImportError:
    stoi = lambda *a, **k: 0
    print('Module `pystoi` not installed, this metric will be a no-op.')


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


class DPTNet(asteroid.models.DPTNet):
    if _recover_noise:
        forward = _forward_single_mask


class GRUNet(torch.nn.Module):

    def __init__(self, hidden_size: int, num_layers: int = 2,
                 bidirectional: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # create a neural network which predicts a TF binary ratio mask
        self.rnn = torch.nn.GRU(
            input_size=int(_fft_size // 2 + 1),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.hidden_size * (1+self.bidirectional),
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
        loss_reg: Callable,
        loss_segm: Callable,
        weights: Optional[torch.Tensor] = None,
        accumulation: bool = False,
        test: bool = False,
        skip_input_metrics: bool = False,
        num_examples_to_save: int = 0
) -> Dict[str, float]:
    """Runs a feedforward pass through a model by unraveling batched data.
    """
    batch_size = inputs.shape[0]
    validation = not bool(model.training)
    context = torch.no_grad() if (validation or test) else suppress()
    r_sisdr_inp: float = 0
    r_sisdr_enh: float = 0
    r_sdr_inp: float = 0
    r_sdr_enh: float = 0
    r_pesq_inp: float = 0
    r_pesq_enh: float = 0
    r_stoi_inp: float = 0
    r_stoi_enh: float = 0
    r_loss: float = 0

    with context:
        for i in range(batch_size):

            # unravel batch
            x = inputs[i].unsqueeze(0).cuda()
            t = targets[i].unsqueeze(0).cuda()

            # forward pass
            y = make_2d(model(x))
            if 0 <= i < num_examples_to_save:
                wav_write(f'example_{i:02d}.wav',
                          y.reshape(-1).detach().cpu().numpy())

            # backwards pass
            if not test:
                if weights is not None:
                    w = weights[i].unsqueeze(0)
                    loss_tensor = torch.mean(
                        loss_segm(y, t, w))
                else:
                    loss_tensor = torch.mean(
                        loss_reg(y, t))
                loss_tensor /= batch_size
                r_loss += float(loss_tensor)
            if not (validation or test):
                loss_tensor.backward()

            # compute PESQ and STOI (perceptual scores) only during testing
            if test:
                _x = x.detach().cpu().numpy().squeeze()
                _y = y.detach().cpu().numpy().squeeze()
                _t = t.detach().cpu().numpy().squeeze()
                if not skip_input_metrics:
                    r_pesq_inp += pesq(sample_rate, _t, _x, 'wb')
                r_pesq_enh += pesq(sample_rate, _t, _y, 'wb')
                if not skip_input_metrics:
                    r_stoi_inp += stoi(_t, _x, sample_rate, True)
                r_stoi_enh += stoi(_t, _y, sample_rate, True)

            # calculate signal improvement
            if not skip_input_metrics:
                r_sdr_inp += float(sdr(x, t, reduction='mean'))
            r_sdr_enh += float(sdr(y, t, reduction='mean'))
            r_sisdr_inp += float(sisdr(x, t, reduction='mean'))
            r_sisdr_enh += float(sisdr(y, t, reduction='mean'))

        r_sdr_inp /= batch_size
        r_sdr_enh /= batch_size
        r_sisdr_inp /= batch_size
        r_sisdr_enh /= batch_size
        r_pesq_inp /= batch_size
        r_pesq_enh /= batch_size
        r_stoi_inp /= batch_size
        r_stoi_enh /= batch_size

    return dict(loss=r_loss,
                sdr_inp=r_sdr_inp,
                sdr_enh=r_sdr_enh,
                sisdri=(r_sisdr_enh-r_sisdr_inp),
                sisdr_inp=r_sisdr_inp,
                sisdr_enh=r_sisdr_enh,
                pesq_inp=r_pesq_inp,
                pesq_enh=r_pesq_enh,
                stoi_inp=r_stoi_inp,
                stoi_enh=r_stoi_enh,
    )


def contrastive_negative_term(ly, lt, term_type: str = 'max'):
    if term_type == 'max':
        return torch.mean(torch.max(ly, lt))
    elif term_type == 'abs':
        return torch.mean(torch.abs(ly - lt))
    else:
        return torch.mean(torch.pow(ly - lt, 2))


def contrastive_feedforward(
        inputs_1: torch.Tensor,
        inputs_2: torch.Tensor,
        targets_1: torch.Tensor,
        targets_2: torch.Tensor,
        labels: torch.BoolTensor,
        loss_reg: Callable,
        loss_segm: Callable,
        lambda_positive: float,
        lambda_negative: float,
        model: torch.nn.Module,
        weights_1: Optional[torch.Tensor] = None,
        weights_2: Optional[torch.Tensor] = None,
        negative_term_type: str = 'max',
        accumulation: bool = False,
        validation: bool = False,
        test: bool = False
) -> Dict[str, float]:
    """Runs a feedforward pass through a model by unraveling batched data.
    """
    labels = labels.bool()
    batch_size = inputs_1.shape[0]
    context = torch.no_grad() if validation else suppress()
    ratio_pos = float(sum(labels) / batch_size)
    ratio_neg = float(sum(~labels) / batch_size)
    use_dp = bool(weights_1 is not None) and bool(weights_2 is not None)
    r_sisdri: float = 0
    r_loss: float = 0
    r_loss_sig: float = 0
    r_loss_pos: float = 0
    r_loss_neg: float = 0

    with context:
        for i in range(batch_size):

            loss_tensor_sig, loss_tensor_pos, loss_tensor_neg = 0, 0, 0

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
            if use_dp:
                w_1 = weights_1[i].unsqueeze(0).cuda()
                w_2 = weights_2[i].unsqueeze(0).cuda()
                w_p = w_1 * w_2
                w = torch.cat([w_1, w_2], dim=0)
                loss_tensor_sig = torch.mean(loss_segm(y, t, w))
                if labels[i]:
                    loss_tensor_pos = torch.mean(
                        loss_segm(y_1, y_2, w_1))
                else:
                    loss_tensor_neg = contrastive_negative_term(
                        loss_segm(y_1, y_2, w_p), loss_segm(t_1, t_2, w_p))
            else:
                loss_tensor_sig = torch.mean(loss_reg(y, t))
                if labels[i]:
                    loss_tensor_pos = torch.mean(
                        loss_reg(y_1, y_2))
                else:
                    loss_tensor_neg = contrastive_negative_term(
                        loss_reg(y_1, y_2), loss_reg(t_1, t_2))

            loss_tensor_sig /= batch_size
            loss_tensor_pos *= lambda_positive / (batch_size / 2)
            loss_tensor_neg *= lambda_negative / (batch_size / 2)
            loss_tensor_total = (
                    loss_tensor_sig + loss_tensor_pos + loss_tensor_neg)

            r_loss += float(loss_tensor_total)
            r_loss_sig += float(loss_tensor_sig)
            r_loss_pos += float(loss_tensor_pos)
            r_loss_neg += float(loss_tensor_neg)

            # backwards pass
            if not validation:
                loss_tensor_total.backward()

            # calculate signal improvement
            r_sisdri += float(sisdr_improvement(y, t, x, 'mean'))

        r_sisdri /= batch_size

    return dict(loss=r_loss,
                loss_sig=r_loss_sig,
                loss_pos=r_loss_pos,
                loss_neg=r_loss_neg,
                sisdri=r_sisdri)


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


def init_dprnn(N=64, L=2, B=128, H=128, R=6, K=250, T='lstm', causal=False):
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
        rnn_type=T,
        bidirectional=(not causal)
    ), model_config)


def init_gru(hidden_size=64, num_layers=2, bidirectional=True):
    model_config = locals()
    return (GRUNet(
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional
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
    if not (model_size in {'tiny', 'small', 'medium', 'large'}):
        raise ValueError('Size must be either "small", "medium", or "large".')
    if model_name == 'convtasnet':
        if model_config:
            model, model_config = init_ctn(**model_config)
        else:
            model, model_config = init_ctn(**{
                'tiny': dict(H=32, B=8, X=7, R=2),
                'small': dict(H=64, B=16, X=7, R=2),
                'medium': dict(H=128, B=32, X=7, R=2),
                'large': dict(H=256, B=64, X=7, R=2),
            }.get(model_size))
    elif model_name == 'grunet':
        if model_config:
            model, model_config = init_gru(**model_config)
        else:
            model, model_config = init_gru(**{
                'tiny': dict(hidden_size=32, num_layers=2),
                'small': dict(hidden_size=64, num_layers=2),
                'medium': dict(hidden_size=128, num_layers=2),
                'large': dict(hidden_size=256, num_layers=2)
            }.get(model_size))
    else:
        raise ValueError(f'Unsupported model name: "{model_name}".')
    model_nparams: int = count_parameters(model)

    return model, model_nparams, model_config


def load_checkpoint(
        path: Union[str, os.PathLike]
) -> (torch.nn.Module, dict, int):

    input_path = pathlib.Path(path)
    print(input_path)

    # If the path suffix is the PyTorch file extension,
    # then it's already a checkpoint
    if input_path.is_file() and input_path.suffix == '.pt':
        checkpoint_path = str(input_path)

    # If it's a directory, get the latest checkpoint
    # from that folder.
    elif input_path.is_dir():
        try:
            m = {
                input_path.joinpath('ckpt_best.pt'),
                input_path.joinpath('ckpt_last.pt')
            }
            checkpoints = set(input_path.glob('*.pt'))
            if m.issubset(checkpoints):
                checkpoints.remove(input_path.joinpath('ckpt_last.pt'))
            checkpoint_path = str(max(checkpoints, key=os.path.getctime))
        except ValueError:
            raise IOError(f'Input directory {str(input_path)} does not contain '
                          f'checkpoints.')
    else:
        raise IOError(f'{str(input_path)} is not a checkpoint or directory.')

    # Get the appropriate config file.
    config_path = pathlib.Path(checkpoint_path).with_name('config.json')
    if not config_path.is_file():
        raise IOError(f'Missing config file at {str(input_path)}.')

    # Load the config file.
    with open(config_path, 'r') as fp:
        config: dict = json.load(fp)

    # Initialize the model
    model = init_model(model_name=config.get('model_name'),
                       model_size=config.get('model_size'))[0]
    ckpt = torch.load(checkpoint_path)
    num_examples: int = ckpt.get('num_examples')
    try:
        model.load_state_dict(ckpt.get('model_state_dict'), strict=True)
    except RuntimeError as e:
        if 'state_dict' in str(e):
            raise RuntimeError(f'{str(checkpoint_path)} is a mismatched model.')
    model.cuda()

    return model, config, num_examples


def count_parameters(network: Any) -> int:
    return sum(p.numel() for p in network.parameters() if p.requires_grad)



def test_denoiser_with_speaker(
        model: torch.nn.Module,
        speaker_id: int = 200,
        num_examples_to_save: int = 0
) -> dict:
    no_op_loss = lambda *a, **k: 0
    dataset = Mixtures(speaker_id, split_speech='test', split_mixture='test',
                       snr_mixture=(-5, 5))
    batch = dataset(100, seed=0)
    results = feedforward(batch.inputs, batch.targets, model,
                          weights=None, accumulation=True,
                          loss_reg=no_op_loss, loss_segm=no_op_loss,
                          test=True, num_examples_to_save=num_examples_to_save)
    return results


@torch.no_grad()
def test_denoiser_from_module(
        model: torch.nn.Module,
        data_te: Union[Mixtures, Sequence[Mixtures]],
        accumulation: bool = False
) -> dict:
    """Evaluates speech enhancement model using provided dataset.
    """
    no_op_loss = lambda *a, **k: 0
    if not isinstance(data_te, (list, tuple)):
        data_te = [data_te]
    results = {}
    for dataset in data_te:
        batch = dataset(100, seed=0)
        key = dataset.speaker_ids_repr
        results[key] = feedforward(
            batch.inputs, batch.targets,
            model, weights=None, accumulation=accumulation, test=True,
            loss_reg=no_op_loss, loss_segm=no_op_loss
        )
    return results


@torch.no_grad()
def test_denoiser_from_file(
        checkpoint_path: Union[str, os.PathLike],
        data_te: Union[Mixtures, Sequence[Mixtures]],
        accumulation: bool = False
) -> dict:
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
    ckpt = torch.load(checkpoint_path)
    num_examples: int = ckpt.get('num_examples')
    model.load_state_dict(ckpt.get('model_state_dict'), strict=True)
    model.cuda()

    results = test_denoiser_from_module(model, data_te, accumulation)
    results['num_examples'] = num_examples

    return results


@torch.no_grad()
def test_denoiser_from_folder(
        checkpoint_folder: Union[str, os.PathLike],
        data_te: Union[Mixtures, Sequence[Mixtures]],
        accumulation: bool = False,
        finetune: int = 0,
        use_last: bool = False
):
    """Selects speech enhancement model checkpoint from folder, and then
    evaluates using provided dataset.
    """
    finetune_suffix = f'_ft_{int(finetune):02d}' if finetune else ''
    # identify the best checkpoint using saved text file
    checkpoint_folder = pathlib.Path(checkpoint_folder)
    if use_last:
        checkpoint_path = checkpoint_folder.joinpath(f'ckpt_last.pt')
    elif checkpoint_folder.joinpath(f'ckpt_best{finetune_suffix}.pt').exists():
        checkpoint_path = checkpoint_folder.joinpath(
            f'ckpt_best{finetune_suffix}.pt')
    else:
        best_step_file = next(checkpoint_folder.glob('best_step*'))
        if not best_step_file.exists():
            raise ValueError(f'Could not find {str(best_step_file)}.')
        with open(best_step_file, 'r') as fp:
            best_step = int(fp.readline())
        checkpoint_path = checkpoint_folder.joinpath(
            f'ckpt_{best_step:08}{finetune_suffix}.pt')
    if not checkpoint_path.exists():
        raise IOError(f'{str(checkpoint_path)} does not exist.')

    return test_denoiser_from_file(checkpoint_path, data_te, accumulation)


@torch.no_grad()
def test_denoiser(
        model: Union[str, os.PathLike, torch.nn.Module],
        data_te: Union[Mixtures, Sequence[Mixtures]],
        accumulation: bool = False,
        finetune: int = 0,
        use_last: bool = False
):
    if isinstance(model, torch.nn.Module):
        return test_denoiser_from_module(model, data_te, accumulation)
    elif isinstance(model, (str, os.PathLike)):
        path = pathlib.Path(model)
        if path.is_dir():
            return test_denoiser_from_folder(path, data_te, accumulation,
                                             finetune, use_last)
        elif path.is_file():
            return test_denoiser_from_file(path, data_te, accumulation)
        else:
            raise ValueError(f'{str(path)} does not exist.')
    else:
        raise ValueError('Expected input to be PyTorch model or filepath.')
