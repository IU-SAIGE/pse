import asteroid.models
import itertools
import numpy as np
import pathlib
import os
import pandas as pd
import soundfile as sf
import torch
import yaml

from asteroid.losses.sdr import singlesrc_neg_sisdr
from datetime import datetime
from ray import tune
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union


_fft_size: int = 1024
_hop_size: int = 256
_window = torch.hann_window(_fft_size)
_eps: float = 1e-8
_batch_size: int = 8  # change this depending on GPU limitations
_sample_rate: int = 16000  # Hz
_example_duration: float = 4.  # seconds
_rng = np.random.default_rng(0)

_root_librispeech: str = '/data/asivara/librispeech/'
_root_demand: str = '/data/asivara/demand/'
_root_fsd50k: str = '/data/asivara/fsd50k_16khz/'


def _jitable_shape(x: torch.Tensor):
    """Gets shape of ``tensor`` as ``torch.Tensor`` type for jit compiler
    .. note::
        Returning ``tensor.shape`` of ``tensor.shape`` directly is not torchscript
        compatible as return type would not be supported.
    Args:
        tensor (torch.Tensor): Tensor
    Returns:
        torch.Tensor: Shape of ``tensor``
    """
    return torch.tensor(x.shape)


def _unsqueeze_to_3d(x: torch.Tensor):
    """Normalize shape of `x` to [batch, n_chan, time]."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x


def _unsqueeze_to_2d(x: torch.Tensor):
    """Normalize shape of `x` to [batch, time]."""
    if x.ndim == 1:
        return x.reshape(1, -1)
    elif x.ndim == 3:
        return x.squeeze(1)
    else:
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


def _forward(self, waveform: torch.Tensor, num_masks: int = 1):
    """Custom forward function to do single-mask two-source estimation.
    Args:
        waveform (torch.Tensor): mixture waveform
    Returns:
        torch.Tensor: estimate waveforms
    """
    # Remember shape to shape reconstruction, cast to Tensor for torchscript
    shape = _jitable_shape(waveform)

    # Reshape to (batch, n_mix, time)
    waveform = _unsqueeze_to_3d(waveform)

    # Real forward
    tf_rep = self.forward_encoder(waveform)
    est_masks = self.forward_masker(tf_rep)
    if num_masks == 1:
        est_masks = est_masks.repeat(1, 2, 1, 1)
        est_masks[:, 1] = 1 - est_masks[:, 1]
    masked_tf_rep = self.apply_masks(tf_rep, est_masks)
    decoded = self.forward_decoder(masked_tf_rep)

    reconstructed = _pad_x_to_y(decoded, waveform)
    return _shape_reconstructed(reconstructed, shape)


def _seed_everything(seed_value: int = 0):
    """Fix pseudo-random seed."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


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

    return (spectrogram, magnitude_spectrogram)


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
    # forward = _forward
    pass


class DPRNNTasNet(asteroid.models.DPRNNTasNet):
    # forward = _forward
    pass


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
        (X, X_magnitude) = _stft(waveform)

        # generate a time-frequency mask
        H = self.rnn(X_magnitude)[0]
        Y = self.dnn(H)
        Y = Y.reshape_as(X_magnitude)

        # convert masked spectrogram back to waveform
        denoised = _istft(X, mask=Y)

        return denoised


def sisdr(estimate: torch.Tensor, target: torch.Tensor,
          reduction: Optional[str] = None):
    """Calculate single source SI-SDR."""
    def _fix(signal):
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal)
        if len(signal.shape) < 2:
            signal = signal.unsqueeze(0)
        if len(signal.shape) > 2:
            signal = signal.squeeze(-2)
        return signal
    ml = min(estimate.shape[-1], target.shape[-1])
    estimate, target = _fix(estimate)[..., :ml], _fix(target)[..., :ml]
    output = -1 * singlesrc_neg_sisdr(estimate, target)
    if reduction == 'mean':
        output = torch.mean(output)
    return output


def sisdr_improvement(estimate: torch.Tensor, target: torch.Tensor,
                      mixture: torch.Tensor, reduction: Optional[str] = None):
    """Calculate estimate to target SI-SDR improvement relative to mixture.
    """
    output = sisdr(estimate, target) - sisdr(estimate, mixture)
    if reduction == 'mean':
        output = torch.mean(output)
    return output


def mix_signals(source: np.ndarray, interferer: np.ndarray, snr_db: float):
    """Function to mix signals.

    Args:
        source (np.ndarray): source signal
        interferer (np.ndarray): interferer signal
        snr_db (float): desired mixture SNR in decibels (by scaling interferer)

    Returns:
        mixture (np.ndarray): mixture signal
    """
    energy_s = np.sum(source**2, axis=-1, keepdims=True)
    energy_n = np.sum(interferer**2, axis=-1, keepdims=True)
    b = np.sqrt((energy_s/energy_n)*(10**(-snr_db/10.)))
    return (source + b*interferer)


def wav_read(filepath: pathlib.Path):
    """Reads mono audio from WAV."""
    y, sr = sf.read(filepath, dtype='float32', always_2d=True)
    if sr != _sample_rate:
        raise IOError(f'Expected sample_rate={_sample_rate}, got {sr}.')
    # always pick up the first channel
    return y[..., 0]


def wav_read_multiple(filepaths: Sequence[pathlib.Path],
                      sample_within: bool = True):
    signals = []
    min_length = int(_example_duration * _sample_rate)
    for filepath in filepaths:
        s = wav_read(filepath)
        duration = len(s) / _sample_rate
        if sample_within:
            if (duration < _example_duration):
                raise ValueError(f'Expected {filepath} to have minimum duration'
                                 f'of {_example_duration} seconds.')
            offset = 0
            try:
                if len(s) > min_length:
                    offset = _rng.integers(0, len(s) - min_length)
            except ValueError as e:
                print(filepath, len(s), min_length)
                raise e
            s = s[offset:offset+min_length]
        signals.append(s)
        if len(signals) > 1:
            if len(signals[-1]) != len(signals[-2]):
                raise ValueError('If sample_within=False, all signals '
                                 'should have equal length.')
    return np.stack(signals, axis=0)


def init_convtasnet(N=512, L=16, B=128, H=512, Sc=128, P=3, X=8, R=3,
                    causal=False):
    model_config = locals()
    return (ConvTasNet(
        # n_src hardcoded to 1 to trigger custom forward / speech enhancement
        n_src=1,
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


def init_dprnntasnet(N=512, L=16, B=128, H=256, Sc=128, P=3, X=8, R=3,
                    causal=False):
    model_config = locals()
    return (DPRNNTasNet(
        # n_src hardcoded to 1 to trigger custom forward / speech enhancement
        n_src=1,
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


def init_model(config: dict) -> torch.nn.Module:
    """Instantiates model based on name and size."""

    # verify config
    expected_keys = {'model_name', 'model_size'}
    if not expected_keys.issubset(set(config.keys())):
        raise ValueError(f'Expected `config` to contain keys: {expected_keys}')

    # instantiate network
    model: torch.nn.Module
    model_config: dict
    size: int
    if config['model_name'] == 'convtasnet':
        model, model_config = init_convtasnet(R={
            'small': 1,
            'medium': 2,
            'large': 3
        }.get(config['model_size']))
    elif config['model_name'] == 'dprnntasnet':
        model, model_config = init_dprnntasnet(R={
            'small': 1,
            'medium': 2,
            'large': 3
        }.get(config['model_size']))
    elif config['model_name'] == 'grunet':
        model = GRUNet(hidden_size={
            'small': 128,
            'medium': 256,
            'large': 512
        }.get(config['model_size']))
        model_config = dict(hidden_size={
            'small': 128,
            'medium': 256,
            'large': 512
        }.get(config['model_size']))
    else:
        raise ValueError('Unsupported model name: "'+config['model_name']+'".')

    return model, model_config


def load_librispeech(dataset_directory: Union[str, os.PathLike] = _root_librispeech):
    """Creates a dataframe from Librispeech (see https://www.openslr.org/12).
    """
    valid_subsets = [
        'train-clean-100',
        'train-clean-360',
        'dev-clean',
        'test-clean'
    ]
    dataset_directory = pathlib.Path(dataset_directory)
    if not dataset_directory.joinpath('dataframe.csv').exists():
        rows = []
        columns = [
            'subset_id',
            'speaker_id',
            'chapter_id',
            'utterance_id',
            'filepath',
            'duration'
        ]
        for filepath in sorted(dataset_directory.rglob('*.wav')):
            subset_id = [_ for _ in valid_subsets if _ in str(filepath)][0]
            speaker_id, chapter_id, utterance_id = filepath.stem.split('-')
            duration = len(wav_read(filepath)) / _sample_rate
            rows.append((subset_id, speaker_id, chapter_id,
                         utterance_id, str(filepath), duration))
        if not len(rows):
            raise ValueError(f'Could not find any .WAV files within '
                             f'{dataset_directory}.')
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(dataset_directory.joinpath('dataframe.csv'),
              header=columns,
              index=False,
              index_label=False)
    else:
        df = pd.read_csv(dataset_directory.joinpath('dataframe.csv'))

    # discard recordings from speakers who possess clipped recordings
    # (manually found using SoX, where 'volume adjustment' == 1.000)
    clipped_speakers = [ '101', '1069', '1175', '118', '1290', '1379', '1456',
        '1552', '1578', '1629', '1754', '1933', '1943', '1963', '198', '204',
        '2094', '2113', '2149', '22', '2269', '2618', '2751', '307', '3168',
        '323', '3294', '3374', '345', '3486', '3490', '3615', '3738', '380',
        '4148', '446', '459', '4734', '481', '5002', '5012', '5333', '549',
        '5561', '5588', '559', '5678', '5740', '576', '593', '6295', '6673',
        '7139', '716', '7434', '7800', '781', '8329', '8347', '882' ]
    df = df[~df['speaker_id'].isin(clipped_speakers)]

    # omit recordings which are smaller than an example
    df = df.query('duration >= @_example_duration')

    # shuffle the recordings, then organize by split
    df = df.sample(frac=1, random_state=0)
    df['subset_id'] = pd.Categorical(df['subset_id'], valid_subsets)
    df = df.sort_values('subset_id')

    return df.reset_index(drop=True)


def load_demand(dataset_directory: Union[str, os.PathLike] = _root_demand):
    """Creates a dataframe from DEMAND (see https://zenodo.org/record/1227121).
    """
    dataset_directory = pathlib.Path(dataset_directory)
    if not dataset_directory.joinpath('dataframe.csv').exists():
        rows = []
        valid_categories = [
            'domestic',
            'nature',
            'office',
            'public',
            'street',
            'transportation'
        ]
        valid_locations = [
            'kitchen',
            'washing',
            'park',
            'hallway',
            'office',
            'resto',
            'psquare',
            'bus',
            'metro',
            'living',
            'field',
            'river',
            'meeting',
            'cafeter',
            'station',
            'traffic',
            'car'
        ]
        columns = [
            'category_id',
            'location_id',
            'filepath',
            'duration'
        ]
        for filepath in sorted(dataset_directory.rglob('*.wav')):
            if 'ch01' not in filepath.stem:
                continue
            category_id = [_ for _ in valid_categories if
                           _[0].upper() == filepath.parent.stem[0].upper()][0]
            location_id = [_ for _ in valid_locations if
                           filepath.parent.stem[1:].upper() in _.upper()][0]
            duration = len(wav_read(filepath)) / _sample_rate
            rows.append((category_id, location_id, str(filepath), duration))
        if not len(rows):
            raise ValueError(f'Could not find any .WAV files within '
                             f'{dataset_directory}.')
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(dataset_directory.joinpath('dataframe.csv'),
              header=columns,
              index=False,
              index_label=False)
    else:
        df = pd.read_csv(dataset_directory.joinpath('dataframe.csv'))

    # shuffle the recordings
    df = df.sample(frac=1, random_state=0)

    return df.reset_index(drop=True)


def load_fsd50k(dataset_directory: Union[str, os.PathLike] = _root_fsd50k):
    """Creates a dataframe from FSD50K (see https://zenodo.org/record/4060432).
    """
    dataset_directory = pathlib.Path(dataset_directory)
    if not dataset_directory.joinpath('dataframe.csv').exists():

        # merge separate dev and eval sets into one big table
        df1 = pd.read_csv(next(dataset_directory.rglob('dev.csv')))
        df2 = pd.read_csv(next(dataset_directory.rglob('eval.csv')))
        df2['split'] = 'test'
        df = pd.concat([df1, df2])

        durations, filepaths = [], []
        for row in df.itertuples():
            subdir = ('FSD50K.eval_audio' if row.split == 'test'
                      else 'FSD50K.dev_audio')
            filepath = dataset_directory.joinpath(subdir, str(row.fname)+'.wav')
            if not filepath.exists():
                raise ValueError(f'{filepath} does not exist.')
            duration = len(wav_read(filepath)) / _sample_rate
            durations.append(duration)
            filepaths.append(filepath)
        df['filepath'] = filepaths
        df['duration'] = durations
        if not len(filepaths):
            raise ValueError(f'Could not find any .WAV files within '
                             f'{dataset_directory}.')
        columns = list(df.columns)
        df.to_csv(dataset_directory.joinpath('dataframe.csv'),
              header=columns,
              index=False,
              index_label=False)
    else:
        df = pd.read_csv(dataset_directory.joinpath('dataframe.csv'))

    # omit sounds labeled as containing speech or music
    df['labels'] = df['labels'].apply(str.lower)
    df = df[~df['labels'].str.contains('speech')]
    df = df[~df['labels'].str.contains('music')]

    # omit recordings which are smaller than an example
    df = df.query('duration >= @_example_duration')

    # remove the label ids for convenience
    del df['mids']

    # shuffle the recordings, then organize by split
    df = df.sample(frac=1, random_state=0)
    df['split'] = pd.Categorical(df['split'], ['train', 'val', 'test'])
    df = df.sort_values('split')

    return df.reset_index(drop=True)


def count_parameters(network: torch.nn.Module):
    return sum(
        p.numel() for p in network.parameters()
        if p.requires_grad
    )


def get_personalized_dataset(
    test_speaker: Union[int, str],
    environment_snr_db: Union[float, Tuple[float, float]],
    num_environments: int = 1):

    # build test set
    rng = np.random.default_rng(0)

    # retrieve datasets
    S_te = load_librispeech().query('subset_id == "test-clean"')
    S_te = S_te[['speaker_id', 'filepath', 'duration']]
    test_speakers = S_te['speaker_id'].astype(str).unique()
    test_speaker = str(test_speaker)
    if test_speaker not in test_speakers:
        raise ValueError(f'Invalid test speaker ID: {test_speaker}.')
    M = load_demand()

    # map each speaker to one or multiple environments
    if num_environments < 1 or not isinstance(num_environments, int):
        raise ValueError('Expected non-zero integer number of environments.')
    elif num_environments > len(M):
        raise ValueError(f'Maximum test-time environments is {len(M)}.')
    elif num_environments == 1:
        test_envs = rng.permutation(len(test_speakers)) % len(M)
        env_mapping = {k: v for (k, v) in zip(test_speakers, test_envs)}
    else:
        n = num_environments
        test_envs = [rng.choice(len(M), n, replace=False)
                     for _ in range(len(test_speakers))]
        env_mapping = {k: list(test_envs[i])
                       for (i, k) in enumerate(test_speakers)}

    # each test speaker in LibriSpeech has at least 300 seconds
    # of utterance data to work with. split their data as follows:
    # - test set = 30 seconds
    # - validation set = 30 seconds
    # - finetune set = 60 seconds
    # - train set = remainder
    utterance_list = S_te.query('speaker_id == @test_speaker')
    utterance_list = utterance_list.sort_values('duration').reset_index()
    utterance_list['cumsum'] = utterance_list['duration'].cumsum()
    split_te = (utterance_list['cumsum']-30).abs().idxmin()
    split_vl = (utterance_list['cumsum']-60).abs().idxmin()
    split_ft = (utterance_list['cumsum']-120).abs().idxmin()
    u_te = utterance_list.iloc[0:split_te]
    u_vl = utterance_list.iloc[split_te:split_vl]
    u_ft = utterance_list.iloc[split_vl:split_ft]
    u_tr = utterance_list.iloc[split_ft:]

    # load all the speaker audio and concatenate it into a
    # single long vector
    s_te = np.concatenate([wav_read(f) for f in u_te.filepath])
    s_vl = np.concatenate([wav_read(f) for f in u_vl.filepath])
    s_ft = np.concatenate([wav_read(f) for f in u_ft.filepath])
    s_tr = np.concatenate([wav_read(f) for f in u_tr.filepath])
    s = np.concatenate([s_te, s_vl, s_ft, s_tr])

    # load the premixture noise profile
    env_indices = env_mapping[test_speaker]
    if num_environments == 1:
        p = wav_read(M.iloc[env_indices]['filepath'])
    else:
        p = np.stack([wav_read(f) for f in
                      M.iloc[env_indices]['filepath']])

    # segment the premixture noise similarly
    split_te = len(s_te)
    split_vl = split_te + len(s_vl)
    split_ft = split_vl + len(s_ft)
    p_te = p[..., 0:split_te]
    p_vl = p[..., split_te:split_vl]
    p_ft = p[..., split_vl:split_ft]
    p_tr = p[..., split_ft:]

    # circularly loop the training premixture noise if it is
    # too short
    tile_count = -(-s_tr.shape[-1]//p_tr.shape[-1])
    if num_environments == 1:
        p_tr = np.tile(p_tr, tile_count)
    else:
        p_tr = np.tile(p_tr, (1, tile_count))
    p_tr = p_tr[..., :len(s_tr)]
    p = np.concatenate([p_te, p_vl, p_ft, p_tr], axis=-1)

    # get the correct mixing scalar based on the overall
    # energies of the speech and premixture datasets
    energy_s = np.sum(s**2, axis=-1, keepdims=True)
    energy_n = np.sum(p**2, axis=-1, keepdims=True)
    snr_db = np.mean(environment_snr_db)
    b = np.sqrt((energy_s/energy_n)*(10**(-snr_db/10.)))

    # combine all the signals
    x_te = s_te + b*p_te

    return {
        'train_speech': s_tr,
        'train_prenoise': p_tr,
        'finetune_speech': s_ft,
        'val_speech': s_vl,
        'val_prenoise': p_vl,
        'test_speech': s_te,
        'test_prenoise': p_te,
        'test_premixture': x_te,
        'environments': {i: j for (i, j) in enumerate(
            M.iloc[env_indices]['location_id'])}
    }


@torch.no_grad()
def test_model(
    model: torch.nn.Module,
    test_speaker: Optional[Union[int, str]] = None,
    environment_snr_db: Union[float, Tuple[float, float]] = 0,
    num_environments: int = 1):
    """Evaluates a speech enhancement model on one or many test set speakers.
    """
    def _run_test(model, test_speaker, environment_snr_db, num_environments):
        dataset = get_personalized_dataset(
            test_speaker, environment_snr_db, num_environments)
        s_te = torch.from_numpy(dataset['test_speech']
            ).float().flatten().unsqueeze(0).cuda()
        x_te = torch.from_numpy(dataset['test_premixture']
            ).float().flatten().unsqueeze(0).cuda()
        s_hat_te = model(x_te)
        return float(sisdr_improvement(s_hat_te, s_te, x_te).mean())

    if test_speaker == None:
        S_te = load_librispeech().query('subset_id == "test-clean"')
        S_te = S_te[['speaker_id', 'filepath', 'duration']]
        test_speakers = sorted(S_te['speaker_id'].astype(str).unique())
    else:
        test_speakers = [test_speaker]

    return {
        k: _run_test(model, k, environment_snr_db, num_environments)
        for k in test_speakers
    }


@torch.no_grad()
def test_checkpoint(
    checkpoint_path: Union[str, os.PathLike],
    test_speaker: Optional[Union[int, str]] = None,
    environment_snr_db: Union[float, Tuple[float, float]] = 0,
    num_environments: int = 1):
    """
    """
    # load a config yaml file which should be in the same location
    yaml_file = pathlib.Path(checkpoint_path).with_name('config.yaml')
    if not yaml_file.exists():
        raise ValueError(f'Could not find {str(yaml_file)}.')
    with open(yaml_file, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    model, _ = init_model(config)
    checkpoint_obj = torch.load(checkpoint_path)
    model_state_dict = checkpoint_obj['model_state_dict']
    model.load_state_dict(model_state_dict, strict=True)
    model.cuda()
    return test_model(model, test_speaker, environment_snr_db, num_environments)


def train_sup(config: dict, checkpoint_path: Optional[str] = None):
    """Train a fully-supervised speech enhancement model."""

    # verify config
    expected_config = {
        'learning_rate': float,
        'model_name': str, 'model_size': str,
        'mixture_snr': Tuple[float, float],
    }
    if not set(expected_config.keys()).issubset(set(config.keys())):
        raise ValueError(f'Expected `config` to contain keys: '
                         f'{set(expected_config.keys())}')
    config['batch_size'] = _batch_size
    config['sample_rate'] = _sample_rate
    config['example_duration'] = _example_duration

    # prepare neural net, optimizer, and loss function
    model, model_config = init_model(config)
    model = model.cuda()
    config['model_config'] = model_config
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config['learning_rate'])
    criterion = torch.nn.MSELoss(reduction='mean')

    # load a previous checkpoint if provided
    init_step = 0
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        init_step = ckpt['step']

    # pick up dataset splits
    S = load_librispeech()
    S_tr = S.query('subset_id == "train-clean-100"')
    S_vl = S.query('subset_id == "dev-clean"')
    S_te = S.query('subset_id == "test-clean"')
    N = load_fsd50k()
    N_tr = N.query('split == "train"')
    N_vl = N.query('split == "val"')
    N_te = N.query('split == "test"')

    # instantiate tensorboard
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    output_directory = pathlib.Path('runs').joinpath(
        current_time + '_' + ray_name(config=config))
    writer = SummaryWriter(output_directory)
    with open(output_directory.joinpath('config.yaml'), 'w',
        encoding='utf-8') as fp:
        yaml.dump(config, fp)

    # keep track of the minimum loss (to early stop)
    min_loss, min_loss_step = np.inf, 0

    # training loop
    try:
        for step in itertools.count(init_step):

            model.train()
            with torch.set_grad_enabled(True):

                # circularly index the datasets
                indices = np.arange(_batch_size * step,
                    _batch_size * (step + 1), 1)
                s = wav_read_multiple(S_tr.filepath[indices % len(S_tr)])
                n = wav_read_multiple(N_tr.filepath[indices % len(N_tr)])

                # mix the signals up at random snrs
                snrs = _rng.uniform(low=config['mixture_snr'][0],
                                    high=config['mixture_snr'][1],
                                    size=(_batch_size, 1))
                x = mix_signals(s, n, snrs)

                # convert to tensors
                s = torch.from_numpy(s).float().cuda()
                x = torch.from_numpy(x).float().cuda()

                # forward propagation
                # (use gradient accumulation for TasNet models)
                if 'tasnet' in config['model_name']:
                    sisdri_tr = 0
                    for i in range(_batch_size):
                        _s, _x = s[i].unsqueeze(0), x[i].unsqueeze(0)
                        s_hat = model(_x)
                        if len(s_hat.shape) == 3:
                            s_hat = s_hat[:, 0]
                        loss_tr = criterion(s_hat, _s).mean()
                        (loss_tr / _batch_size).backward()
                        with torch.no_grad():
                            sisdri_tr += sisdr_improvement(s_hat, _s, _x).mean()
                    sisdri_tr /= _batch_size
                else:
                    s_hat = model(x)
                    if len(s_hat.shape) == 3:
                        s_hat = s_hat[:, 0]
                    loss_tr = criterion(s_hat, s).mean()
                    loss_tr.backward()
                    with torch.no_grad():
                        sisdri_tr = sisdr_improvement(s_hat, s, x).mean()

                # back propagation
                optimizer.step()
                optimizer.zero_grad()

                # write summaries
                writer.add_scalar('MSELoss/train', float(loss_tr), step)
                writer.add_scalar('SISDRi/train', float(sisdri_tr), step)

            if (step % config.get('validate_every', 10)):
                continue

            model.eval()
            with torch.no_grad():

                s = wav_read_multiple(S_vl.filepath[0:_batch_size])
                n = wav_read_multiple(N_vl.filepath[0:_batch_size])
                x = mix_signals(s, n, 0)
                s = torch.from_numpy(s).float().cuda()
                x = torch.from_numpy(x).float().cuda()
                s_hat = model(x)
                if len(s_hat.shape) == 3:
                    s_hat = s_hat[:, 0]
                loss_vl = criterion(s_hat, s).mean()
                sisdri_vl = sisdr_improvement(s_hat, s, x).mean()
                writer.add_scalar('MSELoss/validation', float(loss_tr), step)
                writer.add_scalar('SISDRi/validation', float(sisdri_tr), step)

                # checkpoint whenever validation score improves
                if loss_vl < min_loss:
                    min_loss = loss_vl
                    min_loss_step = step
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, output_directory.joinpath(f'ckpt_{step:08}.pt'))

    except KeyboardInterrupt as e:
        print(f'Manually exited at step {step}; '
              f'best step was {min_loss_step}.')
        pass

    # close the summary
    writer.close()

    # print the location of the checkpoints
    print(f'Saved checkpoints to {output_directory}.')

    # exit the trainer
    return


def ray_launch(config: dict):
    """Parses a Ray configuration dictionary and launches an experiment."""
    print(config)


def ray_name(trial = None, config: Optional[dict] = None):
    if trial:
        config = trial.config
    elif not config:
        raise ValueError('Either `trial` or `config` must be set.')
    name = '{}_{}_{}{}'.format(
        config['training_procedure'], config['model_name'],
        config['model_size'][0].upper(),
        ('_'+config['test_scenario']
            if 'selfsup' in config['training_procedure'] else '')
    )
    return name


def main():

    # perform a sweep across all model configurations
    config = {
        'learning_rate': tune.grid_search([1e-3, 1e-4]),
        'model_name': tune.grid_search([
            'convtasnet', 'dprnntasnet', 'grunet']),
        'model_size': tune.grid_search([
            'small', 'medium', 'large']),
        'training_procedure': tune.grid_search([
            'sup', 'selfsup',
            'selfsup+contrastive', 'selfsup+purify',
            'selfsup+contrastive+purify'])
    }
    tune.run(
        ray_launch,
        trial_name_creator=ray_name,
        trial_dirname_creator=ray_name,
        local_dir='checkpoints',
        keep_checkpoints_num=1,
        checkpoint_score_attr='min-validation_loss',
        progress_reporter=CLIReporter(
            max_report_frequency=30,
        ),
        log_to_file=True,
        verbose=1,
        max_failures=-1,
        config=config
    )


if __name__ == '__main__':
    train_sup(dict(
        learning_rate=1e-4,
        model_name='convtasnet',
        model_size='small',
        training_procedure='sup',
        mixture_snr=(-10, 10)
    ))
