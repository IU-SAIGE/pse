import librosa
import numpy as np
import os
import pandas as pd
import pathlib
import soundfile as sf
import torch

from asteroid.losses.sdr import singlesrc_neg_sisdr
from asteroid.losses.sdr import singlesrc_neg_sdsdr
from asteroid.losses.sdr import singlesrc_neg_snr
from typing import Optional, Sequence, Tuple, Union

_example_duration: float = 4
_sample_rate: int = 16000

_root_librispeech: str = '/data/asivara/librispeech/'
_root_demand: str = '/data/asivara/demand_1ch/'
_root_fsd50k: str = '/data/asivara/fsd50k_16khz/'
_root_musan: str = '/data/asivara/musan/'

_rng = np.random.default_rng(0)


def _make_2d(x: torch.Tensor):
    """Normalize shape of `x` to two dimensions: [batch, time]."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    elif x.ndim == 3:
        return x.squeeze(1)
    else:
        assert x.ndim == 2
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
        assert x.ndim == 3
        return x


def mix_signals(
    source: np.ndarray,
    noise: np.ndarray,
    snr_db: float
) -> np.ndarray:
    """Function to mix signals.

    Args:
        source (np.ndarray): source signal
        noise (np.ndarray): noise signal
        snr_db (float): desired mixture SNR in decibels (scales noise)

    Returns:
        mixture (np.ndarray): mixture signal
    """
    energy_s = np.sum(source**2, axis=-1, keepdims=True)
    energy_n = np.sum(noise**2, axis=-1, keepdims=True)
    b = np.sqrt((energy_s/energy_n)*(10**(-snr_db/10.)))
    return (source + b*noise)


def sparsity_index(
    signal: np.ndarray
) -> float:
    """Defines a sparsity value for a given signal, by computing the
    standard deviation of the segmental root-mean-square (RMS).
    """
    return np.std(librosa.feature.rms(signal).reshape(-1))


def wav_read(
    filepath: Union[str, os.PathLike]
) -> Tuple[np.ndarray, float]:
    """Reads mono audio from WAV.
    """
    y, sr = sf.read(filepath, dtype='float32', always_2d=True)
    if sr != _sample_rate:
        raise IOError(f'Expected sample_rate={_sample_rate}, got {sr}.')
    # always pick up the first channel
    y = y[..., 0]
    return (y, float(len(y)/_sample_rate))


def wav_read_multiple(
    filepaths: Sequence[Union[str, os.PathLike]],
    sample_within: bool = True
) -> np.ndarray:
    """Loads multiple audio signals from file and stacks them up as a batch.
    If `sample_within` is True, randomly offsets and truncates signals to
    equal length, i.e. the original audio signals may be variable-length.
    """
    signals = []
    min_length = int(_example_duration * _sample_rate)
    for filepath in filepaths:
        s, duration = wav_read(filepath)
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


def sisdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    reduction: Optional[str] = None
) -> torch.Tensor:
    """Calculate single source SI-SDR."""
    return sdr(estimate, target, scale_invariant=True, reduction=reduction)


def sisdr_improvement(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    reduction: Optional[str] = None
) -> torch.Tensor:
    """Calculate estimate to target SI-SDR improvement relative to mixture.
    """
    return sdr_improvement(
        estimate, target, mixture, scale_invariant=True, reduction=reduction)


def sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    scale_invariant: bool = False,
    reduction: Optional[str] = None
) -> torch.Tensor:
    """Calculate single source SDR."""
    ml = min(estimate.shape[-1], target.shape[-1])
    estimate = _make_2d(estimate)[..., :ml]
    target = _make_2d(target)[..., :ml]
    if scale_invariant:
        output = -1 * singlesrc_neg_sisdr(estimate, target)
    else:
        output = -1 * singlesrc_neg_snr(estimate, target)
    if reduction == 'mean':
        output = torch.mean(output)
    return output


def sdr_improvement(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    scale_invariant: bool = False,
    reduction: Optional[str] = None
) -> torch.Tensor:
    """Calculate estimate to target SDR improvement relative to mixture.
    """
    output = (
        sdr(estimate, target, scale_invariant=scale_invariant)
        - sdr(estimate, mixture, scale_invariant=scale_invariant)
    )
    if reduction == 'mean':
        output = torch.mean(output)
    return output


def dataframe_librispeech(
    dataset_directory: Union[str, os.PathLike] = _root_librispeech,
    omit_clipped: bool = False
) -> pd.DataFrame:
    """Creates a Pandas DataFrame with files from the LibriSpeech corpus.
    Root directory should mimic archive-extracted folder structure.
    Dataset may be downloaded at `<https://www.openslr.org/12/>`_.
    """
    dataset_directory = pathlib.Path(dataset_directory)
    if not dataset_directory.exists():
        raise ValueError(f'{dataset_directory} does not exist.')
    valid_subsets = [
        'train-clean-100',
        'train-clean-360',
        'dev-clean',
        'test-clean'
    ]
    if not dataset_directory.joinpath('dataframe.csv').exists():
        rows = []
        columns = [
            'subset_id',
            'speaker_id',
            'chapter_id',
            'utterance_id',
            'filepath',
            'duration',
            'sparsity'
        ]
        for filepath in sorted(dataset_directory.rglob('*.wav')):
            subset_id = [_ for _ in valid_subsets if _ in str(filepath)][0]
            speaker_id, chapter_id, utterance_id = filepath.stem.split('-')
            y, duration = wav_read(filepath)
            sparsity = sparsity_index(y)
            rows.append((subset_id, speaker_id, chapter_id,
                         utterance_id, str(filepath), duration, sparsity))
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

    if omit_clipped:
        # discard recordings from speakers who possess clipped recordings
        # (manually found using SoX, where 'volume adjustment' == 1.000)
        clipped_speakers = [ '101', '1069', '1175', '118', '1290', '1379',
            '1456', '1552', '1578', '1629', '1754', '1933', '1943', '1963',
            '198', '204', '2094', '2113', '2149', '22', '2269', '2618', '2751',
            '307', '3168', '323', '3294', '3374', '345', '3486', '3490', '3615',
            '3738', '380', '4148', '446', '459', '4734', '481', '5002', '5012',
            '5333', '549', '5561', '5588', '559', '5678', '5740', '576', '593',
            '6295', '6673', '7139', '716', '7434', '7800', '781', '8329',
            '8347', '882' ]
        df = df[~df['speaker_id'].isin(clipped_speakers)]

    # omit recordings which are smaller than an example
    df = df.query('duration >= @_example_duration')

    # shuffle the recordings
    df = df.sample(frac=1, random_state=0)

    # organize by split
    df['subset_id'] = pd.Categorical(df['subset_id'], valid_subsets)
    df = df.sort_values('subset_id')

    # ensure that all the audio files exist
    if not all(df.filepath.apply(os.path.isfile)):
        raise ValueError(f'Audio files missing, check {dataset_directory}.')

    # reindex and name the dataframe
    df = df.reset_index(drop=True)
    df.index.name = 'LIBRISPEECH'
    return df


def dataframe_demand(
    dataset_directory: Union[str, os.PathLike] = _root_demand
) -> pd.DataFrame:
    """Creates a Pandas DataFrame with files from the DEMAND corpus.
    Root directory should mimic archive-extracted folder structure.
    Dataset may be downloaded at `<https://www.zenodo.org/record/1227121/>`_.
    """
    dataset_directory = pathlib.Path(dataset_directory)
    if not dataset_directory.exists():
        raise ValueError(f'{dataset_directory} does not exist.')
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
            'duration',
            'sparsity'
        ]
        for filepath in sorted(dataset_directory.rglob('*.wav')):
            if 'ch01' not in filepath.stem:
                continue
            category_id = [_ for _ in valid_categories if
                           _[0].upper() == filepath.parent.stem[0].upper()][0]
            location_id = [_ for _ in valid_locations if
                           filepath.parent.stem[1:].upper() in _.upper()][0]
            y, duration = wav_read(filepath)
            sparsity = sparsity_index(y)
            rows.append((
                category_id,
                location_id,
                str(filepath),
                duration,
                sparsity))
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

    # ensure that all the audio files exist
    if not all(df.filepath.apply(os.path.isfile)):
        raise ValueError(f'Audio files missing, check {dataset_directory}.')

    # reindex and name the dataframe
    df = df.reset_index(drop=True)
    df.index.name = 'DEMAND'
    return df


def dataframe_fsd50k(
    dataset_directory: Union[str, os.PathLike] = _root_fsd50k
) -> pd.DataFrame:
    """Creates a Pandas DataFrame with files from the FSD50K corpus.
    Root directory should mimic archive-extracted folder structure.
    Dataset may be downloaded at `<https://www.zenodo.org/record/4060432/>`_.
    """
    dataset_directory = pathlib.Path(dataset_directory)
    if not dataset_directory.exists():
        raise ValueError(f'{dataset_directory} does not exist.')
    if not dataset_directory.joinpath('dataframe.csv').exists():

        # merge separate dev and eval sets into one big table
        df1 = pd.read_csv(next(dataset_directory.rglob('dev.csv')))
        df2 = pd.read_csv(next(dataset_directory.rglob('eval.csv')))
        df2['split'] = 'test'
        df = pd.concat([df1, df2])

        durations, filepaths, sparsities = [], [], []
        for row in df.itertuples():
            subdir = ('FSD50K.eval_audio' if row.split == 'test'
                      else 'FSD50K.dev_audio')
            filepath = dataset_directory.joinpath(subdir, str(row.fname)+'.wav')
            if not filepath.exists():
                raise ValueError(f'{filepath} does not exist.')
            y, duration = wav_read(filepath)
            sparsity = sparsity_index(y)
            durations.append(duration)
            sparsities.append(sparsity_index(y))
            filepaths.append(filepath)
        df['filepath'] = filepaths
        df['duration'] = durations
        df['sparsity'] = sparsities
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

    # remove the label ids for convenience
    del df['mids']

    # omit sounds labeled as containing speech or music
    df['labels'] = df['labels'].apply(str.lower)
    df = df[~df['labels'].str.contains('speech')]
    df = df[~df['labels'].str.contains('music')]

    # omit recordings which are smaller than an example
    df = df.query('duration >= @_example_duration')

    # shuffle the recordings
    df = df.sample(frac=1, random_state=0)

    # organize by split
    df['split'] = pd.Categorical(df['split'], ['train', 'val', 'test'])
    df = df.sort_values('split')

    # ensure that all the audio files exist
    if not all(df.filepath.apply(os.path.isfile)):
        raise ValueError(f'Audio files missing, check {dataset_directory}.')

    # reindex and name the dataframe
    df = df.reset_index(drop=True)
    df.index.name = 'FSD50K'
    return df


def dataframe_musan(
    dataset_directory: Union[str, os.PathLike] = _root_musan
) -> pd.DataFrame:
    """Creates a Pandas DataFrame with files from the MUSAN corpus.
    Root directory should mimic archive-extracted folder structure.
    Dataset may be downloaded at `<https://www.openslr.org/17/>`_.
    """
    dataset_directory = pathlib.Path(dataset_directory)
    if not dataset_directory.exists():
        raise ValueError(f'{dataset_directory} does not exist.')
    if not dataset_directory.joinpath('dataframe.csv').exists():
        rows = []
        noise_dirs = [
            'free-sound',
            'sound-bible',
        ]
        columns = [
            'split',
            'filepath',
            'duration',
            'sparsity'
        ]
        for filepath in sorted(dataset_directory.rglob('*.wav')):
            is_train = bool('FREE-SOUND' in str(filepath).upper())
            is_test = bool('SOUND-BIBLE' in str(filepath).upper())
            if not (is_train or is_test):
                continue
            split_id = 'train' if is_train else 'test'
            y, duration = wav_read(filepath)
            sparsity = sparsity_index(y)
            rows.append((split_id, str(filepath), duration, sparsity))
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

    # omit recordings which are smaller than an example
    df = df.query('duration >= @_example_duration')

    # shuffle the recordings
    df = df.sample(frac=1, random_state=0)

    # ensure that all the audio files exist
    if not all(df.filepath.apply(os.path.isfile)):
        raise ValueError(f'Audio files missing, check {dataset_directory}.')

    # reindex and name the dataframe
    df = df.reset_index(drop=True)
    df.index.name = 'MUSAN'
    return df

