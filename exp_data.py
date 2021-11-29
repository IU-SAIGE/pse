import os
import pathlib
from collections import namedtuple
from typing import List, Optional, Sequence, Tuple, Union, Callable

import json
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from asteroid.losses.sdr import singlesrc_neg_sisdr
from asteroid.losses.sdr import singlesrc_neg_snr
from numpy.random import Generator

from exp_utils import ExperimentError

example_duration: float = 4
sample_rate: int = 16000
example_length: int = int(sample_rate * example_duration)

_root_librispeech: str = '/data/asivara/librispeech/'
_root_demand: str = '/data/asivara/demand_1ch/'
_root_fsd50k: str = '/data/asivara/fsd50k_16khz/'
_root_musan: str = '/data/asivara/musan/'

_eps: float = 1e-8
_rng = np.random.default_rng(0)

Batch = namedtuple('Batch', 'inputs targets pre_snrs post_snrs')

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


def mix_signals(
        source: np.ndarray,
        noise: np.ndarray,
        snr_db: Union[float, np.ndarray]
) -> np.ndarray:
    """Function to mix signals.

    Args:
        source (np.ndarray): source signal
        noise (np.ndarray): noise signal
        snr_db (float): desired mixture SNR in decibels (scales noise)

    Returns:
        mixture (np.ndarray): mixture signal
    """
    energy_s = np.sum(source ** 2, axis=-1, keepdims=True)
    energy_n = np.sum(noise ** 2, axis=-1, keepdims=True)
    b = np.sqrt((energy_s / energy_n) * (10 ** (-snr_db / 10.)))
    return source + b * noise


def sparsity_index(
        signal: np.ndarray
) -> float:
    """Defines a sparsity value for a given signal, by computing the
    standard deviation of the segmental root-mean-square (RMS).
    """
    return float(np.std(librosa.feature.rms(signal).reshape(-1)))


def wav_read(
        filepath: Union[str, os.PathLike]
) -> Tuple[np.ndarray, float]:
    """Reads mono audio from WAV.
    """
    y, sr = sf.read(filepath, dtype='float32', always_2d=True)
    if sr != sample_rate:
        raise IOError(f'Expected sample_rate={sample_rate}, got {sr}.')
    # always pick up the first channel
    y = np.array(y[..., 0])
    return y, float(len(y) / sample_rate)


def wav_read_multiple(
        filepaths: Sequence[Union[str, os.PathLike]],
        concatenate: bool = False,
        seed: Optional[int] = None
) -> np.ndarray:
    """Loads multiple audio signals from file; may be batched or concatenated.
    """
    rng = np.random.default_rng(seed)
    signals = []
    min_length = int(example_duration * sample_rate)
    collate_fn: Callable = np.concatenate if concatenate else np.stack
    for filepath in filepaths:
        s, duration = wav_read(filepath)
        if not concatenate:
            if duration < example_duration:
                raise ValueError(f'Expected {filepath} to have minimum duration'
                                 f'of {example_duration} seconds.')
            offset = 0
            try:
                if len(s) > min_length:
                    offset = rng.integers(0, len(s) - min_length)
            except ValueError as e:
                print(filepath, len(s), min_length)
                raise e
            s = s[offset:offset + min_length]
        signals.append(s)
    return collate_fn(signals, axis=0)


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
            - sdr(mixture, target, scale_invariant=scale_invariant)
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
        clipped_speakers = [
            '101', '1069', '1175', '118', '1290', '1379', '1456', '1552',
            '1578', '1629', '1754', '1933', '1943', '1963', '198', '204',
            '2094', '2113', '2149', '22', '2269', '2618', '2751', '307',
            '3168', '323', '3294', '3374', '345', '3486', '3490', '3615',
            '3738', '380', '4148', '446', '459', '4734', '481', '5002',
            '5012', '5333', '549', '5561', '5588', '559', '5678', '5740',
            '576', '593', '6295', '6673', '7139', '716', '7434', '7800',
            '781', '8329', '8347', '882'
        ]
        df = df[~df['speaker_id'].isin(clipped_speakers)]

    # omit recordings which are smaller than an example
    df = df.query('duration >= @example_duration')

    # organize by split
    def assign_split_per_speaker(
            sgroup,
            duration_s_test: int = 30,
            duration_s_validation: int = 30,
            duration_s_train: int = 60,
            duration_s_prevalidation: int = 30,
    ):
        # designate partition indices based on the nearest cumulative duration
        sp_id = set(sgroup['speaker_id']).pop()
        cs = sgroup['duration'].cumsum()
        offset = min(sgroup.index)
        _d = duration_s_test
        split_te = (cs - _d).abs().idxmin() - offset
        _d += duration_s_validation
        split_vl = (cs - _d).abs().idxmin() - offset
        if split_vl == split_te: split_vl += 1
        _d += duration_s_train
        split_tr = (cs - _d).abs().idxmin() - offset
        if split_tr == split_vl: split_tr += 1
        _d += duration_s_prevalidation
        split_pvl = (cs - _d).abs().idxmin() - offset
        if split_pvl == split_tr: split_pvl += 1

        assert (split_te != split_vl), (sp_id, split_te, split_vl)
        assert (split_vl != split_tr), (sp_id, split_vl, split_tr)
        assert (split_tr != split_pvl), (sp_id, split_tr, split_pvl)

        # assign split
        sgroup.iloc[0:split_te]['split'] = 'test'
        sgroup.iloc[split_te:split_vl]['split'] = 'val'
        sgroup.iloc[split_vl:split_tr]['split'] = 'train'
        sgroup.iloc[split_tr:split_pvl]['split'] = 'preval'

        # return the modified speaker group
        return sgroup

    df = df.assign(split='pretrain').sort_values(['speaker_id', 'duration'])
    g = df.reset_index(drop=True).groupby('speaker_id')
    df = g.apply(assign_split_per_speaker)

    # shuffle the recordings
    df = df.sample(frac=1, random_state=0)

    # organize by subset and split
    df['subset_id'] = pd.Categorical(df['subset_id'], valid_subsets)
    df['split'] = pd.Categorical(df['split'], ['pretrain', 'preval', 'train',
                                               'val', 'test'])
    df = df.sort_values(['subset_id', 'split'])

    # ensure that all the audio files exist
    if not all(df.filepath.apply(os.path.isfile)):
        raise ValueError(f'Audio files missing, check {dataset_directory}.')

    # reindex and name the dataframe
    df = df[['filepath', 'subset_id', 'speaker_id',
             'split', 'duration', 'sparsity']]
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
    df = df[['filepath', 'duration', 'sparsity']]
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
            filepath = dataset_directory.joinpath(subdir, str(row.fname) + '.wav')
            if not filepath.exists():
                raise ValueError(f'{filepath} does not exist.')
            y, duration = wav_read(filepath)
            sparsity = sparsity_index(y)
            durations.append(duration)
            sparsities.append(sparsity)
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

    # omit sounds labeled as containing speech or music
    df['labels'] = df['labels'].apply(str.lower)
    df = df[~df['labels'].str.contains('speech')]
    df = df[~df['labels'].str.contains('music')]

    # omit recordings which are smaller than an example
    df = df.query('duration >= @example_duration')

    # shuffle the recordings
    df = df.sample(frac=1, random_state=0)

    # organize by split
    df['split'] = pd.Categorical(df['split'], ['train', 'val', 'test'])
    df = df.sort_values('split')

    # ensure that all the audio files exist
    if not all(df.filepath.apply(os.path.isfile)):
        raise ValueError(f'Audio files missing, check {dataset_directory}.')

    # reindex and name the dataframe
    df = df[['filepath', 'split', 'duration', 'sparsity']]
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
    df = df.query('duration >= @example_duration')

    # set aside the last sixty training signals for validation
    val_indices = df.query('split == "train"').iloc[-60:].index
    df.loc[val_indices, 'split'] = 'val'

    # organize by subset and split
    df['split'] = pd.Categorical(df['split'], ['train', 'val', 'test'])
    df = df.sort_values(['split'])

    # shuffle the recordings
    df = df.sample(frac=1, random_state=0)

    # ensure that all the audio files exist
    if not all(df.filepath.apply(os.path.isfile)):
        raise ValueError(f'Audio files missing, check {dataset_directory}.')

    # reindex and name the dataframe
    df = df[['filepath', 'split', 'duration', 'sparsity']]
    df = df.reset_index(drop=True)
    df.index.name = 'MUSAN'
    return df


def split_speakers(
        only_tc100: bool = True
) -> Tuple[List, List, List]:
    """Splits LibriSpeech speaker IDs into train, validation, and test sets.
    """
    df = df_librispeech
    root = pathlib.Path(__file__).absolute().parent
    speakers_vl = pd.read_csv(str(root.joinpath('speakers/validation.csv')))
    speakers_te = pd.read_csv(str(root.joinpath('speakers/test.csv')))
    speakers_tr = df['subset_id'].str.contains('train-clean-100')
    if not only_tc100:
        # will use speakers from both the 100hr and 360hr set
        speakers_tr = df['subset_id'].str.contains('train-clean')
    sp_ids_vl = set(speakers_vl['speaker_id'])
    sp_ids_te = set(speakers_te['speaker_id'])
    sp_ids_tr = set(df[speakers_tr]['speaker_id'])
    sp_ids_tr -= sp_ids_vl
    sp_ids_tr -= sp_ids_te
    return sorted(sp_ids_tr), sorted(sp_ids_vl), sorted(sp_ids_te)


def get_noise_corpus(name: str):
    """Returns a noise corpus DataFrame by name."""
    return {
        'musan': df_musan,
        'fsd50k': df_fsd50k,
        'demand': df_demand,
    }.get(name.lower())


class Mixtures:
    """Dataset for noisy speech signals.
    """

    def __init__(
            self,
            speaker_id_or_ids: Union[int, Sequence[int]],
            split_speech: Optional[str] = 'all',
            split_premixture: Optional[str] = 'train',
            split_mixture: Optional[str] = 'train',
            corpus_premixture: str = 'fsd50k',
            corpus_mixture: str = 'musan',
            snr_premixture: Optional[Union[float, Tuple[float, float]]] = None,
            snr_mixture: Optional[Union[float, Tuple[float, float]]] = None,
            dataset_duration: Optional[float] = None,
    ):
        # verify speaker ID(s)
        if isinstance(speaker_id_or_ids, int):
            speaker_id_or_ids = [speaker_id_or_ids]
        elif not isinstance(speaker_id_or_ids, (list, set)):
            raise ValueError('Expected one or a sequence of speaker IDs.')
        if len(speaker_id_or_ids) < 1:
            raise ValueError('Expected one or more speaker IDs.')
        if not set(speaker_id_or_ids).issubset(set(speaker_ids_all)):
            raise ValueError('Invalid speaker IDs, not found in LibriSpeech.')
        self.speaker_ids = speaker_id_or_ids
        self.speaker_ids_repr = ''
        if set(self.speaker_ids) == set(speaker_ids_tr):
            self.speaker_ids_repr = 'speaker_ids_tr'
        elif set(self.speaker_ids) == set(speaker_ids_vl):
            self.speaker_ids_repr = 'speaker_ids_vl'
        elif set(self.speaker_ids) == set(speaker_ids_te):
            self.speaker_ids_repr = 'speaker_ids_te'
        elif set(self.speaker_ids) == set(speaker_ids_all):
            self.speaker_ids_repr = 'speaker_ids_all'

        # missing pairs of arguments
        if not split_premixture:
            if snr_premixture is not None:
                raise ValueError('Missing argument `split_premixture`.')
        if not split_mixture:
            if snr_mixture is not None:
                raise ValueError('Missing argument `split_mixture`.')

        # unpack mixture SNR values
        if isinstance(snr_premixture, tuple):
            snr_premixture_min = float(min(snr_premixture))
            snr_premixture_max = float(max(snr_premixture))
        elif isinstance(snr_premixture, (float, int)):
            snr_premixture_min = float(snr_premixture)
            snr_premixture_max = float(snr_premixture)
        elif snr_premixture is None:
            snr_premixture_min = None
            snr_premixture_max = None
        else:
            raise ValueError('Expected `snr_premixture` to be a float type or '
                             'a tuple of floats.')
        if isinstance(snr_mixture, tuple):
            snr_mixture_min = float(min(snr_mixture))
            snr_mixture_max = float(max(snr_mixture))
        elif isinstance(snr_mixture, (float, int)):
            snr_mixture_min = float(snr_mixture)
            snr_mixture_max = float(snr_mixture)
        elif snr_mixture is None:
            snr_mixture_min = None
            snr_mixture_max = None
        else:
            raise ValueError('Expected `snr_mixture` to be a float type or '
                             'a tuple of floats.')
        self.snr_premixture_min = snr_premixture_min
        self.snr_premixture_max = snr_premixture_max
        self.snr_mixture_min = snr_mixture_min
        self.snr_mixture_max = snr_mixture_max

        # verify corpus partitions
        if not (split_speech in
            ('all', 'pretrain', 'preval', 'train', 'val', 'test')):
            raise ValueError('Expected `split_speech` to be either "all", '
                             '"pretrain", "preval", "train", "val", or "test".')
        if snr_premixture is not None:
            if not (split_premixture in ('train', 'val', 'test')):
                raise ValueError('Expected `split_premixture` to be either '
                                 '"train", "val", or "test".')
        if snr_mixture is not None:
            if not (split_mixture in ('train', 'val', 'test')):
                raise ValueError('Expected `split_mixture` to be either '
                                 '"train", "val", or "test".')
        self.split_speech = split_speech
        self.split_premixture = split_premixture or ''
        self.split_mixture = split_mixture or ''

        # verify dataset duration
        if not isinstance(dataset_duration, (int, float, type(None))):
            raise ValueError('Expected `dataset_duration` to be a number.')
        self.dataset_duration = dataset_duration

        self.index = 0
        self.example_duration = example_duration

        # instantiate corpora
        self.corpus_s = df_librispeech.query(
            f'speaker_id in {self.speaker_ids}')
        if self.split_speech != 'all':
            self.corpus_s = self.corpus_s.query(
                f'split == "{self.split_speech}"')
        self.corpus_m = get_noise_corpus(corpus_premixture).query(
            f'split == "{self.split_premixture}"')
        self.corpus_n = get_noise_corpus(corpus_mixture).query(
            f'split == "{self.split_mixture}"')

        # calculate maximum random offset for all utterances
        max_offset_func = lambda d: d.assign(max_offset=(
                sample_rate * d['duration'] - example_length)).astype({
            'max_offset': int})
        self.corpus_s = max_offset_func(self.corpus_s)
        self.corpus_m = max_offset_func(self.corpus_m)
        self.corpus_n = max_offset_func(self.corpus_n)

        # keep track of the number of utterances, premixture noises,
        # and injected noises
        self.len_s = len(self.corpus_s)
        self.len_m = len(self.corpus_m)
        self.len_n = len(self.corpus_n)
        if self.len_s < 1:
            raise ValueError('Invalid speaker_id')

        # define flags
        self.is_personalized = bool(len(self.speaker_ids) == 1)
        self.add_premixture_noise = bool(
            (snr_premixture is not None) and (self.len_m > 0))
        self.add_noise = bool(
            (snr_mixture is not None) and (self.len_n > 0))

        if not self.is_personalized and self.add_premixture_noise:
            raise ExperimentError('Non-personalized dataset contains '
                                  'premixture noise.')

    def __dict__(self):
        return {
            'flags': {
                'is_personalized': self.is_personalized,
                'add_premixture_noise': self.add_premixture_noise,
                'add_noise': self.add_noise,
            },
            'speaker_ids': self.speaker_ids_repr or self.speaker_ids,
            'snr_premixture_min': self.snr_premixture_min,
            'snr_premixture_max': self.snr_premixture_max,
            'snr_mixture_min': self.snr_mixture_min,
            'snr_mixture_max': self.snr_mixture_max,
            'split_speech': self.split_speech,
            'split_premixture': self.split_premixture,
            'split_mixture': self.split_mixture,
            'dataset_duration': self.dataset_duration
        }

    def __repr__(self):
        return json.dumps(self.__dict__(), indent=2, sort_keys=True)

    def __call__(self, batch_size: int, seed: Optional[int] = None):

        if batch_size < 1:
            raise ValueError('batch_size must be at least 1.')

        if seed is None: self.index += 1
        tmp_index: int = 0 if seed is not None else self.index
        tmp_rng: Generator = np.random.default_rng(tmp_index)

        indices = np.arange(batch_size * tmp_index,
                            batch_size * (tmp_index + 1))
        s_filepaths = self.corpus_s.filepath.iloc[indices % self.len_s]
        m_filepaths = self.corpus_m.filepath.iloc[indices % self.len_m]
        n_filepaths = self.corpus_n.filepath.iloc[indices % self.len_n]

        s = wav_read_multiple(s_filepaths, seed=seed)
        x = p = s

        pre_snrs = np.array(np.nan * np.ones(batch_size))
        if self.add_premixture_noise:
            m = wav_read_multiple(m_filepaths, seed=seed)
            pre_snrs = tmp_rng.uniform(
                self.snr_premixture_min, self.snr_premixture_max,
                (batch_size, 1))
            x = p = mix_signals(s, m, pre_snrs)

        post_snrs = np.array(np.nan * np.ones(batch_size))
        if self.add_noise:
            n = wav_read_multiple(n_filepaths, seed=seed)
            post_snrs = tmp_rng.uniform(
                self.snr_mixture_min, self.snr_mixture_max,
                (batch_size, 1))
            x = mix_signals(p, n, post_snrs)

        scale_factor = float(np.abs(x).max() + _eps)
        return Batch(
            inputs=torch.FloatTensor(x) / scale_factor,  # mixture signal
            targets=torch.FloatTensor(p) / scale_factor,  # premixture signal
            pre_snrs=torch.FloatTensor(pre_snrs),
            post_snrs=torch.FloatTensor(post_snrs)
        )


# expose corpora and speaker lists
df_librispeech = dataframe_librispeech()
df_musan = dataframe_musan()
df_fsd50k = dataframe_fsd50k()
df_demand = dataframe_demand()
speaker_ids_tr, speaker_ids_vl, speaker_ids_te = split_speakers(False)
speaker_ids_all = speaker_ids_tr + speaker_ids_vl + speaker_ids_te
speaker_split_durations = df_librispeech.groupby(
    ['speaker_id', 'split']).agg('sum').duration

# expose training, validation, and test datasets
data_tr_generalist: Mixtures = Mixtures(
    speaker_ids_tr,
    split_speech='all',
    split_mixture='train',
    snr_mixture=(-5, 5)
)
data_ptr_specialist: Tuple[Mixtures] = tuple([
    Mixtures(
        speaker_id,
        split_speech='pretrain',
        split_premixture='train',
        split_mixture='train',
        snr_premixture=(0, 10),
        snr_mixture=(-5, 5)
    ) for speaker_id in speaker_ids_te
])
data_tr_specialist: Tuple[Mixtures] = tuple([
    Mixtures(
        speaker_id,
        split_speech='train',
        split_premixture='train',
        split_mixture='train',
        snr_premixture=(0, 10),
        snr_mixture=(-5, 5)
    ) for speaker_id in speaker_ids_te
])
data_vl_generalist: Mixtures = Mixtures(
    speaker_ids_vl,
    split_speech='all',
    split_mixture='val',
    snr_mixture=(-5, 5)
)
data_pvl_specialist: Tuple[Mixtures] = tuple([
    Mixtures(
        speaker_id,
        split_speech='preval',
        split_premixture='val',
        split_mixture='val',
        snr_premixture=(0, 10),
        snr_mixture=(-5, 5)
    ) for speaker_id in speaker_ids_te
])
data_vl_specialist: Tuple[Mixtures] = tuple([
    Mixtures(
        speaker_id,
        split_speech='val',
        split_premixture='val',
        split_mixture='val',
        snr_premixture=(0, 10),
        snr_mixture=(-5, 5)
    ) for speaker_id in speaker_ids_te
])
data_te_generalist: Mixtures = Mixtures(
    speaker_ids_te,
    split_speech='test',
    split_mixture='test',
    snr_mixture=(-5, 5)
)
data_te_specialist: Tuple[Mixtures] = tuple([
    Mixtures(
        speaker_id,
        split_speech='test',
        split_mixture='test',
        snr_mixture=(-5, 5)
    ) for speaker_id in speaker_ids_te
])