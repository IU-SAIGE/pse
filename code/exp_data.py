import os
import pathlib
from collections import namedtuple
from typing import List, Optional, Sequence, Tuple, Union, Callable

import json
import librosa
import numpy as np
import pandas as pd
import socket
import soundfile as sf
import torch
from asteroid.losses.sdr import singlesrc_neg_sisdr
from asteroid.losses.sdr import singlesrc_neg_snr
from numpy.random import Generator
from scipy.signal import convolve

from exp_utils import ExperimentError

example_duration: float = 4
sample_rate: int = 16000
example_length: int = int(sample_rate * example_duration)

_host = str(socket.gethostname().split('.')[-3:].pop(0))
_root_data: str = dict(
    audio='/media/sdc1/',
    transformer='/data/asivara/',
    juliet='/N/u/asivara/datasets/',
    gan='/media/sdb1/Data/'
).get(_host, '/N/u/asivara/datasets/')
_root_librispeech: str = _root_data + '/librispeech/'
_root_demand: str = _root_data + '/demand/'
_root_fsd50k: str = _root_data + '/fsd50k_16khz/'
_root_musan: str = _root_data + '/musan/'
_root_irsurvey: str = _root_data + '/ir_survey_16khz/'
_root_slr28: str = _root_data + '/RIRS_NOISES/'

_eps: float = 1e-8
_rng = np.random.default_rng(0)

Batch = namedtuple(
    'Batch', ('inputs','targets','pre_snrs','post_snrs'))
ContrastiveBatch = namedtuple(
    'ContrastiveBatch', ('inputs_1','targets_1','inputs_2','targets_2',
                         'labels','pre_snrs','post_snrs'))

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


def wav_write(
        filepath: Union[str, os.PathLike],
        array: np.ndarray
):
    sf.write(filepath, array, samplerate=sample_rate)
    return


def wav_read_multiple(
        filepaths: Sequence[Union[str, os.PathLike]],
        concatenate: bool = False,
        randomly_offset: bool = True,
        seed: Optional[int] = None
) -> np.ndarray:
    """Loads multiple audio signals from file; may be batched or concatenated.
    """
    rng = np.random.default_rng(seed)
    signals = []
    collate_fn: Callable = np.concatenate if concatenate else np.stack
    for filepath in filepaths:
        s, duration = wav_read(filepath)
        if not concatenate:
            # pad shorter signals up to expected length
            if len(s) < example_length:
                lengths = [(0, 0)] * s.ndim
                lengths[-1] = (0, example_length - len(s))
                s = np.pad(s, lengths, mode='constant')

            # randomly offset longer signals if desired
            offset: int = 0
            remainder: int = len(s) - example_length
            if randomly_offset and remainder > 0:
                offset = rng.integers(0, remainder)

            # trim exactly to the expected length
            s = s[offset:offset + example_length]
        signals.append(s)
    return collate_fn(signals, axis=0)


def wav_sample(
        data: np.ndarray,
        num_clips: int,
        seed: Optional[int] = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    start_indices = rng.integers(0, len(data) - example_length - 1, num_clips)
    signals = [data[i:i+example_length] for i in start_indices]
    return np.stack(signals, axis=0)

def sisdr(
        estimate: torch.Tensor,
        target: torch.Tensor,
        reduction: Optional[str] = None
) -> torch.Tensor:
    """Calculate single source SI-SDR."""
    return sdr(estimate, target, reduction=reduction, scale_invariant=True)


def sisdr_improvement(
        estimate: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor,
        reduction: Optional[str] = None
) -> torch.Tensor:
    """Calculate estimate to target SI-SDR improvement relative to mixture.
    """
    return sdr_improvement(
        estimate, target, mixture, reduction=reduction, scale_invariant=True)


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
        reduction: Optional[str] = None,
        scale_invariant: bool = False
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
    if not all([f for f in df.filepath if os.path.isfile(f)]):
        raise ValueError(f'Audio files missing, check {dataset_directory}.')

    # reindex and name the dataframe
    df = df[['filepath', 'subset_id', 'speaker_id',
             'split', 'duration', 'sparsity']]
    df = df.reset_index(drop=True)
    df.index.name = 'LIBRISPEECH'
    return df


def dataframe_demand(
        dataset_directory: Union[str, os.PathLike] = _root_demand,
        empty: bool = False,
) -> pd.DataFrame:
    """Creates a Pandas DataFrame with files from the DEMAND corpus.
    Root directory should mimic archive-extracted folder structure.
    Dataset may be downloaded at `<https://www.zenodo.org/record/1227121/>`_.
    """
    if empty:
        return pd.DataFrame(columns=['filepath', 'duration', 'sparsity'])
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
    if not all([f for f in df.filepath if os.path.isfile(f)]):
        raise ValueError(f'Audio files missing, check {dataset_directory}.')

    # reindex and name the dataframe
    df = df[['filepath', 'duration', 'sparsity']]
    df = df.reset_index(drop=True)
    df.index.name = 'DEMAND'
    return df


def dataframe_fsd50k(
        dataset_directory: Union[str, os.PathLike] = _root_fsd50k,
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
    if not all([f for f in df.filepath if os.path.isfile(f)]):
        raise ValueError(f'Audio files missing, check {dataset_directory}.')

    # reindex and name the dataframe
    df = df[['filepath', 'split', 'duration', 'labels', 'sparsity']]
    df = df.reset_index(drop=True)
    df.index.name = 'FSD50K'
    return df


def dataframe_musan(
        dataset_directory: Union[str, os.PathLike] = _root_musan,
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
    if not all([f for f in df.filepath if os.path.isfile(f)]):
        raise ValueError(f'Audio files missing, check {dataset_directory}.')

    # reindex and name the dataframe
    df = df[['filepath', 'split', 'duration', 'sparsity']]
    df = df.reset_index(drop=True)
    df.index.name = 'MUSAN'
    return df


def dataframe_irsurvey(
        dataset_directory: Union[str, os.PathLike] = _root_irsurvey,
        empty: bool = False,
) -> pd.DataFrame:
    """Creates a Pandas DataFrame with files from the MIT Acoustical
    Reverberation Scene Statistics Survey. Root directory should mimic
    archive-extracted folder structure. Dataset may be downloaded at
    `<https://mcdermottlab.mit.edu/Reverb/IR_Survey.html>`_.
    """
    if empty:
        return pd.DataFrame(columns=['filepath', 'split', 'duration', 'frequency'])
    rows = []
    files = sorted(pathlib.Path(dataset_directory).glob('*.wav'))
    if not len(files) == 270:
        raise ValueError(f'Audio files missing, check {dataset_directory}.')

    for filepath in files:
        ir_name = ''.join(filepath.name.split('_')[1:-1])
        ir_duration = wav_read(filepath)[1]
        if filepath.name == 'h053_Office_ConferenceRoom_stxts.wav':
            ir_frequency = 3
        else:
            try:
                ir_frequency = int(
                    filepath.name.split('txt')[0].split('_')[-1])
            except (Exception,):
                ir_frequency = 1
        rows += [(ir_name, ir_frequency, ir_duration, str(filepath))]

    df = pd.DataFrame(
        rows, columns=['name', 'frequency', 'duration', 'filepath'])

    # shuffle the recordings
    df = df.sample(frac=1, random_state=200).reset_index(drop=True)

    # organize by split
    df['split'] = 'train'
    df.loc[int(270 * .8):int(270 * .9), 'split'] = 'val'
    df.loc[int(270 * .9):, 'split'] = 'test'

    # ensure that all the audio files exist
    if not all([f for f in df.filepath if os.path.isfile(f)]):
        raise ValueError(f'Audio files missing, check {dataset_directory}.')

    # reindex and name the dataframe
    df = df[['filepath', 'split', 'duration', 'frequency']]
    df = df.reset_index(drop=True)
    df.index.name = 'IR_SURVEY'
    return df


def dataframe_slr28(
        dataset_directory: Union[str, os.PathLike] = _root_slr28,
        empty: bool = False,
) -> pd.DataFrame:
    """Creates a Pandas DataFrame with files from the 2017 ICASSP paper,
    "A Study on Data Augmentation of Reverberant Speech for Robust Speech
    Recognition". Root directory should mimic archive-extracted folder
    structure. Dataset may be downloaded at `<https://www.openslr.org/28/>`_.
    """
    if empty:
        return pd.DataFrame(columns=['type', 'id', 'room', 'filepath'])
    def parse_rir_list(filepath: str, real: bool = False):
        sublist = []
        fp = pathlib.Path(filepath)
        if not fp.exists():
            fp = pathlib.Path(dataset_directory).joinpath(fp)
            if not fp.exists():
                raise IOError(f'Missing rir_list {str(fp)}.')
        i_type = 'real' if real else 'simulated'
        for line in open(fp).read().splitlines():
            parts = line.split()
            i_id, i_room = str(parts[1]), str(parts[3])
            i_filepath = pathlib.Path(dataset_directory).joinpath(
                str(parts[4]).replace('RIRS_NOISES/', '')
            )
            if not i_filepath.exists():
                raise IOError(f'Missing file {i_filepath}.')
            sublist.append((i_type, i_id, i_room, str(i_filepath)))
        return sublist

    # add the real impulse responses
    rows = parse_rir_list('real_rirs_isotropic_noises/rir_list', real=True)
    df_real = pd.DataFrame(
        rows, columns=['type', 'id', 'room', 'filepath'])

    # shuffle the recordings
    df_real = df_real.sample(
        frac=1, random_state=0).reset_index(drop=True)
    nrows = len(df_real)

    # organize by split
    df_real['split'] = 'train'
    df_real.loc[int(nrows * .8):int(nrows * .9), 'split'] = 'val'
    df_real.loc[int(nrows * .9):, 'split'] = 'test'

    # add the synthetic impulse responses
    rows = []
    rows += parse_rir_list('simulated_rirs/smallroom/rir_list')
    rows += parse_rir_list('simulated_rirs/mediumroom/rir_list')
    rows += parse_rir_list('simulated_rirs/largeroom/rir_list')
    df_synth = pd.DataFrame(
        rows, columns=['type', 'id', 'room', 'filepath'])

    # shuffle the recordings
    df_synth = df_synth.sample(
        frac=1, random_state=0).reset_index(drop=True)
    nrows = len(df_synth)

    # organize by split
    df_synth['split'] = 'train'
    df_synth.loc[int(nrows * .8):int(nrows * .9), 'split'] = 'val'
    df_synth.loc[int(nrows * .9):, 'split'] = 'test'

    # combine real and synthetic
    df = pd.concat((df_real, df_synth))

    # reindex and name the dataframe
    df = df.reset_index(drop=True)
    df.index.name = 'SLR28'
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
            split_reverb: Optional[str] = None,
            frac_speech: Optional[float] = 1.,
            corpus_premixture: str = 'fsd50k',
            corpus_mixture: str = 'musan',
            snr_premixture: Optional[Union[float, Tuple[float, float]]] = None,
            snr_mixture: Optional[Union[float, Tuple[float, float]]] = None,
            dataset_duration: Union[int, float] = 0
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
        self.frac_speech = frac_speech
        self.speaker_ids_repr = repr(self.speaker_ids)
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
        if split_reverb is not None:
            if not (split_reverb in ('train', 'val', 'test')):
                raise ValueError('Expected `split_reverb` to be either '
                                 '"train", "val", or "test".')
        self.split_speech = split_speech
        self.split_premixture = split_premixture or ''
        self.split_mixture = split_mixture or ''
        self.split_reverb = split_reverb or ''

        # verify dataset duration
        if not isinstance(dataset_duration, (int, float, type(None))):
            raise ValueError('Expected `dataset_duration` to be a number.')
        self.dataset_duration = int(dataset_duration or 0)

        self.index = 0
        self.example_duration = example_duration

        # instantiate corpora
        self.corpus_s = df_librispeech.query(
            f'speaker_id in {self.speaker_ids}')
        if self.split_speech != 'all':
            self.corpus_s = self.corpus_s.query(
                f'split == "{self.split_speech}"')
        if 0 < self.frac_speech < 1:
            self.corpus_s = self.corpus_s.sample(
                frac=frac_speech, random_state=0)
            print('Length of subsampled dataset:', len(self.corpus_s))
        self.corpus_m = get_noise_corpus(corpus_premixture).query(
            f'split == "{self.split_premixture}"')
        self.corpus_n = get_noise_corpus(corpus_mixture).query(
            f'split == "{self.split_mixture}"')
        self.corpus_r = df_irsurvey.query(
            f'split == "{self.split_reverb}"')

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
        self.len_r = len(self.corpus_r)
        if self.len_s < 1:
            raise ValueError('Invalid speaker_id')

        # if a dataset duration is provided,
        # truncate the audio data to the expected size
        self.speech_data = np.array([])
        if self.dataset_duration:
            self.speech_data = wav_read_multiple(
                self.corpus_s.filepath, concatenate=True)
            self.speech_data = self.speech_data[:(
                    self.dataset_duration * sample_rate)]

        # define flags
        self.is_personalized = bool(len(self.speaker_ids) == 1)
        self.add_premixture_noise = bool(
            (snr_premixture is not None) and (self.len_m > 0))
        self.add_noise = bool(
            (snr_mixture is not None) and (self.len_n > 0))
        self.add_reverb = bool(self.len_r > 0)

        if not self.is_personalized and self.add_premixture_noise:
            raise ExperimentError('Non-personalized dataset contains '
                                  'premixture noise.')

        if self.dataset_duration and self.add_premixture_noise:
            raise ExperimentError('Fine-tuning dataset contains '
                                  'premixture noise.')

    def __dict__(self):
        return {
            'flags': {
                'is_personalized': self.is_personalized,
                'add_premixture_noise': self.add_premixture_noise,
                'add_noise': self.add_noise,
            },
            'speaker_ids': self.speaker_ids_repr,
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
        s_filepaths = (list(self.corpus_s.filepath.iloc[indices % self.len_s])
                       if self.len_s else [])
        m_filepaths = (list(self.corpus_m.filepath.iloc[indices % self.len_m])
                       if self.len_m else [])
        n_filepaths = (list(self.corpus_n.filepath.iloc[indices % self.len_n])
                       if self.len_n else [])
        r_filepaths = (list(self.corpus_r.filepath.iloc[indices % self.len_r])
                       if self.len_r else [])

        if self.speech_data.size > 0:
            s = wav_sample(self.speech_data, batch_size, seed=seed)
        else:
            s = wav_read_multiple(s_filepaths, seed=seed)
        x = p = s

        pre_snrs = np.array([])
        if self.add_premixture_noise:
            m = wav_read_multiple(m_filepaths, seed=seed)
            pre_snrs = tmp_rng.uniform(
                self.snr_premixture_min, self.snr_premixture_max,
                (batch_size, 1))
            x = p = mix_signals(s, m, pre_snrs)

        if self.add_reverb:
            r = wav_read_multiple(r_filepaths, randomly_offset=False, seed=seed)
            p_rev = np.empty_like(p)
            p_len = p.shape[-1]
            for i, filt in enumerate(r):
                p_rev[i] = convolve(p[i], filt, mode='full')[:p_len]
            x = p = p_rev

        post_snrs = np.array([])
        if self.add_noise:
            n = wav_read_multiple(n_filepaths, seed=seed)
            post_snrs = tmp_rng.uniform(
                self.snr_mixture_min, self.snr_mixture_max,
                (batch_size, 1))
            x = mix_signals(p, n, post_snrs)

        scale_factor = float(np.abs(x).max() + _eps)
        return Batch(
            inputs=torch.cuda.FloatTensor(x) / scale_factor,
            targets=torch.cuda.FloatTensor(p) / scale_factor,
            pre_snrs=torch.cuda.FloatTensor(pre_snrs),
            post_snrs=torch.cuda.FloatTensor(post_snrs)
        )


class ContrastiveMixtures(Mixtures):

    def __call__(
            self,
            batch_size: int,
            ratio_positive: float = 0.5,
            seed: Optional[int] = None
    ):
        if not (0 <= ratio_positive <= 1):
            raise ValueError('ratio_positive should be between 0 and 1.')
        if batch_size < 2:
            raise ValueError('batch_size must be at least 2.')
        if batch_size % 2:
            raise ValueError('batch_size must be an even number.')

        if seed is None: self.index += 1
        tmp_index: int = 0 if seed is not None else self.index
        tmp_rng: Generator = np.random.default_rng(tmp_index)

        indices = np.arange(batch_size * tmp_index,
                            batch_size * (tmp_index + 1))
        s_filepaths = (list(self.corpus_s.filepath.iloc[indices % self.len_s])
                       if self.len_s else [])
        m_filepaths = (list(self.corpus_m.filepath.iloc[indices % self.len_m])
                       if self.len_m else [])
        n_filepaths = (list(self.corpus_n.filepath.iloc[indices % self.len_n])
                       if self.len_n else [])
        r_filepaths = (list(self.corpus_r.filepath.iloc[indices % self.len_r])
                       if self.len_r else [])

        ordering = tmp_rng.permutation(batch_size//2)
        num_positive = int(batch_size//2 * ratio_positive)
        num_negative = batch_size//2 - num_positive
        labels = np.array([1]*num_positive + [0]*num_negative)

        bx_1, bx_2, bp_1, bp_2, bs_1, bs_2 = [], [], [], [], [], []
        bpre_snrs, bpost_snrs = [], []

        # generate pairs
        for i in range(0, batch_size, 2):

            is_positive = bool(i/2 < num_positive)

            if self.speech_data.size > 0:
                if is_positive:
                    s_1 = s_2 = wav_sample(self.speech_data, 1, seed=seed)
                else:
                    s_1, s_2 = wav_sample(self.speech_data, 2, seed=seed)
            else:
                if is_positive:
                    s_1 = s_2 = wav_read_multiple([s_filepaths[i]], seed=seed)
                else:
                    s_1, s_2 = wav_read_multiple(s_filepaths[i:i+2], seed=seed)

            s_1, s_2 = s_1.reshape(-1), s_2.reshape(-1)

            p_1, p_2 = s_1, s_2
            pre_snr = [None, None]
            if self.add_premixture_noise:
                if is_positive:
                    m_1 = m_2 = wav_read_multiple([m_filepaths[i]], seed=seed)
                    pre_snr = [tmp_rng.uniform(
                        self.snr_premixture_min, self.snr_premixture_max)] * 2
                else:
                    m_1, m_2 = wav_read_multiple(m_filepaths[i:i+2], seed=seed)
                    pre_snr = tmp_rng.uniform(
                        self.snr_premixture_min, self.snr_premixture_max, 2)
                m_1, m_2 = m_1.reshape(-1), m_2.reshape(-1)
                p_1 = mix_signals(s_1, m_1, pre_snr[0])
                p_2 = mix_signals(s_2, m_2, pre_snr[1])

            if self.add_reverb:
                if is_positive:
                    r_1 = r_2 = wav_read_multiple([r_filepaths[i]], seed=seed)
                else:
                    r_1, r_2 = wav_read_multiple(r_filepaths[i:i+2], seed=seed)
                r_1, r_2 = r_1.reshape(-1), r_2.reshape(-1)
                p_len = p_1.shape[-1]
                p_1 = convolve(p_1, r_1, mode='full')[:p_len]
                p_2 = convolve(p_2, r_2, mode='full')[:p_len]

            x_1, x_2 = p_1, p_2
            post_snr = [None, None]
            if self.add_noise:
                if not is_positive:
                    n_1 = n_2 = wav_read_multiple([n_filepaths[i]], seed=seed)
                    post_snr = [tmp_rng.uniform(
                        self.snr_mixture_min, self.snr_mixture_max)] * 2
                else:
                    n_1, n_2 = wav_read_multiple(n_filepaths[i:i+2], seed=seed)
                    post_snr = tmp_rng.uniform(
                        self.snr_mixture_min, self.snr_mixture_max, 2)
                n_1, n_2 = n_1.reshape(-1), n_2.reshape(-1)
                x_1 = mix_signals(p_1, n_1, post_snr[0])
                x_2 = mix_signals(p_2, n_2, post_snr[1])

            bp_1.append(p_1)
            bp_2.append(p_2)
            bx_1.append(x_1)
            bx_2.append(x_2)
            if pre_snr[0]:
                bpre_snrs.append(pre_snr)
            if post_snr[0]:
                bpost_snrs.append(post_snr)

        # stack and shuffle all the data in the right order
        bp_1 = np.stack(bp_1)[ordering]
        bp_2 = np.stack(bp_2)[ordering]
        bx_1 = np.stack(bx_1)[ordering]
        bx_2 = np.stack(bx_2)[ordering]
        if bpre_snrs:
            bpre_snrs = np.stack(bpre_snrs)[ordering]
        if bpost_snrs:
            bpost_snrs = np.stack(bpost_snrs)[ordering]
        labels = labels[ordering]

        scale_factor_1 = float(np.abs(bx_1).max() + _eps)
        scale_factor_2 = float(np.abs(bx_2).max() + _eps)
        scale_factor = max([scale_factor_1, scale_factor_2])
        return ContrastiveBatch(
            inputs_1=torch.cuda.FloatTensor(bx_1) / scale_factor,
            inputs_2=torch.cuda.FloatTensor(bx_2) / scale_factor,
            targets_1=torch.cuda.FloatTensor(bp_1) / scale_factor,
            targets_2=torch.cuda.FloatTensor(bp_2) / scale_factor,
            labels=torch.cuda.BoolTensor(labels),
            pre_snrs=torch.cuda.FloatTensor(bpre_snrs),
            post_snrs=torch.cuda.FloatTensor(bpost_snrs)
        )


# expose corpora and speaker lists
df_librispeech = dataframe_librispeech()
df_musan = dataframe_musan()
df_fsd50k = dataframe_fsd50k()
df_demand = dataframe_demand(empty=True)
df_irsurvey = dataframe_irsurvey(empty=True)
df_slr28 = dataframe_slr28(empty=True)
speaker_ids_tr, speaker_ids_vl, speaker_ids_te = split_speakers(False)
speaker_ids_all = speaker_ids_tr + speaker_ids_vl + speaker_ids_te
speaker_split_durations = df_librispeech.groupby(
    ['speaker_id', 'split']).agg('sum').duration

# expose test sets
data_te_generalist: Mixtures = Mixtures(
    speaker_ids_te, 'test', split_mixture='test', snr_mixture=(-5, 5)
)
data_te_specialist: List[Mixtures] = [
    Mixtures(speaker_id, 'test', corpus_mixture='fsd50k',
             split_mixture='test', snr_mixture=(-5, 5))
    for speaker_id in speaker_ids_te
]
