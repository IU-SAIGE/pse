import itertools
import os
import pathlib
import warnings
from datetime import datetime
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

import exp_data as exd
import exp_models as exm

warnings.filterwarnings('ignore')

_batch_size: int = 8  # change this depending on GPU limitations
_val_batch_size: int = 100

_rng = np.random.default_rng(0)


class EarlyStopping(Exception):
    pass


def get_personalized_dataset(
        speaker_id: Union[int, str],
        environment_snr_db: Union[float, Tuple[float, float]],
        num_environments: int = 1):
    # build test set
    rng = np.random.default_rng(0)

    # retrieve datasets
    S_te = exd.dataframe_librispeech().query('subset_id == "test-clean"')
    S_te = S_te[['speaker_id', 'filepath', 'duration']]
    test_speakers = S_te['speaker_id'].astype(str).unique()
    test_speaker = str(speaker_id)
    if test_speaker not in test_speakers:
        raise ValueError(f'Invalid test speaker ID: {test_speaker}.')
    M = exd.dataframe_demand()

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
    split_te = (utterance_list['cumsum'] - 30).abs().idxmin()
    split_vl = (utterance_list['cumsum'] - 60).abs().idxmin()
    split_ft = (utterance_list['cumsum'] - 120).abs().idxmin()
    u_te = utterance_list.iloc[0:split_te]
    u_vl = utterance_list.iloc[split_te:split_vl]
    u_ft = utterance_list.iloc[split_vl:split_ft]
    u_tr = utterance_list.iloc[split_ft:]

    # load all the speaker audio and concatenate it into a
    # single long vector
    s_te = np.concatenate([exd.wav_read(f) for f in u_te.filepath])
    s_vl = np.concatenate([exd.wav_read(f) for f in u_vl.filepath])
    s_ft = np.concatenate([exd.wav_read(f) for f in u_ft.filepath])
    s_tr = np.concatenate([exd.wav_read(f) for f in u_tr.filepath])
    s = np.concatenate([s_te, s_vl, s_ft, s_tr])

    # load the premixture noise profile
    env_indices = env_mapping[test_speaker]
    if num_environments == 1:
        p = exd.wav_read(M.iloc[env_indices]['filepath'])
    else:
        p = np.stack([exd.wav_read(f) for f in
                      M.iloc[env_indices]['filepath']])

    # segment the premixture noise similarly
    # (note that there is no 'finetuning' premixture noise)
    split_te = len(s_te)
    split_vl = split_te + len(s_vl)
    split_ft = split_vl + len(s_ft)
    p_te = p[..., 0:split_te]
    p_vl = p[..., split_te:split_vl]
    p_tr = p[..., split_vl:]

    # circularly loop the training premixture noise if it is
    # too short
    tile_count = -(-s_tr.shape[-1] // p_tr.shape[-1])
    if num_environments == 1:
        p_tr = np.tile(p_tr, tile_count)
    else:
        p_tr = np.tile(p_tr, (1, tile_count))
    p_tr = p_tr[..., :len(s_tr)]
    p = np.concatenate([p_te, p_vl, p_tr], axis=-1)

    # get the correct mixing scalar based on the overall
    # energies of the speech and premixture datasets
    energy_s = np.sum(s ** 2, axis=-1, keepdims=True)
    energy_n = np.sum(p ** 2, axis=-1, keepdims=True)
    snr_db = np.mean(environment_snr_db)
    b = np.sqrt((energy_s / energy_n) * (10 ** (-snr_db / 10.)))

    # combine all the signals
    x_tr = s_tr + b * p_tr
    x_vl = s_vl + b * p_vl
    x_te = s_te + b * p_te

    if num_environments == 1:
        environments = [M.iloc[env_indices]['location_id']]
    else:
        environments = M.iloc[env_indices]['location_id']

    return {
        'train_speech': s_tr,
        'train_prenoise': p_tr,
        'train_premixture': x_tr,
        'finetune_speech': s_ft,
        'val_speech': s_vl,
        'val_prenoise': p_vl,
        'val_premixture': x_vl,
        'test_speech': s_te,
        'test_prenoise': p_te,
        'test_premixture': x_te,
        'environments': {i: j for (i, j) in enumerate(
            environments)}
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
        s_te = torch.from_numpy(dataset['test_speech']).float().cuda()
        x_te = torch.from_numpy(dataset['test_premixture']).float().cuda()
        if len(x_te.shape) == 2 and len(s_te.shape) == 1:
            s_te = torch.stack([s_te for _ in range(num_environments)], dim=0)
        elif len(x_te.shape) == 1 and len(s_te.shape) == 1:
            s_te = s_te.unsqueeze(0)
            x_te = x_te.unsqueeze(0)
        s_hat_te = model(x_te)
        return float(exd.sisdr_improvement(s_hat_te, s_te, x_te).mean())

    if test_speaker == None:
        S_te = exd.dataframe_librispeech().query('subset_id == "test-clean"')
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
        config = yaml.safe_load(fp)
    model = exm.init_model(config)[0]
    checkpoint_obj = torch.load(checkpoint_path)
    model_state_dict = checkpoint_obj['model_state_dict']
    model.load_state_dict(model_state_dict, strict=True)
    model.cuda()
    return test_model(model, test_speaker, environment_snr_db, num_environments)


def train_sup(config: dict, checkpoint_path: Optional[str] = None,
              patience: int = 300000):
    """Train a fully-supervised speech enhancement model."""

    # verify config
    expected_config = {
        'learning_rate': float,
        'model_name': str,
        'model_size': str,
        'snr_mixture_min': float,
        'snr_mixture_max': float,
    }
    if not set(expected_config.keys()).issubset(set(config.keys())):
        raise ValueError(f'Expected `config` to contain keys: '
                         f'{set(expected_config.keys())}')
    config['batch_size'] = config.get('batch_size', _batch_size)
    config['val_batch_size'] = config.get('val_batch_size', _val_batch_size)
    config['sample_rate'] = config.get('sample_rate', exd.sample_rate)
    config['example_duration'] = config.get('example_duration',
                                            exd.example_duration)

    # prepare neural net, optimizer, and loss function
    model, model_nparams, model_config = exm.init_model(config)
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
    S = exd.dataframe_librispeech()
    S_tr = S.query('subset_id == "train-clean-100"')
    S_vl = S.query('subset_id == "dev-clean"')
    N = exd.dataframe_fsd50k()
    N_tr = N.query('split == "train"')
    N_vl = N.query('split == "val"')

    # instantiate tensorboard
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    output_directory = pathlib.Path('runs').joinpath(
        current_time + '_' + trial_name(config=config))
    writer = SummaryWriter(str(output_directory))
    with open(output_directory.joinpath('config.yaml'), 'w',
              encoding='utf-8') as fp:
        yaml.dump(config, fp)
        print(yaml.dump(config, default_flow_style=False))

    # keep track of the minimum loss (to early stop)
    min_loss, min_loss_step = np.inf, 0

    # training loop
    step: int = init_step
    print(f'Output Directory: {output_directory}')
    try:
        for step in itertools.count(init_step):

            model.train()
            with torch.set_grad_enabled(True):

                # circularly index the datasets
                indices = np.arange(_batch_size * step,
                                    _batch_size * (step + 1), 1)
                s = exd.wav_read_multiple(S_tr.filepath[indices % len(S_tr)])
                n = exd.wav_read_multiple(N_tr.filepath[indices % len(N_tr)])

                # mix the signals up at random snrs
                snrs = _rng.uniform(low=config['snr_mixture_min'],
                                    high=config['snr_mixture_max'],
                                    size=(_batch_size, 1))
                x = exd.mix_signals(s, n, snrs)

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
                            sisdri_tr += exd.sisdr_improvement(s_hat, _s, _x).mean()
                    sisdri_tr /= _batch_size
                else:
                    s_hat = model(x)
                    if len(s_hat.shape) == 3:
                        s_hat = s_hat[:, 0]
                    loss_tr = criterion(s_hat, s).mean()
                    loss_tr.backward()
                    with torch.no_grad():
                        sisdri_tr = exd.sisdr_improvement(s_hat, s, x).mean()

                # back propagation
                optimizer.step()
                optimizer.zero_grad()

                # write summaries
                writer.add_scalar('MSELoss/train', float(loss_tr), step)
                writer.add_scalar('SISDRi/train', float(sisdri_tr), step)

            if (step % config.get('validate_every', 100)):
                continue

            model.eval()
            with torch.no_grad():

                s = exd.wav_read_multiple(S_vl.filepath[0:_val_batch_size])
                n = exd.wav_read_multiple(N_vl.filepath[0:_val_batch_size])
                x = exd.mix_signals(s, n, float(np.mean(config['mixture_snr'])))
                loss_vl, sisdri_vl = [], []
                for i in range(0, _val_batch_size, _batch_size):
                    _s = torch.from_numpy(s[i:i + _batch_size]).float().cuda()
                    _x = torch.from_numpy(x[i:i + _batch_size]).float().cuda()
                    s_hat = model(_x)
                    if len(s_hat.shape) == 3:
                        s_hat = s_hat[:, 0]
                    loss_vl.append(float(criterion(s_hat, _s).mean()))
                    sisdri_vl.append(float(exd.sisdr_improvement(s_hat, _s, _x).mean()))
                loss_vl, sisdri_vl = np.mean(loss_vl), np.mean(sisdri_vl)
                writer.add_scalar('MSELoss/validation', float(loss_vl), step)
                writer.add_scalar('SISDRi/validation', float(sisdri_vl), step)

                # checkpoint whenever validation score improves
                if loss_vl < min_loss:
                    min_loss = loss_vl
                    min_loss_step = step
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, output_directory.joinpath(f'ckpt_{step:08}.pt'))
                    with open(output_directory.joinpath(f'best_step.txt'), 'w') as fp:
                        print(step, file=fp)

                if (step - min_loss_step) > patience:
                    raise EarlyStopping()

    except EarlyStopping as e:
        print(f'Automatically exited with patience for {patience} steps; '
              f'best step was {min_loss_step}.')
        pass

    except KeyboardInterrupt as e:
        print(f'Manually exited at step {step}; '
              f'best step was {min_loss_step}.')
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, output_directory.joinpath(f'ckpt_last.pt'))
        pass

    # close the summary
    writer.close()

    # print the location of the checkpoints
    print(f'Saved checkpoints to {output_directory}.')

    # exit the trainer
    return


def train_unsup(config: dict, checkpoint_path: Optional[str] = None,
                patience: int = 300000):
    """Train a self-supervised pseudo speech enhancement model."""

    # verify config
    expected_config = {
        'test_speaker': Union[int, str],
        'environment_snr_db': Union[float, Tuple[float, float]],
        'num_environments': int,
        'learning_rate': float,
        'use_contrastive_loss': bool,
        'use_purification_loss': bool,
        'model_name': str,
        'model_size': str,
        'mixture_snr': Tuple[float, float],
    }
    if not set(expected_config.keys()).issubset(set(config.keys())):
        raise ValueError(f'Expected `config` to contain keys: '
                         f'{set(expected_config.keys())}')
    config['batch_size'] = config.get('batch_size', _batch_size)
    config['val_batch_size'] = config.get('val_batch_size', _val_batch_size)
    config['sample_rate'] = config.get('sample_rate', exd.sample_rate)
    config['example_duration'] = config.get('example_duration',
                                            exd.example_duration)

    # prepare neural net, optimizer, and loss function
    model, model_nparams, model_config = exm.init_model(config)
    model = model.cuda()
    config['model_config'] = model_config
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config['learning_rate'])
    criterion = torch.nn.MSELoss(reduction='mean')

    # instantiate SNR predictor and segmental loss function if needed
    if config['use_purification_loss']:
        criterion = exm.SegmentalLoss('mse', reduction='mean')
        predictor = exm.SNRPredictor()
        predictor.load_state_dict(torch.load('snr_predictor'), strict=False)
        predictor.cuda()
        predictor.eval()

    # load a previous checkpoint if provided
    init_step = 0
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        init_step = ckpt['step']

    # load personalized dataset
    dataset = get_personalized_dataset(
        config['test_speaker'],
        config['environment_snr_db'],
        config['num_environments'])
    config['environments'] = dataset['environments']
    P_tr = dataset['train_premixture']
    if len(P_tr.shape) == 1:
        P_tr = P_tr[np.newaxis, ...]
    P_vl = dataset['val_premixture']
    if len(P_vl.shape) == 1:
        P_vl = P_vl[np.newaxis, ...]
    N = exd.dataframe_fsd50k()
    N_tr = N.query('split == "train"')
    N_vl = N.query('split == "val"')

    def sample_premixture(data, size: int = config['batch_size']):
        l = int(config['sample_rate'] * config['example_duration'])
        env_indices = _rng.integers(
            low=0,
            high=config['num_environments'],
            size=size)
        starting_sample_indices = _rng.integers(
            low=0,
            high=int(data.shape[-1] - l - 1),
            size=size)
        s = []
        for j in range(size):
            start = int(starting_sample_indices[j])
            s.append(data[env_indices[j], start:start + l])
        return np.stack(s)

    # instantiate tensorboard
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    output_directory = pathlib.Path('runs').joinpath(
        current_time + '_' + trial_name(config=config))
    writer = SummaryWriter(str(output_directory))
    with open(output_directory.joinpath('config.yaml'), 'w',
              encoding='utf-8') as fp:
        yaml.dump(config, fp)
        print(yaml.dump(config, default_flow_style=False))

    # keep track of the minimum loss (to early stop)
    min_loss, min_loss_step = np.inf, 0

    # training loop
    print(f'Output Directory: {output_directory}')
    try:
        for step in itertools.count(init_step):

            model.train()
            with torch.set_grad_enabled(True):

                # sample speech from the premixture data
                s = sample_premixture(P_tr)
                indices = np.arange(config['batch_size'] * step,
                                    config['batch_size'] * (step + 1), 1)
                n = exd.wav_read_multiple(N_tr.filepath[indices % len(N_tr)])

                # mix the signals up at random snrs
                snrs = _rng.uniform(low=config['mixture_snr'][0],
                                    high=config['mixture_snr'][1],
                                    size=(config['batch_size'], 1))
                x = exd.mix_signals(s, n, snrs)

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
                        if config['use_purification_loss']:
                            w = predictor(_s)
                            loss_tr = criterion(s_hat, _s, w).mean()
                        else:
                            loss_tr = criterion(s_hat, _s).mean()
                        (loss_tr / _batch_size).backward()
                        with torch.no_grad():
                            sisdri_tr += exd.sisdr_improvement(s_hat, _s, _x).mean()
                    sisdri_tr /= _batch_size
                else:
                    s_hat = model(x)
                    if len(s_hat.shape) == 3:
                        s_hat = s_hat[:, 0]
                    if config['use_purification_loss']:
                        w = predictor(s)
                        loss_tr = criterion(s_hat, s, w).mean()
                    else:
                        loss_tr = criterion(s_hat, s).mean()
                    loss_tr.backward()
                    with torch.no_grad():
                        sisdri_tr = exd.sisdr_improvement(s_hat, s, x).mean()

                # back propagation
                optimizer.step()
                optimizer.zero_grad()

                # write summaries
                writer.add_scalar('MSELoss/train', float(loss_tr), step)
                writer.add_scalar('SISDRi/train', float(sisdri_tr), step)

            if step % config.get('validate_every', 100):
                continue

            model.eval()
            with torch.no_grad():

                s = sample_premixture(P_vl, _val_batch_size)
                n = exd.wav_read_multiple(N_vl.filepath[0:_val_batch_size])
                x = exd.mix_signals(s, n, float(np.mean(config['mixture_snr'])))
                loss_vl, sisdri_vl = [], []
                for i in range(0, _val_batch_size, _batch_size):
                    _s = torch.from_numpy(s[i:i + _batch_size]).float().cuda()
                    _x = torch.from_numpy(x[i:i + _batch_size]).float().cuda()
                    s_hat = model(_x)
                    if len(s_hat.shape) == 3:
                        s_hat = s_hat[:, 0]
                    loss_vl.append(float(criterion(s_hat, _s).mean()))
                    sisdri_vl.append(float(exd.sisdr_improvement(s_hat, _s, _x).mean()))
                loss_vl, sisdri_vl = np.mean(loss_vl), np.mean(sisdri_vl)
                writer.add_scalar('MSELoss/validation', float(loss_vl), step)
                writer.add_scalar('SISDRi/validation', float(sisdri_vl), step)

                # checkpoint whenever validation score improves
                if loss_vl < min_loss:
                    min_loss = loss_vl
                    min_loss_step = step
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, output_directory.joinpath(f'ckpt_{step:08}.pt'))
                    with open(output_directory.joinpath(f'best_step.txt'), 'w') as fp:
                        print(step, file=fp)

                if (step - min_loss_step) > patience:
                    raise EarlyStopping()

    except EarlyStopping:
        print(f'Automatically exited with patience for {patience} steps; '
              f'best step was {min_loss_step}.')
        pass

    except KeyboardInterrupt:
        print(f'Manually exited at step {step}; '
              f'best step was {min_loss_step}.')
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, output_directory.joinpath(f'ckpt_last.pt'))
        pass

    # close the summary
    writer.close()

    # print the location of the checkpoints
    print(f'Saved checkpoints to {output_directory}.')

    # exit the trainer
    return


def trial_name(trial=None, config: Optional[dict] = None):
    if trial:
        config = trial.config
    elif not config:
        raise ValueError('Either `trial` or `config` must be set.')
    if 'unsup' in config['training_procedure']:
        name = '{}_sp{}_env{}_psnr{}_{}_{}'.format(
            config['training_procedure'],
            str(config['test_speaker']),
            str(int(config['num_environments'])),
            str(int(np.mean(config['environment_snr_db']))),
            config['model_name'],
            config['model_size'][0].upper()
        )
    else:
        name = '{}_{}_{}'.format(
            config['training_procedure'],
            config['model_name'],
            config['model_size'][0].upper()
        )
    return name


if __name__ == '__main__':
    train_sup(dict(
        learning_rate=1e-4,
        model_name='grunet',
        model_size='small',
        training_procedure='sup',
        snr_mixture_min=-10,
        snr_mixture_max=10,
    ))
    # S_te = exd.dataframe_librispeech().query('subset_id == "test-clean"')
    # test_speakers = sorted(S_te['speaker_id'].astype(str).unique())
    # for test_speaker in test_speakers:
    #     train_unsup(dict(
    #         learning_rate=1e-4,
    #         model_name='grunet',
    #         model_size='small',
    #         training_procedure='unsup+dp',
    #         use_contrastive_loss=False,
    #         use_purification_loss=True,
    #         test_speaker=test_speaker,
    #         num_environments=1,
    #         environment_snr_db=5,
    #         mixture_snr=(-10, 10)
    #     ))
