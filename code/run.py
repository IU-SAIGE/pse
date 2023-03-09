import argparse
import copy
import itertools
import json
import os
import socket
import sys
import time
import warnings
from ast import literal_eval
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Optional, List, Sequence
from typing import Union

import asteroid.losses
import numpy as np
import torch
import yaml
from pytorch_lightning import seed_everything
from ray import tune
from torch.utils.tensorboard import SummaryWriter

from exp_data import ContrastiveMixtures, Mixtures
from exp_data import example_duration, sample_rate
from exp_data import speaker_ids_tr, speaker_ids_vl, speaker_ids_te
from exp_models import SegmentalLoss, SNRPredictor, init_model, load_checkpoint, \
    test_denoiser_with_speaker
from exp_models import contrastive_feedforward, feedforward
from exp_utils import EarlyStopping, ExperimentError, SmokeTest

warnings.filterwarnings('ignore')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

_host = str(socket.gethostname().split('.')[-3:].pop(0))
_snrp_path = Path(__file__).resolve().parent.joinpath('snr_predictor')
_tune_kwargs = dict(
    reuse_actors=True,
    log_to_file=True,
    local_dir='.',
    fail_fast=True,
    verbose=1
)

def save_config(
        output_directory: Union[str, os.PathLike],
        config: dict
):
    """Saves the config dict to file."""
    output_directory = Path(output_directory)
    with open(output_directory.joinpath('config.json'), 'w',
              encoding='utf-8') as fp:
        json.dump(config, fp, indent=2, sort_keys=True)
        print(yaml.safe_dump(config, default_flow_style=False))


# noinspection PyTypeChecker
def train_denoiser(
        model_name: str,
        model_size: str,
        data_tr: Mixtures,
        data_vl: Mixtures,
        use_loss_purification: bool = False,
        lambda_p: float = 1.,
        lambda_n: float = 1.,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        checkpoint_path: Optional[str] = None,
        num_examples_validation: int = 1000,
        num_examples_minimum: int = 100000,
        num_examples_earlystopping: int = 100000,
        trial_name: Optional[str] = None,
        output_folder: Union[str, os.PathLike] = f'trials_{_host}',
        early_stopping_metric: str = 'sisdri',
        distance_func: str = 'mse',
        called_by_ray: bool = False,
        run_smoke_test: bool = False
) -> str:

    seed_everything(0)

    # prepare model, optimizer, and loss function
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    model, nparams, model_config = init_model(model_name, model_size)
    model = model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    predictor = torch.nn.Identity()
    if use_loss_purification:
        predictor = SNRPredictor()
        predictor.load_state_dict(torch.load(str(_snrp_path)), strict=False)
        predictor.cuda()
        predictor.eval()

    use_loss_contrastive: bool = bool(isinstance(data_tr, ContrastiveMixtures))
    if not type(data_tr) is type(data_vl):
        raise ValueError('`data_tr` and `data_vl` should be the same type.')

    # load a previous checkpoint if provided
    init_num_examples = 0
    output_directory: Optional[Path] = None
    is_finetuning = bool(data_tr.dataset_duration or 0)
    if checkpoint_path:
        # reuse output directory (to pick up experiment where left off)
        output_directory = Path(checkpoint_path).parent
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        # if finetuning a generalist, make a subdirectory
        if is_finetuning:
            output_directory = output_directory.joinpath(
                current_time + '_ft_' + trial_name)
        # otherwise, resuming training so reuse the old optimizer
        else:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        init_num_examples = ckpt['num_examples']

    # define experiment configuration
    config = {
        'batch_size': batch_size,
        'checkpoint_path': str(checkpoint_path or ''),
        'data_tr': data_tr.__dict__(),
        'data_vl': data_vl.__dict__(),
        'distance_func': distance_func,
        'example_duration': example_duration,
        'lambda_p': lambda_p,
        'lambda_n': lambda_n,
        'learning_rate': learning_rate,
        'model_config': model_config,
        'model_name': model_name,
        'model_nparams': nparams,
        'model_size': model_size,
        'num_examples_minimum': num_examples_minimum,
        'num_examples_earlystopping': num_examples_earlystopping,
        'num_examples_validation': num_examples_validation,
        'sample_rate': sample_rate,
        'speaker_ids': data_tr.speaker_ids_repr,
        'use_loss_contrastive': use_loss_contrastive,
        'use_loss_purification': use_loss_purification,
        'early_stopping_metric': early_stopping_metric,
        'is_finetuning': is_finetuning
    }

    # instantiate tensorboard
    if called_by_ray:
        trial_name = tune.get_trial_name()
    if output_directory is None:
        output_directory = Path(output_folder).joinpath(
            current_time + '_' + trial_name)
    writer = SummaryWriter(str(output_directory))
    save_config(output_directory, config)

    # begin training (use gradient accumulation for TasNet models)
    num_examples: int = init_num_examples
    num_validations: int = ceil(num_examples / num_examples_validation)
    best_score: float = np.inf * (1 if early_stopping_metric == 'loss' else -1)
    best_score_step: int = init_num_examples
    use_gradient_accumulation: bool = not bool('grunet' in model_name)
    print(f'Output Directory: {str(output_directory)}')

    # define the distance function
    if distance_func == 'snr':
        distfunc_reg = asteroid.losses.sdr.SingleSrcNegSDR('snr')
        distfunc_segm = SegmentalLoss('snr', reduction='none')
    elif distance_func == 'sisdr':
        distfunc_reg = asteroid.losses.sdr.SingleSrcNegSDR('sisdr')
        distfunc_segm = SegmentalLoss('sisdr', reduction='none')
    else:
        distfunc_reg = torch.nn.MSELoss(reduction='none')
        distfunc_segm = SegmentalLoss('mse', reduction='none')

    try:
        for num_examples in itertools.count(start=init_num_examples,
                                            step=batch_size):
            model.train()
            if use_loss_contrastive:

                # pick up a training batch
                batch = data_tr(batch_size)
                x_1 = batch.inputs_1.cuda()
                x_2 = batch.inputs_2.cuda()
                p_1 = batch.targets_1.cuda()
                p_2 = batch.targets_2.cuda()

                # estimate data purification weights
                w_1 = predictor(p_1) if use_loss_purification else None
                w_2 = predictor(p_2) if use_loss_purification else None

                # forward propagation
                metrics_tr = contrastive_feedforward(
                    inputs_1=x_1, inputs_2=x_2,
                    targets_1=p_1, targets_2=p_2,
                    weights_1=w_1, weights_2=w_2,
                    lambda_positive=lambda_p, lambda_negative=lambda_n,
                    loss_reg=distfunc_reg, loss_segm=distfunc_segm,
                    labels=batch.labels.cuda(),
                    model=model.cuda(),
                    accumulation=use_gradient_accumulation,
                    validation=False)

            else:

                # pick up a training batch
                batch = data_tr(batch_size)
                x = batch.inputs.cuda()
                p = batch.targets.cuda()

                # estimate data purification weights
                w = predictor(p) if use_loss_purification else None

                # forward propagation
                metrics_tr = feedforward(
                    inputs=x, targets=p, model=model.train(),
                    loss_reg=distfunc_reg, loss_segm=distfunc_segm,
                    weights=w, accumulation=use_gradient_accumulation)

            # update parameters
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if num_examples < (num_validations * num_examples_validation):
                continue

            num_validations += 1
            model.eval()

            validation_time: float = 0
            if run_smoke_test:
                validation_time = time.time()

            with torch.no_grad():

                if use_loss_contrastive:

                    # pick up a validation batch
                    batch = data_vl(batch_size, seed=0)
                    x_1 = batch.inputs_1.cuda()
                    x_2 = batch.inputs_2.cuda()
                    p_1 = batch.targets_1.cuda()
                    p_2 = batch.targets_2.cuda()

                    # estimate data purification weights
                    w_1 = predictor(p_1) if use_loss_purification else None
                    w_2 = predictor(p_2) if use_loss_purification else None

                    # forward propagation
                    metrics_vl = contrastive_feedforward(
                        inputs_1=x_1, inputs_2=x_2,
                        targets_1=p_1, targets_2=p_2,
                        weights_1=w_1, weights_2=w_2,
                        lambda_positive=lambda_p, lambda_negative=lambda_n,
                        loss_reg=distfunc_reg, loss_segm=distfunc_segm,
                        labels=batch.labels.cuda(),
                        model=model.cuda(),
                        accumulation=use_gradient_accumulation,
                        validation=True)

                else:

                    # pick up a validation batch
                    batch = data_vl(batch_size, seed=0)
                    x = batch.inputs.cuda()
                    p = batch.targets.cuda()

                    # estimate data purification weights
                    w = predictor(p) if use_loss_purification else None

                    # forward propagation
                    metrics_vl = feedforward(
                        inputs=x, targets=p, model=model.eval(),
                        loss_reg=distfunc_reg, loss_segm=distfunc_segm,
                        weights=w, accumulation=use_gradient_accumulation)

                # checkpoint whenever validation score improves
                if early_stopping_metric == 'loss':
                    save_ckpt = bool(metrics_vl['loss']<=best_score)
                else:
                    save_ckpt = bool(metrics_vl['sisdri']>=best_score)

                if save_ckpt:
                    best_score = metrics_vl[early_stopping_metric]
                    best_score_step = num_examples
                    best_state_dict = model.state_dict()
                    ckpt_path = output_directory.joinpath('ckpt_best.pt')
                    torch.save({
                        'num_examples': num_examples,
                        'model_name': model_name,
                        'model_config': config,
                        'model_state_dict': best_state_dict,
                        'optimizer_state_dict': optimizer.state_dict()
                    }, ckpt_path)
                    if not called_by_ray:
                        print(f'Examples: {num_examples:>10},\t'
                              'Validation SI-SDRi: '+str(metrics_vl['sisdri']))
                    step_path = output_directory.joinpath('best_step.txt')
                    with open(step_path, 'w') as fp:
                        print(num_examples, file=fp)

                # write summaries
                for (k, v) in metrics_tr.items():
                    if ('_inp' not in k) and ('_enh' not in k):
                        writer.add_scalar(
                            f'train/{k}', float(v), num_examples)
                for (k, v) in metrics_vl.items():
                    if ('_inp' not in k) and ('_enh' not in k):
                        writer.add_scalar(
                            f'validation/{k}', float(v), num_examples)
                writer.add_scalar(
                    f'validation/vl_score', best_score, num_examples)
                if called_by_ray:
                    _e = early_stopping_metric
                    tune.report(**{
                        'num_examples': num_examples,
                        f'vl_{_e}': metrics_vl[_e],
                        f'vl_score': best_score
                    })

                if num_examples > num_examples_minimum:
                    if num_examples - best_score_step > num_examples_earlystopping:
                        raise EarlyStopping()

                if run_smoke_test:
                    validation_time = time.time() - validation_time
                    smoke_path = output_directory.joinpath(f'smoke_test.txt')
                    with open(smoke_path, 'w') as fp:
                        print('Validation Run-Time (in seconds):'
                              f' {validation_time}', file=fp)
                    raise SmokeTest()

    except EarlyStopping:
        step_path = output_directory.joinpath(f'early_stopping.txt')
        with open(step_path, 'w') as fp:
            print(f'{num_examples}\n{best_score_step}\n{best_score}', file=fp)
        print(f'Automatically exited after {num_examples_earlystopping} '
              f'examples; best model saw {best_score_step} examples.')

    except SmokeTest:
        print(f'Exiting due to smoke test.')

    except KeyboardInterrupt:
        print(f'Manually exited at {num_examples} examples; best model saw '
              f'{best_score_step} examples.')
        raise KeyboardInterrupt

    torch.save({
        'num_examples': num_examples,
        'model_name': model_name,
        'model_config': model_config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_directory.joinpath(f'ckpt_last.pt'))

    # run the test set
    model.load_state_dict(best_state_dict)
    te_results = test_denoiser_with_speaker(model, num_examples_to_save=3)
    te_results['num_examples'] = best_score_step
    te_results[f'vl_{early_stopping_metric}'] = best_score

    if called_by_ray:
        tune.report(**te_results)

    with open(output_directory.joinpath('test_results.json'), 'w',
              encoding='utf-8') as fp:
        json.dump(te_results, fp, indent=2, sort_keys=True)
        print(json.dumps(te_results, indent=2, sort_keys=True), '\n')

    # close the summary
    writer.close()
    print(f'Output Directory: {str(output_directory)}')

    # exit the trainer
    return


def finetune_denoiser(
        dataset_duration: float,
        checkpoint_locations: Sequence[Union[str, os.PathLike]],
        learning_rate: float = 1e-4,
        num_examples_validation: int = 1000,
        num_examples_earlystopping: int = 10000,
        output_folder: Union[str, os.PathLike] = f'finetuning_{_host}',
        early_stopping_metric: str = 'sisdri',
        distance_func: str = 'mse'
):
    """Finetunes a denoiser, given checkpoint and dataset size.
    """
    if isinstance(checkpoint_locations, (str, os.PathLike)):
        checkpoint_locations = [checkpoint_locations]
    for checkpoint_location in checkpoint_locations:

        # Load checkpoint and previous settings from file.
        checkpoint_location = checkpoint_location.replace(
            'early_stopping.txt', '')
        base_model, config = load_checkpoint(checkpoint_location)
        model_name = config.get('model_name')
        model_size = config.get('model_size')
        batch_size = config.get('batch_size')
        config['is_finetuning'] = True
        config['dataset_duration'] = dataset_duration
        config['learning_rate'] = learning_rate
        config['num_examples_validation'] = num_examples_validation
        config['num_examples_earlystopping'] = num_examples_earlystopping
        config['output_folder'] = output_folder
        config['early_stopping_metric'] = early_stopping_metric
        config['distance_func'] = distance_func

        # define the distance function
        if distance_func == 'snr':
            distfunc_reg = asteroid.losses.sdr.SingleSrcNegSDR('snr')
            distfunc_segm = SegmentalLoss('snr', reduction='none')
        elif distance_func == 'sisdr':
            distfunc_reg = asteroid.losses.sdr.SingleSrcNegSDR('sisdr')
            distfunc_segm = SegmentalLoss('sisdr', reduction='none')
        else:
            distfunc_reg = torch.nn.MSELoss(reduction='none')
            distfunc_segm = SegmentalLoss('mse', reduction='none')

        # If this is a generalist, loop through all the personalization targets.
        # Else, if it is a specialist, this loop will only run once.
        try:
            speaker_ids = sorted(map(
                int, config.get('speaker_ids').strip('][').split(', ')))
            config['is_generalist'] = False
        except ValueError:
            speaker_ids = speaker_ids_te
            config['is_generalist'] = True
        for speaker_id in speaker_ids:

            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            model = copy.deepcopy(base_model).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            data_tr = Mixtures(
                speaker_id, split_speech='train', split_mixture='train',
                snr_mixture=(-5, 5), dataset_duration=dataset_duration)
            data_vl = Mixtures(
                speaker_id, split_speech='val', split_mixture='val',
                snr_mixture=(-5, 5), dataset_duration=dataset_duration)
            config['data_tr'] = data_tr.__dict__()
            config['data_vl'] = data_vl.__dict__()
            config['speaker_ids'] = data_tr.speaker_ids_repr

            # Instantiate tensorboard
            trial_name = '{}_{}_{}p_{}c_{}{:03}_ft{:02}'.format(
                model_name, model_size,
                'y' if config.get('use_loss_purification') else 'n',
                'y' if config.get('use_loss_contrastive') else 'n',
                'ge' if config.get('is_generalist') else 'sp',
                speaker_id, int(dataset_duration)
            )
            output_directory = Path(output_folder).joinpath(
                current_time + '_' + trial_name)
            writer = SummaryWriter(str(output_directory))
            save_config(output_directory, config)

            # Begin training
            num_examples: int = 0
            num_validations: int = 0
            best_score: float = np.inf * (1 if early_stopping_metric == 'loss'
                                          else
                                          -1)
            best_score_step: int = 0
            use_gradient_accumulation: bool = not bool('grunet' in model_name)
            print(f'Output Directory: {str(output_directory)}')

            try:
                for num_examples in itertools.count(start=0, step=batch_size):

                    batch = data_tr(batch_size)

                    metrics_tr = feedforward(
                        batch.inputs, batch.targets, model.train(),
                        loss_reg=distfunc_reg, loss_segm=distfunc_segm,
                        accumulation=use_gradient_accumulation)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if num_examples < (num_validations*num_examples_validation):
                        continue

                    num_validations += 1
                    batch = data_vl(batch_size, seed=0)
                    metrics_vl = feedforward(
                        batch.inputs, batch.targets, model.eval(),
                        loss_reg=distfunc_reg, loss_segm=distfunc_segm,
                        accumulation=use_gradient_accumulation)

                    # write summaries
                    for (k, v) in metrics_tr.items():
                        if ('_inp' not in k) and ('_enh' not in k):
                            writer.add_scalar(
                                f'train/{k}', float(v), num_examples)
                    for (k, v) in metrics_vl.items():
                        if ('_inp' not in k) and ('_enh' not in k):
                            writer.add_scalar(
                                f'validation/{k}', float(v), num_examples)

                    do_save_checkpoint = {
                        'loss': bool(metrics_vl['loss'] <= best_score),
                        'sisdri': bool(metrics_vl['sisdri'] >= best_score)
                    }.get(early_stopping_metric, False)

                    if do_save_checkpoint:
                        best_score = {
                            'loss': metrics_vl['loss'],
                            'sisdri': metrics_vl['sisdri']
                        }.get(early_stopping_metric, 0)
                        best_score_step = num_examples
                        ckpt_path = output_directory.joinpath('ckpt_best.pt')
                        torch.save({
                            'num_examples': num_examples,
                            'model_name': model_name,
                            'model_config': config,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                        }, ckpt_path)
                        print(f'Examples: {num_examples:>10},\t'
                              'Validation SI-SDRi: '+str(metrics_vl['sisdri']))
                        step_path = output_directory.joinpath('best_step.txt')
                        with open(step_path, 'w') as fp:
                            print(num_examples, file=fp)

                    if (num_examples - best_score_step >
                            num_examples_earlystopping):
                        raise EarlyStopping()

            except EarlyStopping:
                step_path = output_directory.joinpath(f'early_stopping.txt')
                with open(step_path, 'w') as fp:
                    print(f'{num_examples},{best_score_step}', file=fp)
                print(f'Automatically exited after {num_examples_earlystopping}'
                      f' examples; best model saw {best_score_step} examples.')

            writer.close()
            print(f'Output Directory: {str(output_directory)}')

    return


def parse_arguments(
        arg_list: Optional[List[str]] = None
) -> argparse.Namespace:
    """Parses arguments from a list."""
    # use system default arguments
    if arg_list is None: arg_list = sys.argv[1:]
    abs_path = lambda p: Path(p).absolute()

    def t_mixture_snr(string):
        try:
            return_val = float(string)
        except ValueError:
            return_val = literal_eval(string)
        return return_val

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('model_size', type=str,
                        choices={'tiny', 'small', 'medium', 'large'})
    parser.add_argument('--speaker_id', type=int, nargs='+', required=False)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--use_loss_purification', action='store_true')
    parser.add_argument('--use_loss_contrastive', action='store_true')
    parser.add_argument('--lambda_p', type=float, default=1.)
    parser.add_argument('--lambda_n', type=float, default=1.)
    parser.add_argument('--generalist_frac', type=float, default=1.)
    parser.add_argument('--distance_func', type=str,
                        choices={'mse', 'snr', 'sisdr'}, required=True)
    parser.add_argument('--early_stopping_metric', type=str,
                        choices={'loss', 'sisdri'}, default='sisdri')
    parser.add_argument("--premixture_snr",
                        type=t_mixture_snr, default='(0, 15)')
    parser.add_argument("--mixture_snr",
                        type=t_mixture_snr, default='(-5, 5)')
    parser.add_argument('--warm_start', type=abs_path)
    parser.add_argument('--trial_suffix', type=str, default='')
    parser.add_argument('--output_folder', type=abs_path,
                        default=abs_path(__file__).parent / f'runs_{_host}')
    args = parser.parse_args(arg_list)

    # validate warm start argument
    if args.warm_start:
        if Path(args.warm_start).suffix != '.pt':
            raise IOError('Warm start checkpoint should have extension ".pt".')
        if not Path(args.warm_start).is_file():
            raise IOError('Warm start checkpoint does not exist.')
        args.warm_start = str(args.warm_start)

    # validate speaker IDs
    if args.speaker_id:
        # check that speaker IDs are valid for personalization experiments
        if not set(args.speaker_id).issubset(set(speaker_ids_te)):
            raise ExperimentError(
                'Please choose speaker IDs specificed in "speakers/test.csv". '
                'Allowed values are: {}.'.format(speaker_ids_te))
    return args


def hparam_search_cm(
        speaker_id_or_ids: Union[int, Sequence[int]] = 200,
        num_cpus: int = 1,
        num_gpus: int = 1
):
    # define the hyperparameter search space
    search_space = {
        'distance_func': tune.grid_search(['snr',]),
        'use_loss_purification': tune.grid_search([False, True]),
        'lambda_p': tune.grid_search([0, 0.0001, 0.0005, 0.001, 0.005,
                                      0.01, 0.05, 0.1, 0.5, 1]),
        'lambda_n': tune.grid_search([0, 0.0001, 0.0005, 0.001, 0.005,
                                      0.01, 0.05, 0.1, 0.5, 1]),
    }

    def ray_search_cm(config):
        d_tr = ContrastiveMixtures(
            speaker_id_or_ids, split_speech='pretrain',
            split_premixture='train', snr_premixture=(0, 15),
            split_mixture='train', snr_mixture=(-5, 5))
        d_vl = ContrastiveMixtures(
            speaker_id_or_ids, split_speech='preval',
            split_premixture='val', snr_premixture=(0, 15),
            split_mixture='val', snr_mixture=(-5, 5))
        train_denoiser(
            model_name='convtasnet',
            model_size='small',
            data_tr=d_tr,
            data_vl=d_vl,
            use_loss_purification=config['use_loss_purification'],
            lambda_p=config['lambda_p'],
            lambda_n=config['lambda_n'],
            output_folder='.',
            distance_func=config['distance_func'],
            called_by_ray=True,
        )
        return

    analysis = tune.run(
        ray_search_cm,
        name='ray_search_cm',
        config=search_space,
        resources_per_trial={'cpu': num_cpus, 'gpu': num_gpus},
        reuse_actors=True,
        log_to_file=True,
        local_dir='.',
        fail_fast=True,
        verbose=1
    )
    ts = datetime.now().strftime('%b%d_%H-%M-%S')
    analysis.results_df.to_csv(f'ray_search_cm/results_{ts}.csv')
    return


def hparam_search_df(
        speaker_id_or_ids: Union[int, Sequence[int]] = 200,
        num_cpus: int = 1,
        num_gpus: int = 1
):
    # define the hyperparameter search space
    search_space = {
        'model_size': tune.grid_search(['tiny', 'small', 'medium', 'large']),
        'distance_func': tune.grid_search(['mse', 'snr', 'sisdr']),
        'use_loss_purification': tune.grid_search([False, True]),
    }

    def ray_search_distance_func(config):
        d_tr = Mixtures(
            speaker_id_or_ids, split_speech='pretrain',
            split_premixture='train', snr_premixture=(0, 15),
            split_mixture='train', snr_mixture=(-5, 5))
        d_vl = Mixtures(
            speaker_id_or_ids, split_speech='preval',
            split_premixture='val', snr_premixture=(0, 15),
            split_mixture='val', snr_mixture=(-5, 5))
        train_denoiser(
            model_name='convtasnet',
            model_size=config['model_size'],
            data_tr=d_tr,
            data_vl=d_vl,
            use_loss_purification=config['use_loss_purification'],
            output_folder='.',
            distance_func=config['distance_func'],
            called_by_ray=True,
        )
        return

    analysis = tune.run(
        ray_search_distance_func,
        name='ray_search_distance_func',
        config=search_space,
        resources_per_trial={'cpu': num_cpus, 'gpu': num_gpus},
        reuse_actors=True,
        log_to_file=True,
        local_dir='.',
        fail_fast=True,
        verbose=1
    )
    ts = datetime.now().strftime('%b%d_%H-%M-%S')
    analysis.results_df.to_csv(f'ray_search_distance_func/results_{ts}.csv')
    return


def train_all_generalists(
        num_cpus: int = 1,
        num_gpus: int = 1
):
    # define the hyperparameter search space
    search_space = {
        'model_name': 'convtasnet',
        'model_size': tune.grid_search(['medium', 'large']),
        # 'model_size': tune.grid_search(['tiny', 'small', 'medium', 'large']),
        'distance_func': tune.grid_search(['snr', 'sisdr']),
        'generalist_frac': 1,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'mixture_snr': (-5, 5)
    }

    def train_generalist(config: dict):
        train_denoiser(
            model_name=config['model_name'],
            model_size=config['model_size'],
            distance_func=config['distance_func'],
            data_tr=Mixtures(speaker_ids_tr,
                             frac_speech=config['generalist_frac'],
                             split_mixture='train',
                             snr_mixture=config.get('mixture_snr', (-5, 5))),
            data_vl=Mixtures(speaker_ids_vl,
                             split_mixture='val',
                             snr_mixture=config.get('mixture_snr', (-5, 5))),
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            output_folder='.',
            called_by_ray=True,
        )
        return

    analysis = tune.run(
        train_generalist,
        name='train_generalist',
        config=search_space,
        resources_per_trial={'cpu': num_cpus, 'gpu': num_gpus},
        reuse_actors=True,
        log_to_file=True,
        local_dir='.',
        fail_fast=True,
        verbose=3
    )
    ts = datetime.now().strftime('%b%d_%H-%M-%S')
    analysis.results_df.to_csv(f'train_generalist/results_{ts}.csv')
    return


def train_all_specialists(
        num_cpus: int = 1,
        num_gpus: int = 1
):
    # define the hyperparameter search space
    search_space = {
        'model_name': 'convtasnet',
        'model_size': tune.grid_search(['tiny', 'small', 'medium', 'large']),
        'distance_func': tune.grid_search(['snr',]),
        'speaker_id': tune.grid_search([
            201, 250, 254, 307, 405, 446,]),
        'use_loss_contrastive': tune.grid_search([False,]),
        'use_loss_purification': tune.grid_search([False, True]),
        'batch_size': 64,
        'learning_rate': 1e-3,
        'run_smoke_test': False
    }

    def train_specialist(config: dict):
        dc = Mixtures
        if config.get('use_loss_contrastive', False):
            dc = ContrastiveMixtures
        train_denoiser(
            model_name=config['model_name'],
            model_size=config['model_size'],
            distance_func=config['distance_func'],
            data_tr=dc(config['speaker_id'],
                       split_speech='pretrain',
                       split_premixture='train',
                       snr_premixture=config.get('premixture_snr', (0, 15)),
                       split_mixture='train',
                       snr_mixture=config.get('mixture_snr', (-5, 5))),
            data_vl=dc(config['speaker_id'],
                       split_speech='preval',
                       split_premixture='val',
                       snr_premixture=config.get('premixture_snr', (0, 15)),
                       split_mixture='val',
                       snr_mixture=config.get('mixture_snr', (-5, 5))),
            learning_rate=config.get('learning_rate', 1e-3),
            use_loss_purification=config.get('use_loss_purification', False),
            batch_size=config['batch_size'],
            output_folder='.',
            called_by_ray=True,
            run_smoke_test=config.get('run_smoke_test', False)
        )
        return

    analysis = tune.run(
        train_specialist,
        name='train_specialist',
        config=search_space,
        resources_per_trial={'cpu': num_cpus, 'gpu': num_gpus},
        reuse_actors=True,
        log_to_file=True,
        local_dir='.',
        fail_fast=True,
        verbose=3
    )
    ts = datetime.now().strftime('%b%d_%H-%M-%S')
    analysis.results_df.to_csv(f'train_specialist/results_{ts}.csv')
    return

if __name__ == '__main__':
    # hparam_search_cm()
    # hparam_search_df()
    # train_all_generalists()
    train_all_specialists()
