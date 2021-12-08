import argparse
import itertools
import json
import os
import sys
import warnings
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Optional, List
from typing import Union

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from exp_data import ContrastiveMixtures, Mixtures
from exp_data import example_duration, sample_rate
from exp_data import speaker_ids_tr, speaker_ids_vl, speaker_ids_te
from exp_models import SNRPredictor, init_model
from exp_models import contrastive_feedforward, feedforward
from exp_utils import EarlyStopping, ExperimentError

warnings.filterwarnings('ignore')


def save_config(
        output_directory: Union[str, os.PathLike],
        config: dict
):
    output_directory = Path(output_directory)
    with open(output_directory.joinpath('config.json'), 'w',
              encoding='utf-8') as fp:
        json.dump(config, fp, indent=2, sort_keys=True)
        print(yaml.safe_dump(config, default_flow_style=False))


def train_denoiser(
        model_name: str,
        model_size: str,
        data_tr: Mixtures,
        data_vl: Mixtures,
        use_loss_purification: bool = False,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        checkpoint_path: Optional[str] = None,
        num_examples_validation: int = 1000,
        num_examples_earlystopping: int = 100000,
        trial_name: Optional[str] = None,
        output_folder: Union[str, os.PathLike] = 'runs',
        training_metric: str = 'sisdri'
) -> str:
    # prepare model, optimizer, and loss function
    model, nparams, model_config = init_model(model_name, model_size)
    model = model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    predictor = torch.nn.Identity()
    if use_loss_purification:
        predictor = SNRPredictor()
        predictor.load_state_dict(torch.load('snr_predictor'), strict=False)
        predictor.cuda()
        predictor.eval()

    use_loss_contrastive: bool = bool(isinstance(data_tr, ContrastiveMixtures))
    if not type(data_tr) is type(data_vl):
        raise ValueError('`data_tr` and `data_vl` should be the same type.')

    # load a previous checkpoint if provided
    init_num_examples = 0
    output_directory: Optional[Path] = None
    if checkpoint_path:
        # overwrite output directory (to pick up experiment where left off)
        output_directory = Path(checkpoint_path).parent
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        init_num_examples = ckpt['num_examples']

    # define experiment configuration
    config = {
        'batch_size': batch_size,
        'checkpoint_path': str(checkpoint_path or ''),
        'data_tr': data_tr.__dict__(),
        'data_vl': data_vl.__dict__(),
        'example_duration': example_duration,
        'learning_rate': learning_rate,
        'model_config': model_config,
        'model_name': model_name,
        'model_nparams': nparams,
        'model_size': model_size,
        'num_examples_earlystopping': num_examples_earlystopping,
        'num_examples_validation': num_examples_validation,
        'sample_rate': sample_rate,
        'speaker_ids': data_tr.speaker_ids_repr,
        'use_loss_contrastive': use_loss_contrastive,
        'use_loss_purification': use_loss_purification,
        'training_metric': training_metric
    }

    # instantiate tensorboard
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if output_directory is None:
        output_directory = Path(output_folder).joinpath(
            current_time + '_' + trial_name)
    writer = SummaryWriter(str(output_directory))
    save_config(output_directory, config)

    # begin training (use gradient accumulation for TasNet models)
    num_examples: int = init_num_examples
    num_validations: int = ceil(num_examples / num_examples_validation)
    best_score: float = np.inf * (1 if training_metric == 'mse' else -1)
    best_score_step: int = init_num_examples
    use_gradient_accumulation: bool = bool('tasnet' in model_name)
    print(f'Output Directory: {str(output_directory)}')

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
                loss_tr, sisdri_tr = contrastive_feedforward(
                    inputs_1=x_1, inputs_2=x_2,
                    targets_1=p_1, targets_2=p_2,
                    weights_1=w_1, weights_2=w_2,
                    lambda_positive=1., lambda_negative=1.,
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
                loss_tr, sisdri_tr = feedforward(
                    x, p, model, w, use_gradient_accumulation,
                    validation=False)

            # update parameters
            optimizer.step()
            optimizer.zero_grad()

            if num_examples < (num_validations * num_examples_validation):
                continue

            num_validations += 1
            model.eval()
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
                    loss_vl, sisdri_vl = contrastive_feedforward(
                        inputs_1=x_1, inputs_2=x_2,
                        targets_1=p_1, targets_2=p_2,
                        weights_1=w_1, weights_2=w_2,
                        lambda_positive=1., lambda_negative=1.,
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
                    loss_vl, sisdri_vl = feedforward(
                        x, p, model, w, use_gradient_accumulation,
                        validation=True)

                # write summaries
                writer.add_scalar('MSELoss/train',
                                  float(loss_tr), num_examples)
                writer.add_scalar('SISDRi/train',
                                  float(sisdri_tr), num_examples)
                writer.add_scalar('MSELoss/validation',
                                  float(loss_vl), num_examples)
                writer.add_scalar('SISDRi/validation',
                                  float(sisdri_vl), num_examples)

                # checkpoint whenever validation score improves
                if training_metric == 'mse':
                    save_ckpt = bool(float(loss_vl)<=best_score)
                else:
                    save_ckpt = bool(float(sisdri_vl)>=best_score)

                if save_ckpt:
                    if training_metric == 'mse':
                        best_score = float(loss_vl)
                    else:
                        best_score = float(sisdri_vl)
                    best_score_step = num_examples
                    ckpt_path = output_directory.joinpath(
                        f'ckpt_{num_examples:08}.pt')
                    torch.save({
                        'num_examples': num_examples,
                        'model_name': model_name,
                        'model_config': model_config,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, ckpt_path)
                    step_path = output_directory.joinpath(f'best_step.txt')
                    with open(step_path, 'w') as fp:
                        print(num_examples, file=fp)

                if num_examples - best_score_step > num_examples_earlystopping:
                    raise EarlyStopping()

    except EarlyStopping:
        step_path = output_directory.joinpath(f'early_stopping.txt')
        with open(step_path, 'w') as fp:
            print('{},{}'.format(num_examples, best_score_step), file=fp)
        print(f'Automatically exited after {num_examples_earlystopping} '
              f'examples; best model saw {best_score_step} examples.')

    except KeyboardInterrupt:
        print(f'Manually exited at {num_examples} examples; best model saw '
              f'{best_score_step} examples.')

    torch.save({
        'num_examples': num_examples,
        'model_name': model_name,
        'model_config': model_config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_directory.joinpath(f'ckpt_last.pt'))

    # close the summary
    writer.close()

    # print the location of the checkpoints
    print(f'Output Directory: {str(output_directory)}')

    # exit the trainer
    return str(output_directory)


def parse_arguments(arg_list: Optional[List[str]] = None):

    # use system default arguments
    if arg_list is None: arg_list = sys.argv[1:]
    abs_path = lambda p: Path(p).absolute()

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('model_size', type=str,
                        choices={'small', 'medium', 'large'})
    parser.add_argument('--speaker_id', type=int, nargs='+', required=False)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--use_loss_purification', action='store_true')
    parser.add_argument('--use_loss_contrastive', action='store_true')
    parser.add_argument('--training_metric', type=str,
                        choices={'mse', 'sisdri'}, default='sisdri')
    parser.add_argument('--warm_start', type=abs_path)
    parser.add_argument('--output_folder', type=abs_path,
                        default=abs_path(__file__).parent / 'runs')
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


def main():
    args = parse_arguments()

    dataset_class = Mixtures
    if args.use_loss_contrastive:
        dataset_class = ContrastiveMixtures

    # train one or more specialist models
    if args.speaker_id:

        for speaker_id in args.speaker_id:

            d_tr = dataset_class(speaker_id, split_speech='pretrain',
                            split_premixture='train', split_mixture='train',
                            snr_premixture=(0, 15), snr_mixture=(-5, 5))
            d_vl = dataset_class(speaker_id, split_speech='preval',
                            split_premixture='val', split_mixture='val',
                            snr_premixture=(0, 15), snr_mixture=(-5, 5))
            train_denoiser(
                model_name=args.model_name,
                model_size=args.model_size,
                data_tr=d_tr,
                data_vl=d_vl,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                use_loss_purification=args.use_loss_purification,
                trial_name='{}_{}_sp{:03}{}{}'.format(
                    args.model_name, args.model_size, speaker_id,
                    '_yp' if args.use_loss_purification else '_np',
                    '_yc' if args.use_loss_contrastive else '_nc'
                ),
                training_metric=args.training_metric,
                checkpoint_path=args.warm_start,
                output_folder=str(args.output_folder)
            )

    # train one generalist model
    else:

        d_tr = dataset_class(speaker_ids_tr,
                             split_mixture='train',
                             snr_mixture=(-5, 5))
        d_vl = dataset_class(speaker_ids_vl,
                             split_mixture='val',
                             snr_mixture=(-5, 5))
        train_denoiser(
            model_name=args.model_name,
            model_size=args.model_size,
            data_tr=d_tr,
            data_vl=d_vl,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            trial_name='{}_{}'.format(
                args.model_name, args.model_size
            ),
            training_metric=args.training_metric,
            checkpoint_path=args.warm_start,
            output_folder=str(args.output_folder)
        )
    return


if __name__ == '__main__':
    main()

