import argparse
import itertools
import json
import os
import pathlib
import sys
import warnings
from datetime import datetime
from math import ceil
from typing import Optional
from typing import Union

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from exp_data import Mixtures, sample_rate, example_duration
from exp_data import speaker_ids_tr, speaker_ids_vl, speaker_ids_te
from exp_models import init_model, feedforward, SegmentalLoss, SNRPredictor
from exp_models import test_denoiser_from_folder
from exp_utils import EarlyStopping, ExperimentError

warnings.filterwarnings('ignore')


def save_config(
        output_directory: Union[str, os.PathLike],
        config: dict
):
    output_directory = pathlib.Path(output_directory)
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
        batch_size: int = 8,
        checkpoint_path: Optional[str] = None,
        num_examples_validation: int = 1000,
        num_examples_earlystopping: int = 100000,
        trial_name: Optional[str] = None
) -> str:
    # prepare model, optimizer, and loss function
    model, nparams, model_config = init_model(model_name, model_size)
    model = model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss(reduction='mean')
    predictor = torch.nn.Identity()
    if use_loss_purification:
        criterion = SegmentalLoss('mse', reduction='mean')
        predictor = SNRPredictor()
        predictor.load_state_dict(torch.load('snr_predictor'), strict=False)
        predictor.cuda()
        predictor.eval()

    # load a previous checkpoint if provided
    init_num_examples = 0
    if checkpoint_path:
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
        'use_loss_purification': use_loss_purification,
    }

    # instantiate tensorboard
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    output_directory = pathlib.Path('runs').joinpath(
        current_time + '_' + trial_name) # trial_name(config=config))
    writer = SummaryWriter(str(output_directory))
    save_config(output_directory, config)

    # begin training (use gradient accumulation for TasNet models)
    num_examples: int = init_num_examples
    num_validations: int = ceil(num_examples / num_examples_validation)
    min_loss: float = np.inf
    min_loss_num_examples: int = init_num_examples
    use_gradient_accumulation: bool = bool('tasnet' in model_name)
    print(f'Output Directory: {str(output_directory)}')

    try:
        for num_examples in itertools.count(start=init_num_examples,
                                            step=batch_size):
            model.train()
            with torch.set_grad_enabled(True):

                # pick up a training batch
                batch = data_tr(batch_size)
                x = batch.inputs.cuda()
                p = batch.targets.cuda()

                # estimate data purification weights
                w = predictor(p) if use_loss_purification else None

                # forward propagation
                loss_tr, sisdri_tr = feedforward(
                    x, p, model, criterion, w, use_gradient_accumulation,
                    validation=False)

                # update parameters
                optimizer.step()
                optimizer.zero_grad()

            if num_examples > (num_validations * num_examples_validation):
                continue

            num_validations += 1
            model.eval()
            with torch.no_grad():

                # pick up a validation batch
                batch = data_vl(batch_size, seed=0)
                x = batch.inputs.cuda()
                p = batch.targets.cuda()

                # estimate data purification weights
                w = predictor(p) if use_loss_purification else None

                loss_vl, sisdri_vl = feedforward(
                    x, p, model, criterion, w, use_gradient_accumulation,
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
                if loss_vl < min_loss:
                    min_loss = loss_vl
                    min_loss_num_examples = num_examples
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

                if (num_examples - min_loss_num_examples >
                        num_examples_earlystopping):
                    raise EarlyStopping()

    except EarlyStopping:
        print(f'Automatically exited after {num_examples_earlystopping} '
              f'examples; best model saw {min_loss_num_examples} examples.')

    except KeyboardInterrupt:
        print(f'Manually exited at {num_examples} examples; best model saw '
              f'{min_loss_num_examples} examples.')
        torch.save({
            'num_examples': num_examples,
            'model_name': model_name,
            'model_config': model_config,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, output_directory.joinpath(f'ckpt_last.pt'))
        sys.exit(-1)

    # close the summary
    writer.close()

    # print the location of the checkpoints
    print(f'Output Directory: {str(output_directory)}')

    # exit the trainer
    return str(output_directory)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('model_size', type=str,
                        choices={'small', 'medium', 'large'})
    parser.add_argument('--speaker_id', type=int, nargs='+', required=False)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--use_loss_purification', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(args)

    # train one or more specialist models
    if args.speaker_id:

        # check that speaker IDs are valid for personalization experiments
        if not set(args.speaker_id).issubset(set(speaker_ids_te)):
            raise ExperimentError(
                'Please choose speaker IDs specificed in "speakers/test.csv". '
                'Allowed values are: {}.'.format(speaker_ids_te))

        for speaker_id in args.speaker_id:

            d_tr = Mixtures(speaker_id, split_speech='pretrain',
                            split_premixture='train', split_mixture='train',
                            snr_premixture=(0, 10), snr_mixture=(-5, 5))
            d_vl = Mixtures(speaker_id, split_speech='preval',
                            split_premixture='val', split_mixture='val',
                            snr_premixture=(0, 10), snr_mixture=(-5, 5))

            results_directory = train_denoiser(
                model_name=args.model_name,
                model_size=args.model_size,
                data_tr=d_tr,
                data_vl=d_vl,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                use_loss_purification=args.use_loss_purification,
                trial_name='{}_{}_sp{:03}{}'.format(
                    args.model_name, args.model_size, speaker_id,
                    '_dp' if args.use_loss_purification else ''
                )
            )
            print(test_denoiser_from_folder(results_directory))

    # train one generalist model
    else:

        d_tr = Mixtures(speaker_ids_tr,
                        split_mixture='train',
                        snr_mixture=(-5, 5))
        d_vl = Mixtures(speaker_ids_vl,
                        split_mixture='val',
                        snr_mixture=(-5, 5))

        results_directory = train_denoiser(
            model_name=args.model_name,
            model_size=args.model_size,
            data_tr=d_tr,
            data_vl=d_vl,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            trial_name='{}_{}'.format(
                args.model_name, args.model_size
            )
        )
        print(test_denoiser_from_folder(results_directory))

    return


if __name__ == '__main__':
    main()

