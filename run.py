import argparse
import itertools
import os
import pathlib
import json
import warnings
from datetime import datetime
from typing import Optional
from typing import Union

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from exp_data import Mixtures, sample_rate, example_duration
from exp_data import speaker_ids_tr, speaker_ids_vl
from exp_models import init_model, feedforward, SegmentalLoss, SNRPredictor
from exp_utils import EarlyStopping

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
        num_steps_validation: int = 100,
        num_steps_earlystopping: int = 1000,
        trial_name: Optional[str] = None
):
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
    init_step = 0
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        init_step = ckpt['step']

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
        'num_steps_earlystopping': num_steps_earlystopping,
        'num_steps_validation': num_steps_validation,
        'sample_rate': sample_rate,
        'use_loss_purification': use_loss_purification,
    }

    # instantiate tensorboard
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    output_directory = pathlib.Path('runs').joinpath(
        current_time + '_' + trial_name) # trial_name(config=config))
    writer = SummaryWriter(str(output_directory))
    save_config(output_directory, config)

    # begin training (use gradient accumulation for TasNet models)
    step: int = init_step
    min_loss: float = np.inf
    min_loss_step: int = init_step
    use_gradient_accumulation: bool = bool('tasnet' in model_name)
    print(f'Output Directory: {str(output_directory)}')

    try:
        for step in itertools.count(init_step):

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

            if step % num_steps_validation:
                continue

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
                writer.add_scalar('MSELoss/train', float(loss_tr), step)
                writer.add_scalar('SISDRi/train', float(sisdri_tr), step)
                writer.add_scalar('MSELoss/validation', float(loss_vl), step)
                writer.add_scalar('SISDRi/validation', float(sisdri_vl), step)

                # checkpoint whenever validation score improves
                if loss_vl < min_loss:
                    min_loss = loss_vl
                    min_loss_step = step
                    ckpt_path = output_directory.joinpath(f'ckpt_{step:08}.pt')
                    torch.save({
                        'step': step,
                        'model_name': model_name,
                        'model_config': model_config,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, ckpt_path)
                    step_path = output_directory.joinpath(f'best_step.txt')
                    with open(step_path, 'w') as fp:
                        print(step, file=fp)

                if (step - min_loss_step) > num_steps_earlystopping:
                    raise EarlyStopping()

    except EarlyStopping:
        print(f'Automatically exited after waiting {num_steps_earlystopping} '
              f'steps; best step was {min_loss_step}.')

    except KeyboardInterrupt:
        print(f'Manually exited at step {step}; best step was {min_loss_step}.')
        torch.save({
            'step': step,
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
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('model_size', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-s', '--sweep', action='store_true')
    args = parser.parse_args()
    print(args)
    if args.sweep:
        for (b, l) in itertools.product([64, 128], [1e-3, 5e-4, 1e-4]):
            train_denoiser(
                model_name=args.model_name,
                model_size=args.model_size,
                data_tr=Mixtures(speaker_ids_tr, 'all', snr_mixture=(-5, 5)),
                data_vl=Mixtures(speaker_ids_vl, 'all', snr_mixture=(-5, 5)),
                learning_rate=l,
                batch_size=b,
                trial_name='{}_{}_{:02}_{}'.format(
                    args.model_name, args.model_size, b, l
                )
            )
    else:
        train_denoiser(
            model_name=args.model_name,
            model_size=args.model_size,
            data_tr=Mixtures(speaker_ids_tr, 'all', snr_mixture=(-5, 5)),
            data_vl=Mixtures(speaker_ids_vl, 'all', snr_mixture=(-5, 5)),
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            trial_name='{}_{}_{:02}_{}'.format(
                args.model_name, args.model_size, args.batch_size,
                args.learning_rate
            )
        )

