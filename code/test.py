import pathlib
import re
import sys
from datetime import datetime

import torch
from ray import tune

from exp_data import Mixtures
from exp_models import load_checkpoint, feedforward


def get_timestamp() -> str:
    # format_str = "%A, %d %b %Y %H:%M:%S %p"
    format_str = "%Y_%b_%d"
    result = str(datetime.now().strftime(format_str))
    return result


def no_op_loss(*args, **kwargs):
    return 0


@torch.no_grad()
def test_function(
        filepath: str,
        print_to_console: bool = True,
        write_to_file: bool = True,
        called_by_tune: bool = True
):
    use_gradient_accumulation = not bool('grunet' in filepath)
    filepath = pathlib.Path(filepath.strip().replace('early_stopping.txt', ''))

    # load the experiment configuration (should be in the same directory)
    model, config, num_examples = load_checkpoint(filepath)

    # indentify the personalization target (if there is one)
    # and prepare the speaker-specific test sets
    speaker_id = 200
    if 'ray' not in str(filepath):
        try:
            match = re.match(r'.*_(sp|ge)(\d\d\d).*', str(filepath))
            speaker_id = int(match.group(2))
        except AttributeError:
            raise NotImplementedError('need to add support for generalists')
    dataset = Mixtures(speaker_id,
                       split_speech='test',
                       split_mixture='test',
                       snr_mixture=(-5, 5))

    # run the test
    batch = dataset(100, seed=0)
    results = feedforward(batch.inputs, batch.targets, model,
                          weights=None, accumulation=use_gradient_accumulation,
                          test=True, loss_reg=no_op_loss, loss_segm=no_op_loss)

    if print_to_console:
        print(f'{filepath} ({num_examples}),{results}')
    if write_to_file:
        with open('log.txt', 'a') as op:
            print(f'{filepath} ({num_examples}),{results}', file=op)
    if called_by_tune:
        tune.report(**results)
        return
    else:
        return results


def main(use_tune: bool = False):

    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    else:
        p = pathlib.Path('/N/u/asivara/2022-jstsp/0408_hparams_cm').rglob(
            'ckpt_best.pt')
        folders = [str(f) for f in p if '619b6' in str(f)]
    # else:
    #     raise ValueError('Expected subsequent arguments to be checkpoint paths '
    #                      'or directory.')

    def test_wrapper(config):
        return test_function(
            filepath=config['filepath'],
            called_by_tune=True
        )

    if use_tune:
        tune.run(
            test_wrapper,
            config={
                'filepath': tune.grid_search(folders)
            },
            resources_per_trial={'cpu': 1, 'gpu': 0.25},
            local_dir=f'test_results-({get_timestamp()})'
        )
        pass
    else:
        for f in folders:
            test_function(f, called_by_tune=False)


if __name__ == '__main__':
    main(False)
