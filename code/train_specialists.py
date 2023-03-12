import argparse
import os
import sys
from ray import tune

from datetime import datetime
from exp_data import ContrastiveMixtures, Mixtures
from exp_utils import get_config_from_yaml
from run import train_denoiser


ROOT_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))


def train_specialist(config: dict):

    data_class = Mixtures
    if config.get('use_loss_contrastive', False):
        data_class = ContrastiveMixtures

    data_train = data_class(
        config['speaker_id'],
        config['folder_librispeech'],
        config['folder_fsd50k'],
        config['folder_musan'],
        split_speech='pretrain',
        split_premixture='train',
        snr_premixture=(0, 15),
        split_mixture='train',
        snr_mixture=(-5, 5)
    )
    data_validation = data_class(
        config['speaker_id'],
        config['folder_librispeech'],
        config['folder_fsd50k'],
        config['folder_musan'],
        split_speech='preval',
        split_premixture='val',
        snr_premixture=(0, 15),
        split_mixture='val',
        snr_mixture=(-5, 5)
    )

    train_denoiser(
        model_name=config['model_name'],
        model_size=config['model_size'],
        distance_func=config['distance_func'],
        data_tr=data_train,
        data_vl=data_validation,
        learning_rate=config['learning_rate'],
        use_loss_purification=config['use_loss_purification'],
        batch_size=config['batch_size'],
        output_folder=config['output_folder'],
        called_by_ray=True,
        run_smoke_test=config['run_smoke_test']
    )

    return


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--run_smoke_test",
        help="check if a single training iteration runs succesfully",
        action="store_true")
    args = parser.parse_args()

    config = get_config_from_yaml(ROOT_DIR+'/conf_specialists.yaml')
    os.environ['CUDA_VISIBLE_DEVICES'] = config['available_devices']

    analysis = tune.run(
        train_specialist,
        name='train_specialist',
        config={
            'model_name': tune.grid_search(config['model_name']),
            'model_size': tune.grid_search(config['model_size']),
            'distance_func': tune.grid_search(config['distance_func']),
            'speaker_id': tune.grid_search(config['speaker_id']),
            'use_loss_contrastive': tune.grid_search(
                config['use_loss_contrastive']),
            'use_loss_purification': tune.grid_search(
                config['use_loss_purification']),
            'batch_size': tune.grid_search(config['batch_size']),
            'learning_rate': tune.grid_search(config['learning_rate']),
            'folder_librispeech': config['folder_librispeech'],
            'folder_fsd50k': config['folder_fsd50k'],
            'folder_musan': config['folder_musan'],
            'sample_rate': config['sample_rate'],
            'example_duration': config['example_duration'],
            'output_folder': config['output_folder'],
            'run_smoke_test': args.run_smoke_test
        },
        resources_per_trial={
            'cpu': config['num_cpus_per_experiment'],
            'gpu': config['num_gpus_per_experiment']
        },
        reuse_actors=True,
        log_to_file=True,
        local_dir=config['output_folder'],
        fail_fast=True,
        verbose=3
    )
    ts = datetime.now().strftime('%b%d_%H-%M-%S')
    output_filepath = os.path.join(
        config['output_folder'], f'train_specialist/results_{ts}.csv')
    analysis.results_df.to_csv(output_filepath)
    print('Completed training specialist(s).')

