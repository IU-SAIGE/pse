import collections
import json
import pathlib

import ast

from exp_models import test_denoiser
from exp_data import Mixtures, data_te_specialist
from time import sleep
import re
import sys

if len(sys.argv) > 1:
    folders = sys.argv[1:]
else:
    raise ValueError('Expected subsequent arguments to be checkpoint paths '  
                     'or directory.')

class Row(collections.namedtuple(
    'Row',
    ['checkpoint_path', 'test_metrics', 'model_name', 'model_size',
     'is_generalist', 'speaker_id', 'loss_contrastive', 'loss_purification',
     'finetune_duration', 'num_examples']
)):
    def __repr__(self):
        return (f'{self.checkpoint_path},'
                f'{self.test_metrics},'
                f'{self.model_name},'
                f'{self.model_size},'
                f'{int(self.is_generalist)},'
                f'{self.speaker_id},'
                f'{int(self.loss_contrastive)},'
                f'{int(self.loss_purification)},'
                f'{int(self.finetune_duration)},'
                f'{int(self.num_examples)}')


# print csv header row
print(','.join(Row._fields))

for f in folders:
    try:
        path = f.strip()
        path = path.replace('early_stopping.txt', '')
        accumulation = not bool('grunet' in f)
        checkpoint_path = pathlib.Path(path)
        if checkpoint_path.is_file():
            config_file = checkpoint_path.with_name('config.json')
        else:
            config_file = checkpoint_path.joinpath('config.json')
        if not config_file.exists():
            raise ValueError(f'Could not find {str(config_file)}.')
        with open(config_file, 'r') as fp:
            config: dict = json.load(fp)
        sp_id = ''
        try:
            sp_id = int(re.match(r'.*_(sp|ge)(\d\d\d).*', f).group(2))
            dataset = Mixtures(
                sp_id, 'test', split_mixture='test', snr_mixture=(-5,5))
            results, num_examples = test_denoiser(
                path, data_te=dataset, accumulation=accumulation,
                use_last=False)
        except AttributeError:
            results, num_examples = test_denoiser(
                path, data_te=data_te_specialist, accumulation=accumulation,
                use_last=False)
        for k, v in results.items():
            if isinstance(ast.literal_eval(k), list):
                k = ast.literal_eval(k).pop()
            # print(Row(
            #     checkpoint_path=checkpoint_path,
            #     test_metrics=v,
            #     model_name=config['model_name'],
            #     model_size=config['model_size'],
            #     is_generalist=not ('_sp' in f),
            #     speaker_id=k,
            #     loss_contrastive=int(config['use_loss_contrastive']),
            #     loss_purification=int(config['use_loss_purification']),
            #     finetune_duration=config.get('dataset_duration', 0),
            #     num_examples=num_examples
            # ))
            print(f'{checkpoint_path} ({num_examples}),{v}')
    except RuntimeError as e:
        if 'state_dict' in str(e):
            print(f'Skipping {f} due to mismatched model.')
    sleep(0)