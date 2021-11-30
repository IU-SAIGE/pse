from exp_models import test_denoiser
from exp_data import Mixtures
from time import sleep
import re

folders = [
'Nov30_01-33-59_convtasnet_small_sp019',
'Nov30_01-34-09_convtasnet_small_sp019_dp',
]

for f in sorted(folders):
    try:
        accumulation = bool('tasnet' in f)
        if '_sp' in f:
            sp_id = int(re.match(r'.*_sp(\d\d\d).*', f).group(1))
            dataset = Mixtures(
                sp_id, 'test', split_mixture='test', snr_mixture=(-5,5))
            result = test_denoiser('runs/'+f,
                                   data_te=dataset,
                                   accumulation=accumulation)
        else:
            result = test_denoiser('runs/'+f, accumulation=accumulation)
        print(result)
    except RuntimeError as e:
        if 'state_dict' in str(e):
            print(f'Skipping {f} due to mismatched model.')
    sleep(0.1)