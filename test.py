from exp_models import test_denoiser
from exp_data import Mixtures, data_te_specialist
from time import sleep
import json
import re
import sys

if len(sys.argv) > 1:
    folders = sys.argv[1:]
else:
    folders = [
        '/N/u/asivara/2022-jstsp/sisdri/Dec10_17-36-37_grunet_large_np_nc_sp019/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec10_20-28-03_grunet_large_np_nc_sp026/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec10_22-40-20_grunet_large_np_nc_sp039/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec11_01-15-09_grunet_large_np_nc_sp040/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec11_04-43-46_grunet_large_np_nc_sp078/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec11_06-56-10_grunet_large_np_nc_sp083/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec11_09-03-00_grunet_large_np_nc_sp087/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec11_12-01-13_grunet_large_np_nc_sp089/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec11_14-14-46_grunet_large_np_nc_sp118/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec11_16-27-07_grunet_large_np_nc_sp125/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec11_19-03-54_grunet_large_np_nc_sp163/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec11_21-17-38_grunet_large_np_nc_sp196/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec12_00-00-07_grunet_large_np_nc_sp198/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec12_04-53-35_grunet_large_np_nc_sp200/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec12_07-35-00_grunet_large_np_nc_sp201/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec12_11-12-36_grunet_large_np_nc_sp250/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec12_14-24-35_grunet_large_np_nc_sp254/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec12_16-37-05_grunet_large_np_nc_sp307/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec12_20-13-38_grunet_large_np_nc_sp405/early_stopping.txt',
        '/N/u/asivara/2022-jstsp/sisdri/Dec12_22-23-18_grunet_large_np_nc_sp446/early_stopping.txt',
    ]

for f in folders:
    try:
        path = f.strip()
        path = path.replace('early_stopping.txt', '')
        accumulation = bool('tasnet' in f)
        if '_sp' in f:
            sp_id = int(re.match(r'.*_sp(\d\d\d).*', f).group(1))
            dataset = Mixtures(
                sp_id, 'test', split_mixture='test', snr_mixture=(-5,5))
            result = test_denoiser(path,
                                   data_te=dataset,
                                   accumulation=accumulation,
                                   use_last=False)
        else:
            result = test_denoiser(path,
                                   data_te=data_te_specialist,
                                   accumulation=accumulation,
                                   use_last=False)
        print(json.dumps(result, indent=1))
    except RuntimeError as e:
        if 'state_dict' in str(e):
            print(f'Skipping {f} due to mismatched model.')
    sleep(0)