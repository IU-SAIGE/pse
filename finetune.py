import sys
from run import finetune_denoiser

if __name__ == '__main__':
    finetune_denoiser(dataset_duration=float(sys.argv[1]),
                      checkpoint_locations=sys.argv[2:])

