import numpy as np
import os
import torch
import torch.nn.functional
import yaml


class EarlyStopping(Exception):
    pass


class SmokeTest(Exception):
    pass


class ExperimentError(Exception):
    pass


def make_2d(x: torch.Tensor):
    """Normalize shape of `x` to two dimensions: [batch, time]."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    elif x.ndim == 3:
        return x.squeeze(1)
    else:
        if x.ndim != 2: raise ValueError('Could not force 2d.')
        return x


def make_3d(x: torch.Tensor):
    """Normalize shape of `x` to three dimensions: [batch, n_chan, time]."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        if x.ndim != 3: raise ValueError('Could not force 3d.')
        return x


def pad_x_to_y(x: torch.Tensor, y: torch.Tensor, axis: int = -1):
    """Right-pad or right-trim first argument to have same size as second argument
    Args:
        x (torch.Tensor): Tensor to be padded.
        y (torch.Tensor): Tensor to pad `x` to.
        axis (int): Axis to pad on.
    Returns:
        torch.Tensor, `x` padded to match `y`'s shape.
    """
    if axis != -1:
        raise NotImplementedError
    inp_len = y.shape[axis]
    output_len = x.shape[axis]
    return torch.nn.functional.pad(x, [0, inp_len - output_len])


def shape_reconstructed(reconstructed: torch.Tensor, size: torch.Tensor):
    """Reshape `reconstructed` to have same size as `size`
    Args:
        reconstructed (torch.Tensor): Reconstructed waveform
        size (torch.Tensor): Size of desired waveform
    Returns:
        torch.Tensor: Reshaped waveform
    """
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed


def get_config_from_yaml(yaml_filepath: str):

    if not os.path.exists(yaml_filepath):
        raise OSError(f'{yaml_filepath} not found')

    config = {}
    with open(yaml_filepath) as fp:
        config = yaml.safe_load(fp)
        nonlist_keys = (
            'available_devices',
            'num_gpus_per_experiment',
            'num_cpus_per_experiment',
            'output_folder',
            'folder_librispeech',
            'folder_fsd50k',
            'folder_musan',
            'sample_rate',
            'example_duration',
        )
        for k in config.keys():
            if k not in nonlist_keys:
                if not isinstance(config[k], list):
                    config[k] = [config[k],]

    return config
