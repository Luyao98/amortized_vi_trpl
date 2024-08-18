import torch as ch
import torch.nn as nn
import numpy as np
import random


def initialize_weights(model: nn.Module, initialization_type: str, scale: float = 2 ** 0.5, init_w=0.01,
                       preserve_bias_layers=None):
    if preserve_bias_layers is None:
        preserve_bias_layers = []

    for name, p in model.named_parameters():
        # initialize the specified layers
        if any(layer in name for layer in preserve_bias_layers):
            if len(p.data.shape) >= 2:
                ch.nn.init.zeros_(p)
            continue

        # initialize other parameters
        elif initialization_type == "normal":
            if len(p.data.shape) >= 2:
                p.data.normal_(init_w)  # 0.01
            else:
                p.data.zero_()
        elif initialization_type == "uniform":
            if len(p.data.shape) >= 2:
                p.data.uniform_(-init_w, init_w)
            else:
                p.data.zero_()
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_normal_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                nn.init.orthogonal_(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError(
                "Not a valid initialization type. Choose one of 'normal', 'uniform', 'xavier', and 'orthogonal'")


def add_value_to_diag(diag_elements):
    return diag_elements + 1e-5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    ch.manual_seed(seed)
    ch.cuda.manual_seed_all(seed)
    ch.backends.cudnn.deterministic = True
    ch.backends.cudnn.benchmark = False


def generate_init_biases(n_components, dim, scale):
    if dim == 2:
        angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
        init_bias_mean_list = [[scale * np.cos(angle), scale * np.sin(angle)] for angle in angles]
    elif dim == 10:
        # dummy version for testing
        angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
        init_bias_mean_list = [[scale * np.cos(angle), scale * np.sin(angle),
                                scale * np.cos(angle), scale * np.sin(angle),
                                scale * np.cos(angle), scale * np.sin(angle),
                                scale * np.cos(angle), scale * np.sin(angle),
                                scale * np.cos(angle), scale * np.sin(angle)] for angle in angles]
    else:
        raise ValueError(f"Invalid dim {dim}. Now only support 2 or 10.")
    return init_bias_mean_list
