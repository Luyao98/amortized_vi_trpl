# utils

import torch as ch
import torch.nn as nn
import numpy as np


def fill_triangular(x, upper=False):
    m = np.int32(x.shape[-1])
    # Formula derived by solving for n: m = n(n+1)/2.
    n = np.sqrt(0.25 + 2. * m) - 0.5
    if n != np.floor(n):
        raise ValueError('Input right-most shape ({}) does not '
                         'correspond to a triangular matrix.'.format(m))
    n = np.int32(n)
    new_shape = x.shape[:-1] + (n, n)

    ndims = len(x.shape)
    if upper:
        x_list = [x, ch.flip(x[..., n:], dims=[ndims - 1])]
    else:
        x_list = [x[..., n:], ch.flip(x, dims=[ndims - 1])]

    x = ch.cat(x_list, dim=-1).reshape(new_shape)
    x = ch.triu(x) if upper else ch.tril(x)
    return x


def fill_triangular_inverse(x, upper=False):
    n = np.int32(x.shape[-1])
    m = np.int32((n * (n + 1)) // 2)

    ndims = len(x.shape)
    if upper:
        initial_elements = x[..., 0, :]
        triangular_part = x[..., 1:, :]
    else:
        initial_elements = ch.flip(x[..., -1, :], dims=[ndims - 2])
        triangular_part = x[..., :-1, :]

    rotated_triangular_portion = ch.flip(ch.flip(triangular_part, dims=[ndims - 1]), dims=[ndims - 2])
    consolidated_matrix = triangular_part + rotated_triangular_portion

    end_sequence = consolidated_matrix.reshape(x.shape[:-2] + (n * (n - 1),))

    y = ch.cat([initial_elements, end_sequence[..., :m - n]], dim=-1)
    return y


def diag_bijector(f: callable, x):
    return x.tril(-1) + f(x.diagonal(dim1=-2, dim2=-1)).diag_embed() + x.triu(1)


def inverse_softplus(x):
    return (x.exp() - 1.).log()


def initialize_weights(model: nn.Module, initialization_type: str, scale: float = 2 ** 0.5, init_w=3e-3):
    for p in model.parameters():
        if initialization_type == "normal":
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


# def mahalanobis_distance(x, mean, precision):
#     # return shape (batch_size, 1)
#     diff = x - mean
#     diff = diff.unsqueeze(1)
#     diff_precision = ch.matmul(diff, precision)
#     maha_dist = ch.matmul(diff_precision, diff.transpose(-2, -1))
#     maha_dist = maha_dist.squeeze(-1)
#     return maha_dist


def maha(mean: ch.Tensor, old_mean: ch.Tensor, old_chol: ch.Tensor):
    diff = (mean - old_mean)[..., None]
    return ch.linalg.solve_triangular(old_chol, diff, upper=False)[0].pow(2).sum([-2, -1])


def covariance(std: ch.Tensor):
    std = std.view((-1,) + std.shape[-2:])
    return (std @ std.permute(0, 2, 1)).squeeze(0)


def torch_batched_trace(x) -> ch.Tensor:
    return ch.diagonal(x, dim1=-2, dim2=-1).sum(-1)


def precision(std: ch.Tensor):
    return ch.cholesky_solve(ch.eye(std.shape[-1], dtype=std.dtype, device=std.device), std, upper=False)


def gaussian_kl(mean, chol, mean_other, chol_other):
    k = mean.shape[-1]

    det_term = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    det_term_other = 2 * chol_other.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    cov = covariance(chol)
    prec_other = precision(chol_other)

    maha_part = .5 * maha(mean, mean_other, chol_other)
    # trace_part = (var * precision_other).sum([-1, -2])
    trace_part = torch_batched_trace(prec_other @ cov)
    cov_part = .5 * (trace_part - k + det_term_other - det_term)

    return maha_part, cov_part
