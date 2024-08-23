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


def fill_triangular_gmm(chols, n_components, init_std):
    minimal_std = 1e-3
    diag_activation = nn.Softplus()

    tril_matrices = []
    for i in range(n_components):
        chol_vec = chols[:, i, :]
        tril_matrix = fill_triangular(chol_vec)
        tril_matrix = diag_bijector(lambda z: diag_activation(z + inverse_softplus(init_std - minimal_std)) + minimal_std, tril_matrix)
        tril_matrices.append(tril_matrix)

    tril_matrices_stacked = ch.stack(tril_matrices, dim=1)
    return tril_matrices_stacked


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


def get_numpy(x):
    return x.cpu().detach().numpy()


def torch_batched_trace(x) -> ch.Tensor:
    """
    Compute trace in n,m of batched matrix
    Args:
        x: matrix with shape [a,...l, n, m]

    Returns: trace with shape [a,...l]

    """
    return ch.diagonal(x, dim1=-2, dim2=-1).sum(-1)


def gumbel_softmax_sample(logits, temperature=1.0):
    gumbel_noise = -ch.log(-ch.log(ch.rand_like(logits) + 1e-20) + 1e-20)
    y = logits + gumbel_noise
    return ch.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1.0, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = ch.zeros_like(y)
        y_hard.scatter_(1, y.argmax(dim=1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y