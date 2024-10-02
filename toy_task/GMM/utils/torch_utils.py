import torch as ch
import torch.nn as nn
import numpy as np


def tensorize(x, cpu=True, dtype=ch.float32):
    """
    Utility function for turning arrays into tensors
    Args:
        x: data
        cpu: Whether to generate a CPU or GPU tensor
        dtype: dtype of tensor

    Returns:
        gpu/cpu tensor of x with specified dtype
    """
    return cpu_tensorize(x, dtype) if cpu else gpu_tensorize(x, dtype)


def gpu_tensorize(x, dtype=None):
    """
    Utility function for turning arrays into cuda tensors
    Args:
        x: data
        dtype: dtype to generate

    Returns:
        gpu tensor of x
    """
    dtype = dtype if dtype else x.dtype
    return ch.tensor(x).type(dtype).cuda()


def cpu_tensorize(x, dtype=None):
    """
    Utility function for turning arrays into cpu tensors
    Args:
        x: data
        dtype: dtype to generate

    Returns:
        cpu tensor of x
    """
    dtype = dtype if dtype else x.dtype
    return ch.tensor(x).type(dtype)


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
        tril_matrix = diag_bijector(
            lambda z: diag_activation(z + inverse_softplus(init_std - minimal_std)) + minimal_std, tril_matrix
        )
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


def gmm_log_density_grad_point(
        target_gate: ch.Tensor,  # (n_components,)
        target_mean: ch.Tensor,  # (n_components, d_z)
        target_chol: ch.Tensor,  # (n_components, d_z, d_z)
        x: ch.Tensor  # (d_z,)
) -> ch.Tensor:
    """
    Compute the log density gradient of a single point `x` under a GMM with multiple components.

    Parameters:
    target_gate (torch.Tensor): Tensor of shape (n_components,) representing the log weights (unnormalized).
    target_mean (torch.Tensor): Tensor of shape (n_components, d_z) representing the mean of each Gaussian component.
    target_chol (torch.Tensor): Tensor of shape (n_components, d_z, d_z) representing the Cholesky decomposition of the covariance matrix of each component.
    x (torch.Tensor): Tensor of shape (d_z,) representing the input point.

    Returns:
    torch.Tensor: Tensor of shape (d_z,) representing the gradient of the log density at the point `x`.
    """
    # Number of components and dimension of data
    n_components, d_z = target_mean.shape

    # Compute precision matrices using Cholesky decomposition
    precision = ch.inverse(target_chol)  # shape (n_components, d_z, d_z)
    precision_t = ch.transpose(precision, -1, -2)  # Transpose the last two dimensions

    # Calculate log probabilities for each Gaussian component
    diffs = target_mean - x.unsqueeze(0)  # (n_components, d_z) - (d_z,) -> (n_components, d_z)
    mahalanobis_term = ch.einsum('ni,nij,nj->n', diffs, precision_t, diffs)  # (n_components,)
    log_prob_components = -0.5 * mahalanobis_term - ch.sum(ch.log(ch.diagonal(target_chol, dim1=1, dim2=2)),
                                                              dim=1)
    # Combine with log gates to get full log density
    log_component_densities = log_prob_components + target_gate
    log_density = ch.logsumexp(log_component_densities, dim=0)

    # Compute responsibilities
    log_responsibilities = log_component_densities - log_density  # (n_components,)
    responsibilities = ch.exp(log_responsibilities)  # (n_components,)

    # Compute the gradient
    grad_log_density = ch.zeros(d_z, device=x.device)
    for k in range(n_components):
        diff_k = diffs[k]  # (d_z,)
        grad_k = precision_t[k] @ diff_k  # (d_z,)
        grad_log_density += responsibilities[k] * grad_k  # Weight by responsibility

    return grad_log_density


def gradient_ascent_on_point(
        target_gate: ch.Tensor,  # (n_components,)
        target_mean: ch.Tensor,  # (n_components, d_z)
        target_chol: ch.Tensor,  # (n_components, d_z, d_z)
        x: ch.Tensor,  # (d_z,)
        learning_rate: float,  # Learning rate (alpha)
        n: int  # Number of iterations
) -> ch.Tensor:
    """
    Perform gradient ascent to update point `x` to move towards higher density regions of the GMM.

    Parameters:
    target_gate (torch.Tensor): Tensor of shape (n_components,) representing the log weights (unnormalized).
    target_mean (torch.Tensor): Tensor of shape (n_components, d_z) representing the mean of each Gaussian component.
    target_chol (torch.Tensor): Tensor of shape (n_components, d_z, d_z) representing the Cholesky decomposition of the covariance matrix of each component.
    x (torch.Tensor): Tensor of shape (d_z,) representing the input point.
    learning_rate (float): Step size for gradient ascent.
    n (int): Number of gradient ascent steps.

    Returns:
    torch.Tensor: Updated tensor of shape (d_z,) representing the new location of the point `x`.
    """
    for i in range(n):
        # Compute the gradient of the log density at the point x
        grad = gmm_log_density_grad_point(target_gate, target_mean, target_chol, x)

        # Update the point x using gradient ascent
        x = x + learning_rate * grad
    return x