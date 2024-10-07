from typing import Tuple, Union

import torch as ch
from toy_task.GMM.utils.torch_utils import torch_batched_trace


def maha(mean: ch.Tensor, old_mean: ch.Tensor, old_chol: ch.Tensor) -> ch.Tensor:
    diff = (mean - old_mean)[..., None]
    return ch.linalg.solve_triangular(old_chol, diff, upper=False).pow(2).sum([-2, -1])


def maha_sqrt(mean: ch.Tensor, old_mean: ch.Tensor, old_sqrt: ch.Tensor) -> ch.Tensor:
    diff = (mean - old_mean)[..., None]
    return (ch.linalg.solve(old_sqrt, diff) ** 2).sum([-2, -1])


def covariance(std: ch.Tensor) -> ch.Tensor:
    cov = std @ std.transpose(-1, -2)
    return cov


def precision(std: ch.Tensor) -> ch.Tensor:
    return ch.cholesky_solve(ch.eye(std.shape[-1], dtype=std.dtype, device=std.device), std, upper=False)


def gaussian_kl(mean, chol, mean_other, chol_other):
    k = mean.shape[-1]

    det_term = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    det_term_other = 2 * chol_other.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    cov = covariance(chol)
    prec_other = precision(chol_other)

    maha_part = .5 * maha(mean, mean_other, chol_other)
    trace_part = torch_batched_trace(prec_other @ cov)
    cov_part = .5 * (trace_part - k + det_term_other - det_term)

    return maha_part, cov_part


def gaussian_wasserstein_non_commutative(mean, sqrt, mean_old, sqrt_old) -> Union[Tuple[ch.Tensor, ch.Tensor],
                                                                    Tuple[ch.Tensor, ch.Tensor, ch.Tensor, ch.Tensor]]:
    """
    Compute mean part and cov part of W_2(p || q) with p,q ~ N(y, SS)
    This version DOES NOT assume commutativity of both distributions, i.e. covariance matrices.
    This is more general and does not make any assumptions.
    Args:
        mean and chol of current dist.
        mean_old and col_old of old dist.

    Returns: mean part of W2, cov part of W2

    """
    cov = sqrt @ sqrt

    batch_dim, dim = mean.shape

    mean_part = maha_sqrt(mean, mean_old, sqrt_old)

    # cov constraint scaled with precision of old dist
    # W2 objective for cov assuming normal W2 objective for mean
    identity = ch.eye(dim, dtype=sqrt.dtype, device=sqrt.device)
    sqrt_inv_other = ch.linalg.solve(sqrt_old, identity)
    c = sqrt_inv_other @ cov @ sqrt_inv_other

    # compute inner parenthesis of trace in W2,
    # Only consider lower triangular parts, given cov/sqrt(cov) is symmetric PSD.
    eigvals, _ = ch.linalg.eigh(c, UPLO='L')
    # make use of the following property to compute the trace of the root: 𝐴^2𝑥=𝐴(𝐴𝑥)=𝐴𝜆𝑥=𝜆(𝐴𝑥)=𝜆^2𝑥
    cov_part = torch_batched_trace(identity + c) - 2 * eigvals.sqrt().sum(1)

    return mean_part, cov_part
