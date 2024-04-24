import torch as ch


def maha(mean: ch.Tensor, old_mean: ch.Tensor, old_chol: ch.Tensor) -> ch.Tensor:
    diff = (mean - old_mean)[..., None]
    return ch.linalg.solve_triangular(old_chol, diff, upper=False)[0].pow(2).sum([-2, -1])


def covariance(std: ch.Tensor) -> ch.Tensor:
    std = std.view((-1,) + std.shape[-2:])
    return (std @ std.permute(0, 2, 1)).squeeze(0)


def torch_batched_trace(x: ch.Tensor) -> ch.Tensor:
    return ch.diagonal(x, dim1=-2, dim2=-1).sum(-1)


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