import cpp_projection

import numpy as np
import torch as ch


def mean_projection(mean, old_mean, old_chol, eps_mu):

    mean_diff = (mean - old_mean).unsqueeze(-1)
    maha = ch.linalg.solve_triangular(old_chol, mean_diff, upper=False).pow(2).sum([-2, -1])
    batch_shape = mean.shape[:-1]
    mask = maha > eps_mu

    # if nothing has to be projected skip computation
    if mask.any():
        omega = ch.ones(batch_shape, dtype=mean.dtype, device=mean.device)
        omega[mask] = ch.sqrt(maha[mask] / eps_mu) - 1.
        omega = ch.max(-omega, omega)[..., None]

        m = (mean + omega * old_mean) / (1 + omega + 1e-16)
        proj_mean = ch.where(mask[..., None], m, mean)
    else:
        proj_mean = mean

    return proj_mean


class CovKLProjection(ch.autograd.Function):
    sop = None

    @staticmethod
    def get_sop(batch_shape, dim, max_eval=100):
        if not CovKLProjection.sop:
            CovKLProjection.sop = \
                cpp_projection.BatchedCovOnlyProjection(batch_shape, dim, max_eval=max_eval)
        return CovKLProjection.sop

    @staticmethod
    def forward(ctx, *args, **kwargs):
        old_chol, target_chol, target_cov, eps = args
        old_chol_np = old_chol.numpy()
        target_chol_np = target_chol.numpy()
        target_cov_np = target_cov.numpy()

        batch_shape = old_chol.shape[0]
        dim = old_chol.shape[-1]

        epss = eps * np.ones(batch_shape)

        p_sop = CovKLProjection.get_sop(batch_shape, dim)
        proj_cov = p_sop.forward(epss, old_chol_np, target_chol_np, target_cov_np)
        ctx.proj = p_sop

        return target_cov.new(proj_cov)

    @staticmethod
    def backward(ctx, *grad_outputs):
        sop = ctx.proj
        d_covs, = grad_outputs
        d_covs_np = d_covs.numpy()
        d_covs_np = np.atleast_2d(d_covs_np)
        df_covs = sop.backward(d_covs_np)
        df_covs = np.atleast_2d(df_covs)
        return d_covs.new(df_covs), None, None, None

