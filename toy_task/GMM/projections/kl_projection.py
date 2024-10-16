import cpp_projection

import numpy as np
import torch as ch
from toy_task.GMM.utils.torch_utils import get_numpy

class KLProjection(ch.autograd.Function):
    projection_op = None

    @staticmethod
    def get_projection_op(batch_shape, dim, max_eval=100):
        if not KLProjection.projection_op:
            KLProjection.projection_op = \
                cpp_projection.BatchedProjection(batch_shape, dim, eec=False, constrain_entropy=False, max_eval=max_eval)
        return KLProjection.projection_op

    @staticmethod
    def forward(ctx, *args, **kwargs):
        p, q, eps = args
        mean, cov = p  # target distribution
        old_mean, old_cov = q  # old distribution
        mean = get_numpy(mean)
        cov = get_numpy(cov)
        old_mean = get_numpy(old_mean)
        old_cov = get_numpy(old_cov)

        batch_shape, dim = mean.shape
        beta = np.nan
        epss = eps * np.ones(batch_shape)
        betas = beta * np.ones(batch_shape)

        p_op = KLProjection.get_projection_op(batch_shape, dim)
        proj_mean, proj_cov = p_op.forward(epss, betas, old_mean, old_cov, mean, cov)
        proj_chol = np.linalg.cholesky(proj_cov)
        ctx.proj = p_op

        return ch.from_numpy(proj_mean), ch.from_numpy(proj_chol)

    @staticmethod
    def backward(ctx, *grad_outputs):
        projection_op = ctx.proj
        d_means, d_covs = grad_outputs
        df_means, df_covs = projection_op.backward(d_means.numpy(), d_covs.numpy())
        return ch.tensor(df_means), ch.tensor(df_covs)
