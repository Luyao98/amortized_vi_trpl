import cpp_projection
import numpy as np
import torch as ch
from typing import Any, Tuple

from trust_region_projections.model.conditionalgaussian_model import ConditionalGaussianModel
from trust_region_projections.projection.base_projection_layer import BaseProjectionLayer, mean_projection
from trust_region_projections.utils.projection_utils import gaussian_kl
from trust_region_projections.utils.torch_utils import get_numpy


class KLProjectionLayer(BaseProjectionLayer):

    def _trust_region_projection(self, policy: ConditionalGaussianModel, p: Tuple[ch.Tensor, ch.Tensor],
                                 q: Tuple[ch.Tensor, ch.Tensor], eps: ch.Tensor, eps_cov: ch.Tensor, **kwargs):
        """
        Runs KL projection layer and constructs cholesky of covariance
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: (modified) kl bound/ kl bound for mean part
            eps_cov: (modified) kl bound for cov part
            **kwargs:

        Returns:
            projected mean, projected cholesky
        """
        mean, std = p
        old_mean, old_std = q

        # project mean with closed form
        mean_part, _ = gaussian_kl(policy, p, q)
        proj_mean = mean_projection(mean, old_mean, mean_part, eps)

        # transfer cholesky into full cov to adapt cpp_projection
        cov = policy.covariance(std)
        # old_cov = policy.covariance(old_std)

        # transfer back to chol matrix
        proj_cov = KLProjectionGradFunctionCovOnly.apply(std, cov, old_std, eps_cov)
        proj_std = ch.cholesky(proj_cov, upper=False)

        return proj_mean, proj_std


class KLProjectionGradFunctionCovOnly(ch.autograd.Function):
    projection_op = None

    @staticmethod
    def get_projection_op(batch_shape, dim, max_eval=100):
        if not KLProjectionGradFunctionCovOnly.projection_op:
            # garantee only one isntance exisits
            KLProjectionGradFunctionCovOnly.projection_op = \
                cpp_projection.BatchedCovOnlyProjection(batch_shape, dim, max_eval=max_eval)
        return KLProjectionGradFunctionCovOnly.projection_op

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        std, cov, old_chol, eps_cov = args

        batch_shape = std.shape[0]
        dim = std.shape[-1]

        std_np = get_numpy(std)
        cov_np = get_numpy(cov)
        old_chol = get_numpy(old_chol)
        eps = get_numpy(eps_cov) * np.ones(batch_shape)

        p_op = KLProjectionGradFunctionCovOnly.get_projection_op(batch_shape, dim)
        ctx.proj = p_op

        proj_cov = p_op.forward(eps, old_chol, std_np, cov_np)

        return std.new(proj_cov)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        projection_op = ctx.proj
        d_cov, = grad_outputs

        d_cov_np = get_numpy(d_cov)
        d_cov_np = np.atleast_2d(d_cov_np)
        df_covs = projection_op.backward(d_cov_np)
        df_covs = np.atleast_2d(df_covs)

        return d_cov.new(df_covs), None, None