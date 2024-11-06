import cpp_projection

import numpy as np
import torch as ch
from joblib import Parallel, delayed

from toy_task.GMM.utils.torch_utils import get_numpy
from toy_task.GMM.utils.projection_utils import gaussian_kl


def parallel_split_kl_projection(mean_pred, chol_pred, b_mean_old, b_chol_old, eps_mean, eps_cov, minibatch_size):
    """
    does't work due to split computational graph
    Parallel implementation of the split KL projection using joblib for efficiency and avoiding CPU problem.

    Parameters:
    - mean_pred (torch.Tensor): The predicted means with shape (batch_size, n_components, dz).
    - chol_pred (torch.Tensor): The predicted Cholesky decompositions with shape (batch_size, n_components, dz, dz).
    - b_mean_old (torch.Tensor): The old means used for comparison with shape (batch_size, n_components, dz).
    - b_chol_old (torch.Tensor): The old Cholesky decompositions with shape (batch_size, n_components, dz, dz).
    - eps_mean (float): The threshold for mean projection.
    - eps_cov (float): The threshold for covariance projection.
    - minibatch_size (int): The size of minibatches for parallel processing.

    Returns:
    - mean_proj (torch.Tensor): The projected means with shape (batch_size, n_components, dz).
    - chol_proj (torch.Tensor): The projected Cholesky decompositions with shape (batch_size, n_components, dz, dz).
    """
    batch_size, n_components, dz = mean_pred.shape

    def compute_projection(mean, chol, old_mean, old_chol, eps_mean, eps_cov):
        """
        hook function.

        Parameters:
        - mean, chol, old_mean, old_chol, eps_mean, eps_cov: Same as the main function.

        Returns:
        - mean_proj_flatten_minibatch (torch.Tensor): Flattened projected means for the minibatch.
        - chol_proj_flatten_minibatch (torch.Tensor): Flattened projected Cholesky decompositions for the minibatch.
        """
        mean_proj_flatten_minibatch, chol_proj_flatten_minibatch = split_kl_projection(
            mean.view(-1, dz), chol.view(-1, dz, dz),
            old_mean.view(-1, dz).clone().detach(), old_chol.view(-1, dz, dz).clone().detach(),
            eps_mean, eps_cov
        )
        return mean_proj_flatten_minibatch, chol_proj_flatten_minibatch

    # Use joblib for parallel computation
    results = Parallel(n_jobs=-1)(delayed(compute_projection)(
        mean_pred[i:i + minibatch_size], chol_pred[i:i + minibatch_size],
        b_mean_old[i:i + minibatch_size], b_chol_old[i:i + minibatch_size],
        eps_mean, eps_cov
    ) for i in range(0, batch_size, minibatch_size))

    # Collect and concatenate results
    mean_proj_list, chol_proj_list = zip(*results)
    mean_proj_flatten = ch.cat(mean_proj_list, dim=0)
    chol_proj_flatten = ch.cat(chol_proj_list, dim=0)

    return mean_proj_flatten.view(batch_size, n_components, dz), chol_proj_flatten.view(batch_size, n_components, dz,
                                                                                        dz)


def split_kl_projection(mean, chol, old_mean, old_chol, eps_mean, eps_cov):
    """
    Computes the split KL divergence-based projection for mean and covariance.

    Parameters:
    - mean (torch.Tensor): The predicted means with shape (num_samples, dz).
    - chol (torch.Tensor): The predicted Cholesky decompositions with shape (num_samples, dz, dz).
    - old_mean (torch.Tensor): The old means with shape (num_samples, dz).
    - old_chol (torch.Tensor): The old Cholesky decompositions with shape (num_samples, dz, dz).
    - eps_mean (float): Threshold for the mean projection.
    - eps_cov (float): Threshold for the covariance projection.

    Returns:
    - mean_proj (torch.Tensor): The projected means with the same shape as `mean`.
    - chol_proj (torch.Tensor): The projected Cholesky decompositions with the same shape as `chol`.
    """
    assert mean.dim() == 2
    assert chol.dim() == 3
    assert old_mean.dim() == 2
    assert old_chol.dim() == 3

    maha_part, cov_part = gaussian_kl(mean, chol, old_mean, old_chol)

    # Project the mean
    mean_proj = mean_projection(mean, old_mean, maha_part, eps_mean)

    # Project the covariance
    cov = chol @ chol.transpose(-1, -2)
    try:
        mask = cov_part > eps_cov
        chol_proj = ch.zeros_like(chol)
        chol_proj[~mask] = chol[~mask]
        if mask.any():
            cov_proj = CovKLProjection.apply(old_chol, chol.detach(), cov, eps_cov)

            # Check for NaNs in the projected covariance and use the old Cholesky if necessary
            is_nan = cov_proj.mean([-2, -1]).isnan() * mask
            if is_nan.any():
                chol_proj[is_nan] = old_chol[is_nan]
                mask *= ~is_nan

            chol_proj[mask], failed_mask = ch.linalg.cholesky_ex(cov_proj[mask])
            failed_mask = failed_mask.type(ch.bool)
            if ch.any(failed_mask):
                chol_proj[failed_mask] = old_chol[failed_mask]
    except Exception as e:
        import logging
        logging.error('Projection failed, taking old cholesky for projection.')
        print("Projection failed, taking old cholesky for projection.")
        chol_proj = old_chol
        # raise e

    return mean_proj, chol_proj


def mean_projection(mean, old_mean, maha, eps_mu):
    """
    Projects the mean based on the Mahalanobis distance.

    Parameters:
    - mean (torch.Tensor): The predicted means with shape (num_samples, dz).
    - old_mean (torch.Tensor): The old means with shape (num_samples, dz).
    - maha (torch.Tensor): The Mahalanobis distance with shape (num_samples,).
    - eps_mu (float): The threshold for mean projection.

    Returns:
    - proj_mean (torch.Tensor): The projected means with the same shape as `mean`.
    """
    batch_shape = mean.shape[:-1]
    mask = maha > eps_mu

    # Skip computation if no projection is needed
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
    def get_sop(batch_shape, dim, max_eval=200):
        if not CovKLProjection.sop:
            CovKLProjection.sop = \
                cpp_projection.BatchedCovOnlyProjection(batch_shape, dim, max_eval=max_eval)
        return CovKLProjection.sop

    @staticmethod
    def forward(ctx, *args, **kwargs):
        old_chol, chol, cov, eps = args
        old_chol_np = get_numpy(old_chol)
        chol_np = get_numpy(chol)
        cov_np = get_numpy(cov)

        batch_shape = old_chol.shape[0]
        dim = old_chol.shape[-1]

        epss = eps * np.ones(batch_shape)

        p_sop = CovKLProjection.get_sop(batch_shape, dim)
        ctx.proj = p_sop
        proj_cov = p_sop.forward(epss, old_chol_np, chol_np, cov_np)

        return cov.new(proj_cov)

    @staticmethod
    def backward(ctx, *grad_outputs):
        sop = ctx.proj
        d_covs, = grad_outputs
        d_covs_np = get_numpy(d_covs)
        d_covs_np = np.atleast_2d(d_covs_np)
        df_chols= sop.backward(d_covs_np)
        df_chols = np.atleast_2d(df_chols)
        return d_covs.new(df_chols), None, None, None
