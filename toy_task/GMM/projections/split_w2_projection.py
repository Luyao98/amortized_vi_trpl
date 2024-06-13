import torch as ch
from toy_task.GMM.utils.projection_utils import gaussian_wasserstein_non_commutative


def mean_projection(mean, old_mean, maha, eps_mu):

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


def split_w2_projection(mean, sqrt, old_mean, old_sqrt, eps_mean, eps_cov):
    batch_shape = mean.shape[0]

    ####################################################################################################################
    # precompute mean and cov part of W2, which are used for the projection.
    # Both parts differ based on precision scaling.
    # If activated, the mean part is the maha distance and the cov has a more complex term in the inner parenthesis.
    mean_part, cov_part = gaussian_wasserstein_non_commutative(mean, sqrt, old_mean, old_sqrt)

    ####################################################################################################################
    # project mean (w/ or w/o precision scaling)
    proj_mean = mean_projection(mean, old_mean, mean_part, eps_mean)

    ####################################################################################################################
    # project covariance (w/ or w/o precision scaling)

    cov_mask = cov_part > eps_cov
    if cov_mask.any():
        # Gradient issue with ch.where, it executes both paths and gives NaN gradient.
        eta = ch.ones(batch_shape, dtype=sqrt.dtype, device=sqrt.device)
        eta[cov_mask] = ch.sqrt(cov_part[cov_mask] / eps_cov) - 1.
        eta = ch.max(-eta, eta)

        safe_eta = eta + (eta == 0) * 1e-16

        new_sqrt = (sqrt + ch.einsum('i,ijk->ijk', eta, old_sqrt)) / (1. + safe_eta)[..., None, None]

        proj_sqrt = sqrt.clone()
        proj_sqrt[cov_mask] = new_sqrt[cov_mask]
        # new_sqrt = (sqrt + ch.einsum('i,ijk->ijk', eta, old_sqrt)) / (1. + eta + 1e-16)[..., None, None]
        # proj_sqrt = ch.where(cov_mask[..., None, None], new_sqrt, sqrt)
        # proj_sqrt = sqrt.clone()
        # proj_sqrt[cov_mask] = new_sqrt[cov_mask]
        #
        # is_nan = proj_sqrt.mean([-2, -1]).isnan() * cov_mask
        # if is_nan.any():
        #     proj_sqrt[is_nan] = old_sqrt[is_nan]
        #     print("NaN in sqrt projection")


    # if cov_mask.any():
    #     # gradient issue with ch.where, it executes both paths and gives NaN gradient.
    #     eta = ch.ones(batch_shape, dtype=sqrt.dtype, device=sqrt.device)
    #     eta[cov_mask] = ch.sqrt(cov_part[cov_mask] / eps_cov) - 1.
    #     eta = ch.max(-eta, eta)
    #
    #     new_sqrt = (sqrt + ch.einsum('i,ijk->ijk', eta, old_sqrt)) / (1. + eta + 1e-16)[..., None, None]
    #     proj_sqrt = ch.where(cov_mask[..., None, None], new_sqrt, sqrt)
    else:
        proj_sqrt = sqrt

    return proj_mean, proj_sqrt
