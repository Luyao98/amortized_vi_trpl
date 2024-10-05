from typing import Tuple, Union

import numpy as np
from einops import reduce, rearrange, repeat, einsum
import torch

from daft.src.gmm_util.weighted_logsumexp import weighted_logsumexp


def prec_to_prec_chol(prec: torch.Tensor) -> torch.Tensor:
    """
    :param prec: tensor shape ([batch_dims], n_components, d_z, d_z)
    :return: tensor shape ([batch_dims], n_components, d_z, d_z)
    """

    precs_chol = torch.linalg.cholesky(prec)
    return precs_chol


def prec_to_cov_chol(prec: torch.Tensor) -> torch.Tensor:
    """
    :param prec: tensor shape ([batch_dims], n_components, d_z, d_z)
    :return: tensor shape ([batch_dims], n_components, d_z, d_z)
    """
    Lf = torch.linalg.cholesky(torch.flip(prec, (-2, -1)))
    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    Id = torch.eye(prec.shape[-1], dtype=prec.dtype, device=prec.device)
    L = torch.linalg.solve_triangular(L_inv, Id, upper=False)
    return L


def cov_chol_to_cov(cov_chol: torch.Tensor) -> torch.Tensor:
    """
    :param cov_chol: tensor shape ([batch_dims], n_components, d_z, d_z)
    :return: tensor shape ([batch_dims], n_components, d_z, d_z)
    """
    cov = torch.matmul(cov_chol, cov_chol.transpose(-1, -2))
    return cov


def sample_gmm(
    n_samples: int, log_w: torch.Tensor, mean: torch.Tensor, cov_chol: torch.Tensor
) -> torch.Tensor:
    """
    Samples from the whole GMM (all components) based on the log_w, means and precs
    :param n_samples: int
    :param log_w: tensor shape ([batch_dims], n_components)
    :param mean: tensor shape ([batch_dims], n_components, d_z)
    :param cov_chol: tensor shape ([batch_dims], n_components, d_z, d_z)
    :return: tensor shape (n_samples, [batch_dims], d_z)
    """
    # sample gmm
    # TODO: this is quite slow

    samples = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(logits=log_w),
        component_distribution=torch.distributions.MultivariateNormal(
            loc=mean, scale_tril=cov_chol
        ),
    ).sample((n_samples,))
    return samples


def gmm_log_component_densities(
    z: torch.Tensor, mean: torch.Tensor, prec_chol: torch.Tensor
) -> torch.Tensor:
    """
    Compute log density from each component separately
    :param z: Tensor shape (n_samples, [batch_dims], d_z)
    :param mean:  tensor shape ([batch_dims], n_components, d_z)
    :param prec_chol: tensor shape ([batch_dims], n_components, d_z, d_z)
    :return: tensor shape (n_samples [batch_dims], n_components]
    """

    d_z = z.shape[-1]

    # compute log component densities
    # diff z - mean
    diffs = rearrange(z, "n_samples ... d_z -> n_samples ... 1 d_z") - rearrange(
        mean, "... n_comps d_z -> 1 ... n_comps d_z"
    )
    # diffs has now shape (n_samples, [batch_dims], n_components, d_z)
    broadcasted_prec_chol = rearrange(prec_chol, "... n_comps d_z d_z2 -> 1 ... n_comps d_z d_z2")
    # diffs @ L
    diff_times_prec_chol = einsum(diffs, broadcasted_prec_chol, "... dz, ... dz dz2 -> ... dz2")
    # (diffs @ L).T @ (diffs @ L)
    mahalas = -0.5 * einsum(diff_times_prec_chol, diff_times_prec_chol, "... dz, ... dz -> ...")

    # - 0.5 log(det(cov)) = log(det(L)) for L = prec_chol
    det_part = reduce(
        torch.log(torch.diagonal(prec_chol, dim1=-2, dim2=-1)),
        "... n_comps d_z -> ... n_comps",
        "sum",
    )
    # -d_z/2 * log(2 * pi)
    const_part = -0.5 * d_z * torch.log(2 * torch.tensor(np.pi))
    log_component_densities = const_part + det_part + mahalas

    return log_component_densities


def gmm_log_density(
    log_w: torch.Tensor,
    log_component_densities: torch.Tensor,
):
    """
    Compute log density from the whole GMM using the log component densities
    :param log_w: weights of the GMM, tensor shape ([batch_dims], n_components)
    :param log_component_densities: previously computed log component densities, tensor shape (n_samples, [batch_dims], n_components)
    :return: log densities of the GMM, tensor shape (n_samples, [batch_dims])
    """
    # compute log density
    log_joint_densities = log_component_densities + rearrange(log_w, "... n_comps -> 1 ... n_comps")
    # log_joint densities has shape (n_samples, [batch_dims], n_components)
    log_density = torch.logsumexp(log_joint_densities, dim=-1)
    return log_density


def gmm_log_responsibilities(
    log_w: torch.Tensor,
    log_component_densities: torch.Tensor,
    log_density: torch.Tensor,
):
    """
    Compute log responsibilities of the GMM, that is log p(o|x) = log p(x|o) + log p(o) - log p(x)
    :param log_w: weights of the GMM, tensor shape ([batch_dims], n_components)
    :param log_component_densities: tensor shape (n_samples, [batch_dims], n_components)
    :param log_density: tensor shape (n_samples, [batch_dims])
    :return: tensor shape (n_samples, [batch_dims], n_components)
    """
    # compute log density and responsibilities
    log_joint_densities = log_component_densities + rearrange(
        log_w, "... n_components -> 1 ... n_components"
    )
    log_responsibilities = log_joint_densities - rearrange(
        log_density, "n_samples ... -> n_samples ... 1"
    )

    return log_responsibilities


def gmm_log_density_grad(
    z: torch.Tensor,
    log_w: torch.Tensor,
    mean: torch.Tensor,
    prec: torch.Tensor,
    prec_chol: torch.Tensor,
    compute_grad: bool,
) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    """
    Compute log density and optionally the gradient of the log density of a GMM
    :param z: Tensor shape (n_samples, [batch_dims], d_z)
    :param log_w: Tensor shape ([batch_dims], n_components)
    :param mean: tensor shape ([batch_dims], n_components, d_z)
    :param prec: tensor shape ([batch_dims], n_components, d_z, d_z)
    :param cov_chol: tensor shape ([batch_dims], n_components, d_z, d_z)
    :param compute_grad: bool
    :return: Tuple with log density with shape (n_samples, [batch_dims]), and the gradient of the log density with shape
            (n_samples, [batch_dims], d_z), if required
    """
    log_component_densities = gmm_log_component_densities(
        z=z,
        mean=mean,
        prec_chol=prec_chol,
    )
    log_density = gmm_log_density(
        log_w=log_w,
        log_component_densities=log_component_densities,
    )

    # compute gradient
    if compute_grad:
        # compute log density and responsibilities
        log_responsibilities = gmm_log_responsibilities(
            log_w=log_w,
            log_component_densities=log_component_densities,
            log_density=log_density,
        )
        # compute gradient d/dz log q(z)
        # changed order in difference to avoid the minus at the beginning
        diffs = rearrange(mean, "... n_comps d_z -> 1 ... n_comps d_z") - rearrange(
            z, "n_samples ... d_z -> n_samples ... 1 d_z"
        )
        # diffs has shape (n_samples, [batch_dims], n_components, d_z)
        prec_times_diff = einsum(
            rearrange(prec, "... -> 1 ..."),
            diffs,
            "n_samples ... n_comps dz dz2, n_samples ... n_comps dz2 -> n_samples ... n_comps dz",
        )
        log_density_grad, sign = weighted_logsumexp(
            logx=rearrange(log_responsibilities, "n_samples ... n_comps -> n_samples ... n_comps 1")
            + torch.log(torch.abs(prec_times_diff)),
            w=torch.sign(prec_times_diff),
            dim=-2,
            return_sign=True,
        )
        log_density_grad = sign * torch.exp(log_density_grad)
    else:
        log_density_grad = None
    return log_density, log_density_grad
