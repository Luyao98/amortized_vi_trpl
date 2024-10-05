import math
from typing import Union
from typing import Tuple

import numpy as np
import torch

from daft.src.gmm_util.gmm import GMM
from daft.src.multi_daft_vi.util_lnpdf import mini_batch_function_no_grad, mini_batch_function_grad

class LNPDF:
    def mini_batch_log_density(
        self,
        z: torch.Tensor,
        mini_batch_size: int,
        compute_grad: bool = False,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        z: (num_samples, batch_dim, d_z)
        compute_grad: whether to compute the gradient of the log density w.r.t. z
        mini_batch_size: size of the mini batches to split up the num_samples dimension
        Returns:
            log_density: (num_samples, batch_dim)
            grad: (num_samples, batch_dim, d_z) or None
        """
        if compute_grad:
            return mini_batch_function_grad(lambda z: self.log_density(z, compute_grad=True), z, mini_batch_size)
        else:
            return mini_batch_function_no_grad(lambda z: self.log_density(z, compute_grad=False), z, mini_batch_size)

    def log_density(
        self,
        z: torch.Tensor,
        compute_grad: bool = False,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        z: (num_samples, batch_dim, d_z)
        compute_grad: whether to compute the gradient of the log density w.r.t. z
        Returns:
            log_density: (num_samples, batch_dim)
            grad: (num_samples, batch_dim, d_z) or None
        """
        raise NotImplementedError

    def get_num_dimensions(self):
        raise NotImplementedError

    def can_sample(self):
        return False

    def sample(self, n: int):
        raise NotImplementedError


class GmmLNPDF(LNPDF):
    """
    Dummy tgt distribution for testing. This is internally a GMM.
    """

    def __init__(
        self,
        target_weights: np.ndarray,
        target_means: np.ndarray,
        target_covars: np.ndarray,
    ):
        prec = torch.linalg.inv(torch.tensor(target_covars.astype(np.float32)))

        self.target_gmm = GMM(
            log_w=torch.log(torch.tensor(target_weights.astype(np.float32))),
            mean=torch.tensor(target_means.astype(np.float32)),
            prec=prec,
        )

    def log_density(
        self,
        z: torch.Tensor,
        compute_grad: bool = False,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        return self.target_gmm.log_density(z=z, compute_grad=compute_grad)

    def get_num_dimensions(self):
        return self.target_gmm.mean.shape[-1]

    def can_sample(self):
        return True

    def sample(self, n: int):
        return self.target_gmm.sample(n_samples=n)


def U(theta: float):
    return np.array(
        [
            [math.cos(theta), math.sin(theta)],
            [-math.sin(theta), math.cos(theta)],
        ]
    )


def make_simple_target(n_tasks: int):
    pi = math.pi

    # weights
    w_true = np.array([0.5, 0.3, 0.2])

    # means
    mu_true = np.array(
        [
            [-2.0, -2.0],
            [2.0, -2.0],
            [0.0, 2.0],
        ]
    )

    # covs
    cov1 = np.array([[0.5, 0.0], [0.0, 1.0]])
    cov1 = U(pi / 4) @ cov1 @ np.transpose(U(pi / 4))
    cov2 = np.array([[0.5, 0.0], [0.0, 1.0]])
    cov2 = U(-pi / 4) @ cov2 @ np.transpose(U(-pi / 4))
    cov3 = np.array([[1.0, 0.0], [0.0, 2.0]])
    cov3 = U(pi / 2) @ cov3 @ np.transpose(U(pi / 2))
    cov_true = np.stack([cov1, cov2, cov3], axis=0)

    # stack the parameters n_tasks times
    w_true = np.stack([w_true] * n_tasks, axis=0)
    mu_true = np.stack([mu_true] * n_tasks, axis=0)
    cov_true = np.stack([cov_true] * n_tasks, axis=0)

    # generate tgt dist
    target_dist = GmmLNPDF(
        target_weights=w_true,
        target_means=mu_true,
        target_covars=cov_true,
    )

    return target_dist


def make_star_target(n_tasks: int, n_components: int):
    # Source: Lin et al.

    ## weights
    w_true = np.ones((n_components,)) / n_components

    ## means and precs
    # first component
    mus = [np.array([1.5, 0.0])]
    precs = [np.diag([1.0, 100.0])]
    # other components are generated through rotation
    theta = 2 * math.pi / n_components
    for _ in range(n_components - 1):
        mus.append(U(theta) @ mus[-1])
        precs.append(U(theta) @ precs[-1] @ np.transpose(U(theta)))
    assert len(w_true) == len(mus) == len(precs) == n_components

    mu_true = np.stack(mus, axis=0)
    prec_true = np.stack(precs, axis=0)
    cov_true = np.linalg.inv(prec_true)

    # repeat parameters n_tasks times
    w_true = np.repeat(w_true[None, ...], repeats=n_tasks, axis=0)
    mu_true = np.repeat(mu_true[None, ...], repeats=n_tasks, axis=0)
    cov_true = np.repeat(cov_true[None, ...], repeats=n_tasks, axis=0)

    # check shapes
    assert w_true.shape == (n_tasks, n_components)
    assert mu_true.shape == (n_tasks, n_components, 2)
    assert cov_true.shape == (n_tasks, n_components, 2, 2)

    # generate tgt dist
    target_dist = GmmLNPDF(
        target_weights=w_true,
        target_means=mu_true,
        target_covars=cov_true,
    )

    return target_dist


def make_single_gaussian(n_tasks: int, mean: list, std: list):
    # repeat parameters n_tasks times
    mean = np.array(mean)
    std = np.array(std)
    mean = np.repeat(mean[None, ...], repeats=n_tasks, axis=0)
    mean = mean[:, None, ...]
    covar = np.eye(mean.shape[-1]) * std ** 2
    if mean.shape[-1] == 2:
        theta = np.random.uniform(0, 2 * math.pi)
        covar = U(theta) @ covar @ np.transpose(U(theta))
    covar = np.repeat(covar[None, ...], repeats=n_tasks, axis=0)
    covar = covar[:, None, ...]


    # generate tgt dist
    target_dist = GmmLNPDF(
        target_weights=np.ones((n_tasks, 1)),
        target_means=mean,
        target_covars=covar,
    )

    return target_dist


def get_context(n_contexts, context_dim):
    # n_contexts is n_task

    context_bound_low = -3
    context_bound_high = 3
    contexts = np.random.uniform(context_bound_low, context_bound_high, size=(n_contexts, context_dim))
    return contexts


def make_contextual_star_target(n_tasks: int, n_components: int):
    # 2d contextual star tgt

    ## contexts
    ctx = get_context(n_tasks, 2)

    ## weights
    w_true = np.ones((n_tasks, n_components)) / n_components

    ## means and precs
    # first component
    mus = [np.array([1.5, 0.0])]
    diag1 = np.sin(ctx[:, 0]) + 1.1  # Shape: (batch_size,)
    diag2 = 0.05 * np.cos(ctx[:, 1]) + 0.08
    zeros = np.zeros_like(ctx[:, 0])
    chols = [np.stack([np.stack([diag1, zeros], axis=1),
                     np.stack([zeros, diag2], axis=1)], axis=1)]  # Shape: (batch_size, 2, 2)
    # other components are generated through rotation
    theta = 2 * math.pi / n_components
    for _ in range(n_components - 1):
        mus.append(U(theta) @ mus[-1])
        chols.append(U(theta) @ chols[-1] @ np.transpose(U(theta)))

    mu_true = np.stack(mus, axis=0)
    chols_true = np.stack(chols, axis=1)
    cov_true = chols_true @ np.transpose(chols_true, (0, 1, 3, 2))

    # repeat parameters n_tasks times
    mu_true = np.repeat(mu_true[None, ...], repeats=n_tasks, axis=0)

    # check shapes
    assert w_true.shape == (n_tasks, n_components)
    assert mu_true.shape == (n_tasks, n_components, 2)
    assert cov_true.shape == (n_tasks, n_components, 2, 2)

    # generate tgt dist
    target_dist = GmmLNPDF(
        target_weights=w_true,
        target_means=mu_true,
        target_covars=cov_true,
    )

    return target_dist