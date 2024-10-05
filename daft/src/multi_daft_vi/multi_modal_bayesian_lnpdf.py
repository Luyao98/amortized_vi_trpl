from typing import Union, Tuple

import numpy as np
import torch

from daft.src.gmm_util.gmm import GMM
from daft.src.multi_daft_vi.lnpdf import LNPDF, U
from daft.src.multi_daft_vi.util_multi_daft import create_initial_gmm_parameters


class MultiModalBayesianLNPDF(LNPDF):
    """
    Dummy Bayesian likelihood for testing purposes: 2D GMM for the likelihood and 2D GMM for the prior
    """

    def __init__(
        self, num_comps=20, num_prior_comps=4, n_tasks=1, likelihood_scale=1.0, total_scale=1.0
    ):
        self.num_comps = num_comps
        self.d_z = 2
        self.num_prior_comps = num_prior_comps
        self.n_tasks = n_tasks
        self.likelihood_scale = likelihood_scale
        self.total_scale = total_scale

        mean = torch.tensor(
            np.random.uniform(-5, 5, size=(self.n_tasks, self.num_comps, self.d_z)).astype(
                np.float32
            )
        )

        cov = []
        for i in range(self.num_comps):
            cov_comp = np.eye(2) * np.random.uniform(0.1, 5.0, size=2)
            angle = np.random.uniform(0, 2 * np.pi)
            cov_comp = U(angle) @ cov_comp @ np.transpose(U(angle))
            cov.append(cov_comp)
        cov_true = np.stack(cov, axis=0)
        cov_true = np.stack([cov_true] * self.n_tasks, axis=0)
        prec = torch.linalg.inv(torch.tensor(cov_true.astype(np.float32)))
        log_w = torch.log(
            torch.tensor(
                np.ones((self.n_tasks, self.num_comps)).astype(np.float32) / self.num_comps
            )
        )
        self.likelihood_gmm = GMM(
            log_w=log_w,
            mean=mean,
            prec=prec,
        )

        prior_log_w, prior_mean, prior_prec = create_initial_gmm_parameters(
            d_z=2,
            n_tasks=self.n_tasks,
            n_components=self.num_prior_comps,
            prior_scale=1.0,
            initial_var=1.0,
        )

        self.prior_gmm = GMM(prior_log_w, prior_mean, prior_prec)

    def log_density(
        self,
        z: torch.Tensor,
        compute_grad: bool = False,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        likelihood, likelihood_grad = self.likelihood_gmm.log_density(
            z=z, compute_grad=compute_grad
        )
        prior, prior_grad = self.prior_gmm.log_density(z=z, compute_grad=compute_grad)
        result = self.total_scale * (self.likelihood_scale * likelihood + prior)
        if compute_grad:
            return result, self.total_scale * (self.likelihood_scale * likelihood_grad + prior_grad)
        else:
            return result, None

    def get_num_dimensions(self):
        return self.likelihood_gmm.mean.shape[-1]

    def can_sample(self):
        return True

    def sample(self, n: int):
        return False
