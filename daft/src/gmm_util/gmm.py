from typing import Tuple, Union

import torch

from daft.src.gmm_util.util import (
    prec_to_prec_chol,
    prec_to_cov_chol,
    cov_chol_to_cov,
    sample_gmm,
    gmm_log_density_grad,
    gmm_log_component_densities,
    gmm_log_density,
    gmm_log_responsibilities,
)


class GMM:
    def __init__(
        self, log_w: torch.Tensor, mean: torch.Tensor, prec: torch.Tensor, device: str or None = None
    ):
        # check input
        self.n_components = log_w.shape[-1]
        self.d_z = mean.shape[-1]
        assert log_w.shape[-1:] == (self.n_components,)
        assert mean.shape[-2:] == (self.n_components, self.d_z)
        assert prec.shape[-3:] == (self.n_components, self.d_z, self.d_z)

        self._device = device

        self._log_w = log_w.to(device)
        self._mean = mean.to(device)
        self._prec = prec.to(device)
        self._prec_chol = prec_to_prec_chol(prec=self._prec)
        self._cov_chol = prec_to_cov_chol(prec=self._prec)
        self._cov = cov_chol_to_cov(self._cov_chol)

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def log_w(self) -> torch.Tensor:
        return self._log_w

    @property
    def prec(self) -> torch.Tensor:
        return self._prec

    @property
    def prec_chol(self) -> torch.Tensor:
        return self._prec_chol

    @property
    def cov_chol(self) -> torch.Tensor:
        return self._cov_chol

    @property
    def cov(self) -> torch.Tensor:
        return self._cov

    @mean.setter
    def mean(self, value: torch.Tensor) -> None:
        """
        :param value: shape ([batch_dims], n_components, d_z)
        :return: None
        """
        assert value.shape[-2:] == (self.n_components, self.d_z)
        self._mean = value

    @log_w.setter
    def log_w(self, value):
        """
        :param value: shape ([batch_dims], n_components)
        :return: None
        """
        assert value.shape[-1:] == (self.n_components,)
        self._log_w = value

    @prec.setter
    def prec(self, value):
        """
        Updates all variance related parameters
        :param value: shape ([batch_dims], n_components, d_z, d_z)
        :return: None
        """
        self._prec = value
        self._prec_chol = prec_to_prec_chol(prec=self._prec)
        self._cov_chol = prec_to_cov_chol(prec=self._prec)
        self._cov = cov_chol_to_cov(self._cov_chol)

    @prec_chol.setter
    def prec_chol(self, value: torch.Tensor) -> None:
        raise NotImplementedError

    @cov_chol.setter
    def cov_chol(self, value: torch.Tensor) -> None:
        raise NotImplementedError

    @cov.setter
    def cov(self, value: torch.Tensor) -> None:
        raise NotImplementedError

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Samples from the whole GMM (all components) based on the log_w, means and precs
        :param n_samples: int
        :return: tensor of shape (n_samples, [batch_dims], d_z)
        """
        return sample_gmm(
            n_samples=n_samples, log_w=self.log_w, mean=self.mean, cov_chol=self.cov_chol
        )

    def log_density(
        self,
        z: torch.Tensor,
        compute_grad: bool = False,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        return gmm_log_density_grad(
            z=z,
            log_w=self.log_w,
            mean=self.mean,
            prec=self.prec,
            prec_chol=self.prec_chol,
            compute_grad=compute_grad,
        )

    def log_component_densities(self, z: torch.Tensor) -> torch.Tensor:
        return gmm_log_component_densities(z=z, mean=self.mean, prec_chol=self.prec_chol)

    def log_responsibilities(self, z: torch.Tensor) -> torch.Tensor:
        log_component_densities = gmm_log_component_densities(
            z=z,
            mean=self.mean,
            prec_chol=self.prec_chol,
        )
        log_density = gmm_log_density(
            log_w=self.log_w,
            log_component_densities=log_component_densities,
        )
        return gmm_log_responsibilities(
            log_w=self.log_w,
            log_component_densities=log_component_densities,
            log_density=log_density,
        )

    def sample_all_components(self, num_samples_per_component):
        """
        Draws num_samples_per_component from each Gaussian component of every GMM in this model.
        :param num_samples_per_component: int
        :return: tensor of shape (num_samples_per_component, [batch_dims], n_components, d_z)
        """
        # TODO: is this efficient?
        samples = torch.distributions.MultivariateNormal(
            loc=self.mean, scale_tril=self.cov_chol, validate_args=False
        ).sample((num_samples_per_component,))
        return samples

    def get_params_dict(self):
        """
        Returns a dictionary containing the full state of the GMM. Useful for saving and loading
        """
        return {
            "log_w": self.log_w,
            "mean": self.mean,
            "prec": self.prec,
        }


if __name__ == "__main__":
    # create a batch of 2 GMMs with dimension 1 and 3 components
    log_w = torch.tensor([[-3.0, -1.5, -1.0], [-1.0, -2.0, -1.0]])
    mean = torch.tensor([[[-10.0], [1.0], [4.0]], [[-3.0], [1.0], [4.0]]])
    prec = torch.tensor([[[[1.0]], [[1.0]], [[1.0]]], [[[1.0]], [[1.0]], [[1.0]]]])
    # print shapes
    print(log_w.shape)
    print(mean.shape)
    print(prec.shape)
    # create GMM object
    gmm = GMM(log_w=log_w, mean=mean, prec=prec)
    # sample from GMM
    samples = gmm.sample(n_samples=10000)
    print(samples.shape)
    # plot samples from first GMM
    import matplotlib.pyplot as plt

    plt.hist(samples[:, 0, :].numpy(), bins=200)
    plt.show()
