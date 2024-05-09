from abc import ABC, abstractmethod
import torch as ch


class AbstractGMM(ABC):
    """
    Abstract class representing a Gaussian Mixture Model.
    """

    @abstractmethod
    def forward(self, x: ch.Tensor) -> tuple:
        """
        Forward pass to compute parameters of the Gaussian components based on input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Tuple containing component weights, means, and covariance matrices.
        """
        pass

    @abstractmethod
    def covariance(self, chol: ch.Tensor) -> ch.Tensor:
        """
        Compute covariance matrices from Cholesky decompositions.

        Args:
            chol (torch.Tensor): Cholesky decompositions of covariance matrices.

        Returns:
            torch.Tensor: Covariance matrices.
        """
        pass

    @abstractmethod
    def get_rsamples(self, mean: ch.Tensor, chol: ch.Tensor, n_samples: int) -> ch.Tensor:
        """
        Generate reparameterized samples from the Gaussian distributions.

        Args:
            mean (torch.Tensor): Means of the Gaussian components.
            chol (torch.Tensor): Cholesky decompositions of the covariance matrices.
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Reparameterized samples.
        """
        pass

    @abstractmethod
    def get_samples_gmm(self, log_gates: ch.Tensor, means: ch.Tensor, chols: ch.Tensor, num_samples: int) -> ch.Tensor:
        """
        Generate samples from the Gaussian Mixture Model.

        Args:
            log_gates (torch.Tensor): Logarithm of the mixing weights for the components.
            means (torch.Tensor): Means of the Gaussian components.
            chols (torch.Tensor): Cholesky decompositions of the covariance matrices of the components.
            num_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Samples generated from the GMM.
        """
        pass

    @abstractmethod
    def log_prob(self, mean: ch.Tensor, chol: ch.Tensor, samples: ch.Tensor) -> ch.Tensor:
        """
        Calculate the log probabilities of given samples under the Gaussian distributions.

        Args:
            mean (torch.Tensor): Means of the Gaussian components.
            chol (torch.Tensor): Cholesky decompositions of the covariance matrices.
            samples (torch.Tensor): Samples for which to compute the log probabilities.

        Returns:
            torch.Tensor: Log probabilities of the samples.
        """
        pass

    @abstractmethod
    def log_prob_gmm(self, means: ch.Tensor, chols: ch.Tensor, log_gates: ch.Tensor, samples: ch.Tensor) -> ch.Tensor:
        """
        Calculate the log probabilities of given samples under the Gaussian Mixture Model.

        Args:
            means (torch.Tensor): Means of the Gaussian components.
            chols (torch.Tensor): Cholesky decompositions of the covariance matrices of the components.
            log_gates (torch.Tensor): Logarithm of the mixing weights.
            samples (torch.Tensor): Samples for which to compute the log probabilities.

        Returns:
            torch.Tensor: Log probabilities of the samples under the GMM.
        """
        pass

    @abstractmethod
    def auxiliary_reward(self, j: int, gate_old: ch.Tensor, mean_old: ch.Tensor, chol_old: ch.Tensor,
                         samples: ch.Tensor) -> ch.Tensor:
        """
        Compute the auxiliary reward for reinforcement learning applications.

        Args:
            j (int): Index of the component.
            gate_old (torch.Tensor): Previous mixing weights.
            mean_old (torch.Tensor): Previous means of the Gaussian components.
            chol_old (torch.Tensor): Previous Cholesky decompositions of the covariance matrices.
            samples (torch.Tensor): Samples for which to compute the auxiliary reward.

        Returns:
            torch.Tensor: Computed auxiliary rewards.
        """
        pass
