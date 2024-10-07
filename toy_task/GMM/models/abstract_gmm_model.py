from abc import ABC, abstractmethod
import torch as ch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily


class AbstractGMM(ABC):
    """
    Abstract class representing a Gaussian Mixture Model.
    """

    @abstractmethod
    def forward(self, x: ch.Tensor) -> (ch.Tensor, ch.Tensor, ch.Tensor):
        """
        Forward pass to compute parameters of the Gaussian components based on input.

        Args:
            x (torch.Tensor): Input context tensor.

        Returns:
            (ch.Tensor, ch.Tensor, ch.Tensor): component weights, means, and cholesky matrices.
        """
        pass

    def covariance(self, chol: ch.Tensor) -> ch.Tensor:
        """
        Compute covariance matrices from Cholesky decompositions.

        Args:
            chol (torch.Tensor): Cholesky matrices of the GMM component with shape (n_contexts, ..., dz, dz)

        Returns:
            torch.Tensor: Covariance matrices with the same shape
        """

        cov_matrix = chol @ chol.transpose(-1, -2)

        return cov_matrix

    def get_rsamples(self,
                     means: ch.Tensor,
                     chols: ch.Tensor,
                     n_samples: int
                     ) -> ch.Tensor:
        """
        Generate reparameterized samples from the Gaussian distributions.

        Args:
            means (torch.Tensor): Means of the GMM components with shape (n_contexts, n_components, dz)
            chols (torch.Tensor): Cholesky matrices of the GMM components with shape (n_contexts, n_components, dz, dz)
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Reparameterized samples with shape (n_samples, n_contexts, n_components, dz)
        """

        rsamples = MultivariateNormal(loc=means, scale_tril=chols).rsample(ch.Size([n_samples]))

        return rsamples

    def get_samples_gmm(self,
                        log_gates: ch.Tensor,
                        means: ch.Tensor,
                        chols: ch.Tensor,
                        n_samples: int
                        ) -> ch.Tensor:
        """
        Generate samples from the Gaussian Mixture Model.

        Args:
            log_gates (torch.Tensor): Logarithm of the mixing weights with shape (n_contexts, n_components)
            means (torch.Tensor): Means of the GMM components with shape (n_contexts, n_components, dz)
            chols (torch.Tensor): Cholesky matrices of the GMM component with shape (n_contexts, n_components, dz, dz)
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Samples generated from the GMM with shape (n_samples, n_contexts, dz)
        """

        if log_gates.shape[1] == 1:
            samples = MultivariateNormal(means.squeeze(1), scale_tril=chols.squeeze(1)).sample(ch.Size([n_samples]))
        else:
            samples = MixtureSameFamily(
                mixture_distribution=Categorical(logits=log_gates),
                component_distribution=MultivariateNormal(loc=means, scale_tril=chols
                ),
            ).sample(ch.Size([n_samples]))

        return samples

    def log_prob(self,
                 means: ch.Tensor,
                 chols: ch.Tensor,
                 samples: ch.Tensor
                 ) -> ch.Tensor:
        """
        Calculate the log probabilities of given samples under the Gaussian distributions.

        Args:
            means (torch.Tensor): Mean of the GMM components with shape (n_contexts, n_components, dz)
            chols (torch.Tensor): Cholesky matrix of the GMM components with shape (n_contexts, n_components, dz, dz)
            samples (torch.Tensor):  shape (n_samples, n_contexts, n_components_samples, dz)

        Returns:
            torch.Tensor : Log probabilities of the samples with shape (n_samples, n_contexts, n_components)
        """

        log_probs = MultivariateNormal(loc=means, scale_tril=chols).log_prob(samples)

        return log_probs

    def log_prob_gmm(self,
                     means: ch.Tensor,
                     chols: ch.Tensor,
                     log_gates: ch.Tensor,
                     samples: ch.Tensor
                     ) -> ch.Tensor:
        """
        Calculate the log probabilities of given samples under the Gaussian Mixture Model.

        Args:
            log_gates (torch.Tensor): Logarithm of the mixing weights with shape (n_contexts, n_components)
            means (torch.Tensor): Means of the GMM components with shape (n_contexts, n_components, dz)
            chols (torch.Tensor): Cholesky matrices of the GMM component with shape (n_contexts, n_components, dz, dz)
            samples (torch.Tensor):  shape (n_samples, n_contexts, dz)

        Returns:
            torch.Tensor: Log probabilities of the samples under the GMM with shape (n_samples, n_contexts)
        """

        # adaption for plot2d_matplotlib
        if means.shape[-1] != samples.shape[-1]:
            assert samples.shape[-1] == 2
            means = means[..., :2]
            chols = chols[..., :2, :2]

        log_probs_gmm = MixtureSameFamily(
            mixture_distribution=Categorical(logits=log_gates),
            component_distribution=MultivariateNormal(loc=means, scale_tril=chols
            ),
        ).log_prob(samples)

        return log_probs_gmm

    def log_responsibilities_gmm(self,
                                 means: ch.Tensor,
                                 chols: ch.Tensor,
                                 log_gates: ch.Tensor,
                                 samples: ch.Tensor
                                 ) :
        """
        Compute log responsibilities of the GMM, that is log p(o|x) = log p(x|o) + log p(o) - log p(x)

        Args:
            log_gates (torch.Tensor): Logarithm of the mixing weights with shape (n_contexts, n_components)
            means (torch.Tensor): Means of the GMM components with shape (n_contexts, n_components, dz)
            chols (torch.Tensor): Cholesky matrices of the GMM component with shape (n_contexts, n_components, dz, dz)
            samples (torch.Tensor):  shape (n_samples, n_contexts, n_components, dz)
        Returns:
            torch.Tensor: Log responsibilities of the GMM with shape (n_samples, n_contexts, n_components)
        """

        log_probs = self.log_prob(means, chols, samples) # (n_samples, n_contexts, n_components)

        # important: to calculate the responsibilities, we need log samples of all components on all components.
        # i.e. the correct shape is (n_samples, n_contexts, n_components_samples, n_components),
        # actually n_components_samples is always equal to n_components, but they have different meaning
        _, _, n_components_samples, _ = samples.shape

        means_expand = means.unsqueeze(1).expand(-1, n_components_samples, -1, -1)
        chols_expand = chols.unsqueeze(1).expand(-1, n_components_samples, -1, -1, -1)
        log_probs_expand = MultivariateNormal(loc=means_expand, scale_tril=chols_expand).log_prob(samples.unsqueeze(3))
        log_model = ch.logsumexp(log_gates.unsqueeze(0).unsqueeze(-2) + log_probs_expand, dim=-1)
        log_responsibilities = log_gates.unsqueeze(0) + log_probs - log_model

        return log_responsibilities
