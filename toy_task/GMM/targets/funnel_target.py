from typing import Callable


import torch as ch
from torch.distributions import uniform, Normal, MultivariateNormal
import matplotlib.pyplot as plt

from toy_task.GMM.targets.abstract_target import AbstractTarget


class FunnelTarget(AbstractTarget, ch.nn.Module):
    def __init__(self,
                 sig_fn: Callable,
                 context_dim: int,
                 dim: int = 10,
                 context_bound_low: float = -3,
                 context_bound_high: float = 3
                 ):
        """
        Initializes the FunnelTarget with a signal function, context dimension, and sample dimension.

        Parameters:
        - sig_fn (Callable): A function to compute the standard deviation based on context.
        - context_dim (int): The dimensionality of the context.
        - dim (int, optional): The dimensionality of the generated samples. Default is 10.
        """
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.sig = sig_fn
        self.context_dist = uniform.Uniform(context_bound_low, context_bound_high)

    def get_contexts(self, n_contexts: int) -> ch.Tensor:
        """
        Generates a set of contexts for the Funnel distribution.

        Parameters:
        - n_contexts (int): The number of contexts to generate.

        Returns:
        - ch.Tensor: A tensor of shape (n_contexts, context_dim) representing the contexts.
        """
        size = ch.Size([n_contexts, self.context_dim])
        contexts = self.context_dist.sample(size)
        return contexts

    def sample(self, contexts, n_samples):
        """
        Samples from the Funnel distribution given contexts.

        Parameters:
        - contexts (ch.Tensor): A tensor of shape (n_contexts, context_dim) representing the contexts.
        - n_samples (int): The number of samples to draw.

        Returns:
        - ch.Tensor: A tensor of shape (n_samples, n_contexts, dim) representing the samples.
        """
        n_contexts = contexts.shape[0]
        sigs = self.sig(contexts.cpu())

        # Sample the first dimension v from a Normal distribution
        v = Normal(loc=ch.zeros(n_contexts), scale=sigs).sample((n_samples,))
        v = v.unsqueeze(-1)  # Reshape to (S, C, 1)

        # Sample the other dimensions conditioned on v
        other_dim = self.dim - 1
        variance_other = ch.exp(v).expand(-1, -1, other_dim)  # Expand to (S, C, 9)
        cov_other = ch.diag_embed(variance_other)
        mean_other = ch.zeros_like(variance_other)
        samples_other = MultivariateNormal(loc=mean_other, covariance_matrix=cov_other).sample()  # (S, C, 9)

        # Combine the first dimension with the other sampled dimensions
        full_samples = ch.cat((v, samples_other), dim=-1)  # (S, C, 10)
        return full_samples.to(contexts.device)

    def log_prob_tgt(self, contexts, samples):
        """
        Computes the log-probability of samples under the Funnel distribution.

        Parameters:
        - contexts (ch.Tensor): The context vectors of shape (n_contexts, context_dim).
        - samples (ch.Tensor): The samples for which log-probability is computed. It can have shape (n_samples, n_contexts, dim) or (n_samples, n_contexts, 2).

        Returns:
        - ch.Tensor: A tensor representing the log-probabilities for the samples of shape (n_samples, n_contexts).
        """
        n_contexts = contexts.shape[0]
        sigs = self.sig(contexts).to(contexts.device)

        if samples.dim() == 2:
            # For plotting (2D case), expand the samples to the necessary shape
            assert samples.shape[-1] == 2
            samples = samples.unsqueeze(1).expand(-1, n_contexts, -1)
            other_dim = 1
        else:
            other_dim = self.dim - 1

        # Calculate log-probabilities for the first dimension v
        v = samples[:, :, 0]
        log_density_v = Normal(loc=ch.zeros(n_contexts).to(contexts.device), scale=sigs).log_prob(v)

        # Calculate log-probabilities for the other dimensions
        variance_other = ch.exp(v).unsqueeze(-1).expand(-1, -1, other_dim)
        cov_other = ch.diag_embed(variance_other)
        mean_other = ch.zeros_like(variance_other)
        log_density_other = MultivariateNormal(loc=mean_other, covariance_matrix=cov_other).log_prob(
            samples[:, :, 1:])

        # Combine log-probabilities
        log_prob = log_density_v + log_density_other
        return log_prob

    def visualize(self, contexts, n_samples=None):
        """
        Visualizes the Funnel distribution for the given contexts.

        Parameters:
        - contexts (ch.Tensor): The context vectors parameterizing the Funnel.
        - n_samples (int, optional): Number of samples to draw for visualization. Default is None.
        """
        fig, axes = plt.subplots(1, contexts.shape[0], figsize=(5 * contexts.shape[0], 5))
        for i, c in enumerate(contexts):
            # Create a 2D grid for visualization
            v_range = ch.linspace(-2.5 * 3, 2.5 * 3, 100)
            other_range = ch.linspace(-10, 10, 100)
            V, O = ch.meshgrid(v_range, other_range, indexing='ij')
            samples = ch.stack([V, O], dim=-1).view(-1, 2)

            # Compute log-probabilities for the grid points
            log_probs = self.log_prob_tgt(c.unsqueeze(0), samples).squeeze(1).view(100, 100)
            probs = ch.exp(log_probs)

            ax = axes[i]
            # Plot the contour map of the distribution
            ax.contourf(V.numpy(), O.numpy(), probs.numpy(), levels=50, cmap='viridis')

            if n_samples is not None:
                # Sample points and plot them on the contour map
                samples = self.sample(c.unsqueeze(0), n_samples)
                ax.scatter(samples[..., 0], samples[..., 1], color='red', alpha=0.5)

            ax.axis("scaled")
            ax.set_title("Funnel Distribution")
            ax.set_xlabel("$v$")
            ax.set_ylabel("$x[0]$")
            ax.grid(True)
        plt.tight_layout()
        plt.show()


def get_sig_fn(contexts):
    """
    Computes the signal function for standard deviation based on the context.

    Parameters:
    - contexts (ch.Tensor): The context vectors.

    Returns:
    - ch.Tensor: A tensor representing the computed standard deviations.
    """
    if contexts.shape[1] == 1:
        # Context with 1D: Compute standard deviation based on sin function
        sig = ch.sin(3 * contexts[:, 0] + 1) + 1.1
    else:
        # Context with 2D: Compute standard deviation based on product of context elements
        sig = ch.sin(3 * contexts[:, 0] * contexts[:, 1] + 1) + 1.1
    return sig
