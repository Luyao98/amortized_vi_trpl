from typing import Callable


import torch as ch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
import math

from toy_task.GMM.targets.gaussian_mixture_target import ConditionalGMMTarget


class ConditionalSTARTarget(ConditionalGMMTarget):
    """
    A conditional Gaussian Mixture Model target distribution.
    """
    def __init__(self,
                 gate_fn: Callable,
                 mean_fn: Callable,
                 chol_fn: Callable,
                 context_dim: int,
                 context_bound_low: float = -3,
                 context_bound_high: float = 3
                 ):
        """
        Initializes the ConditionalSTARTarget with functions defining the GMM's components.

        Parameters:
        - gate_fn (Callable): A function to generate the mixture weights given a context.
        - mean_fn (Callable): A function to generate the means of the Gaussian components given a context.
        - chol_fn (Callable): A function to generate the Cholesky decomposition of the covariance matrices given a context.
        - context_dim (int): The dimensionality of the context.
        - context_bound_low (float, optional): Lower bound for the uniform context distribution.
        - context_bound_high (float, optional): Upper bound for the uniform context distribution.
        """
        super().__init__(gate_fn, mean_fn, chol_fn, context_dim, context_bound_low, context_bound_high)

    def sample(self,
               contexts: ch.Tensor,
               n_samples: int
               ) -> ch.Tensor:
        """
        Samples from the conditional GMM given contexts.

        Parameters:
        - contexts (ch.Tensor): The context vectors parameterizing the GMM with shape (n_contexts, context_dim).
        - n_samples (int): The number of samples to draw.

        Returns:
        - ch.Tensor: shape (n_samples, n_contexts, d_z).
        """

        log_gates = self.gate_fn(contexts)
        means = self.mean_fn(contexts)
        chols = self.chol_fn(contexts)
        covs = chols @ chols.transpose(-1, -2)

        samples = MixtureSameFamily(
            mixture_distribution=Categorical(logits=log_gates),
            component_distribution=MultivariateNormal(loc=means, covariance_matrix=covs
            ),
        ).sample(ch.Size([n_samples,]))
        return samples

    def log_prob_tgt(self,
                     contexts: ch.Tensor,
                     samples: ch.Tensor
                     ) -> ch.Tensor:
        """
        Calculates the log-probability of samples under the conditional GMM.

        Parameters:
        - contexts (ch.Tensor): The context vectors parameterizing the GMM with shape (n_contexts, context_dim).
        - samples (ch.Tensor):  shape (n_samples, n_contexts, d_z) or (n_samples, n_contexts, n_components, d_z)

        Returns:
        - ch.Tensor: Target log densities with shape (n_samples, n_contexts) or (n_samples, n_contexts, n_components).
        """

        log_gates = self.gate_fn(contexts) # (n_contexts, n_components_tgt)
        means = self.mean_fn(contexts)  # (n_contexts, n_components_tgt, d_z)
        chols = self.chol_fn(contexts)  # (n_contexts, n_components_tgt, d_z, d_z)
        covs = chols @ chols.transpose(-1, -2)  # (n_contexts, n_components_tgt, d_z, d_z)

        tgt = MixtureSameFamily(
            mixture_distribution=Categorical(logits=log_gates),
            component_distribution=MultivariateNormal(loc=means, covariance_matrix=covs)
        )
        if samples.dim() == 3:
            log_probs = tgt.log_prob(samples)
        else:
            n_samples, n_contexts, n_components, d_z = samples.shape
            reshaped_samples = samples.permute(0, 2, 1, 3).reshape(-1, n_contexts, d_z)  # (n_samples * n_components, n_contexts, d_z)
            reshaped_log_probs = tgt.log_prob(reshaped_samples)  # (n_samples * n_components, n_contexts)
            log_probs = reshaped_log_probs.view(n_samples, n_components, n_contexts).permute(0, 2, 1)

        return log_probs


def get_weights_fn(n_components):
    def get_weights(contexts):
        batch_size = contexts.shape[0]
        weights = ch.ones((batch_size,n_components)) / n_components
        return ch.log(weights)
    return get_weights

def U(theta: float):
    return ch.tensor(
        [
            [math.cos(theta), math.sin(theta)],
            [-math.sin(theta), math.cos(theta)],
        ]
    )


def get_chol_fn(n_components):
    def cat_chol(c):
        diag1 = ch.sin(c[:, 0]) + 1.1
        zeros = ch.zeros_like(c[:, 0])
        if c.shape[-1] == 1:
            diag3 = 0.05 * ch.cos(c[:, 0]) + 0.08
            chol = ch.stack([ch.stack([diag1, zeros], dim=1),
                             ch.stack([zeros, diag3], dim=1)], dim=1)
        elif c.shape[-1] == 2:
            diag2 = 0.05 * ch.cos(c[:, 1]) + 0.08
            diag4 = 0.05 * ch.sin(c[:, 0]) * ch.cos(c[:, 1])
            chol = ch.stack([ch.stack([diag1, zeros], dim=1),
                             ch.stack([diag4, diag2], dim=1)], dim=1)
        else:
            raise ValueError("Context dimension must be 1 or 2")
        chols = [chol]
        theta = 2 * math.pi / n_components
        rotation = U(theta).to(chol.device)
        for _ in range(n_components - 1):
            chols.append(rotation @ chols[-1] @ rotation.transpose(0, 1))
        return ch.stack(chols, dim=1)
    return cat_chol


def get_mean_fn(n_components):
    def generate_star_means(contexts):
        batch_size = contexts.shape[0]
        # First component mean
        mus = [ch.tensor([1.5, 0.0], device=contexts.device)]
        # Other components generated through rotation
        theta = 2 * math.pi / n_components
        rotation = U(theta).to(contexts.device)
        for _ in range(n_components - 1):
            mus.append(rotation @ mus[-1])
        mu_true = ch.stack(mus, dim=0)
        return mu_true.repeat(batch_size, 1, 1)
    return generate_star_means


def get_star_target(n_components, context_dim):
    gate_target = get_weights_fn(n_components)
    mean_target = get_mean_fn(n_components)
    chol_target = get_chol_fn(n_components)
    star_target = ConditionalSTARTarget(gate_target, mean_target, chol_target, context_dim)
    return star_target
