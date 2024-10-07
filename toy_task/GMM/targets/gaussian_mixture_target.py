from typing import Callable
import torch as ch
from torch.distributions import uniform, MultivariateNormal, Categorical, MixtureSameFamily
import numpy as np
import matplotlib.pyplot as plt

from toy_task.GMM.targets.abstract_target import AbstractTarget


class ConditionalGMMTarget(AbstractTarget, ch.nn.Module):
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
        Initializes the ConditionalGMMTarget with functions defining the GMM's components.

        Parameters:
        - gate_fn (Callable): A function to generate the mixture weights given a context.
        - mean_fn (Callable): A function to generate the means of the Gaussian components given a context.
        - chol_fn (Callable): A function to generate the Cholesky decomposition of the covariance matrices given a context.
        - context_dim (int): The dimensionality of the context.
        - context_bound_low (float, optional): Lower bound for the uniform context distribution.
        - context_bound_high (float, optional): Upper bound for the uniform context distribution.
        """
        super().__init__()
        self.context_dim = context_dim
        self.context_dist = uniform.Uniform(context_bound_low, context_bound_high)
        self.gate_fn = gate_fn
        self.mean_fn = mean_fn
        self.chol_fn = chol_fn

    def get_contexts(self,
                     n_contexts: int
                     ) -> ch.Tensor:
        """
        Generates a set of contexts for the conditional GMM.

        Parameters:
        - n_contexts (int): Number of contexts to generate.

        Returns:
        - ch.Tensor: shape (n_contexts, context_dim).
        """
        size = ch.Size([n_contexts, self.context_dim])
        contexts = self.context_dist.sample(size)
        return contexts

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
        device = contexts.device
        log_gates = self.gate_fn(contexts).to(device)
        means = self.mean_fn(contexts).to(device)
        chols = self.chol_fn(contexts).to(device)

        samples = MixtureSameFamily(
            mixture_distribution=Categorical(logits=log_gates),
            component_distribution=MultivariateNormal(loc=means, scale_tril=chols
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
        device = contexts.device
        log_gates = self.gate_fn(contexts).to(device) # (n_contexts, n_components_tgt)
        means = self.mean_fn(contexts).to(device) # (n_contexts, n_components_tgt, d_z)
        chols = self.chol_fn(contexts).to(device) # (n_contexts, n_components_tgt, d_z, d_z)

        tgt = MixtureSameFamily(
            mixture_distribution=Categorical(logits=log_gates),
            component_distribution=MultivariateNormal(loc=means, scale_tril=chols)
        )
        if samples.dim() == 3:
            log_probs = tgt.log_prob(samples)
        else:
            n_samples, n_contexts, n_components, d_z = samples.shape
            reshaped_samples = samples.permute(0, 2, 1, 3).reshape(-1, n_contexts, d_z)  # (n_samples * n_components, n_contexts, d_z)
            reshaped_log_probs = tgt.log_prob(reshaped_samples)  # (n_samples * n_components, n_contexts)
            log_probs = reshaped_log_probs.view(n_samples, n_components, n_contexts).permute(0, 2, 1)

        return log_probs

    def visualize(self,
                  contexts: ch.Tensor,
                  n_samples: int = None
                  ):
        """
        Visualizes the target distribution given the contexts.

        Parameters:
        - contexts (ch.Tensor): The context vectors parameterizing the GMM.
        - n_samples (int, optional): Number of samples to draw for visualization. Default is None.
        """
        fig, axes = plt.subplots(1, contexts.shape[0], figsize=(5 * contexts.shape[0], 5))
        for i, c in enumerate(contexts):
            x, y = np.meshgrid(np.linspace(-30, 30, 300), np.linspace(-30, 30, 300))
            grid = ch.tensor(np.c_[x.ravel(), y.ravel()], dtype=ch.float32)
            pdf_values = self.log_prob_tgt(c.unsqueeze(0), grid)
            pdf_values = pdf_values.exp().view(300, 300).numpy()

            ax = axes[i]
            ax.contourf(x, y, pdf_values, levels=50, cmap='viridis')
            if n_samples is not None:
                samples = self.sample(c.unsqueeze(0), n_samples)
                ax.scatter(samples[..., 0], samples[..., 1], color='red', alpha=0.5)
            ax.set_title(f'Target {i + 1} with context {c}')

        plt.tight_layout()
        plt.show()


def spiral(
        t: ch.Tensor,
        contexts: ch.Tensor,
        a: float = 0.3
) -> ch.Tensor:
    """
    Generates a spiral pattern based on input contexts.

    Parameters:
    - t (ch.Tensor): A tensor representing the t-values used to generate the spiral.
    - contexts (ch.Tensor): Context vectors used to parameterize the spiral.
    - a (float, optional): Scaling factor for the spiral. Default is 0.3.

    Returns:
    - ch.Tensor: A tensor containing the generated spiral coordinates with shape (n_contexts, t_dim, 2).
    """
    if contexts.shape[-1] == 1:
        b = t + 0.1 * contexts
    elif contexts.shape[-1] == 2:
        b = t + 0.1 * contexts[:, 0].unsqueeze(1)
    else:
        raise ValueError('Context dimension must be 1 or 2')
    x = a * t * ch.cos(b)
    y = a * t * ch.sin(b)
    return ch.stack([x, y], dim=-1)


def get_mean_fn(
        n_components: int
) -> Callable:
    """
    Generates a function to compute the means of the Gaussian components of the GMM.

    Parameters:
    - n_components (int): Number of Gaussian components in the mixture.

    Returns:
    - Callable: A function that takes contexts as input and returns the component means.
    """
    def generate_spiral_means(
            contexts: ch.Tensor
    ) -> ch.Tensor:
        """
        Computes the means of the Gaussian components based on spiral generation.

        Parameters:
        - contexts (ch.Tensor): The context vectors used to parameterize the means.

        Returns:
        - ch.Tensor: shape (n_contexts, n_components, 2).
        """
        t_values = np.linspace(0, 14 * np.pi, n_components, endpoint=False)
        means = spiral(ch.tensor(t_values, dtype=ch.float32, device=contexts.device), contexts)
        return means
    return generate_spiral_means


def get_weights_fn(
        n_components: int
) -> Callable:
    """
    Generates a function to compute the mixture weights of the Gaussian components.

    Parameters:
    - n_components (int): Number of Gaussian components in the mixture.

    Returns:
    - Callable: A function that takes contexts as input and returns the log-softmax weights.
    """
    def get_weights(
            contexts: ch.Tensor
    ) -> ch.Tensor:
        """
        Computes the mixture weights of the Gaussian components based on the input contexts.

        Parameters:
        - contexts (ch.Tensor): The context vectors used to parameterize the weights.

        Returns:
        - ch.Tensor: A tensor containing the log-softmax weights with shape (n_contexts, n_components).
        """
        weights = []
        if contexts.shape[-1] == 1:
            for i in range(n_components):
                if i % 2 == 0:
                    weights.append(ch.sin((i + 1) * contexts[:, 0]))
                else:
                    weights.append(ch.cos((i + 1) * contexts[:, 0]))
        elif contexts.shape[-1] == 2:
            for i in range(n_components):
                if i % 2 == 0:
                    weights.append(ch.sin((i + 1) * contexts[:, 0]))
                else:
                    weights.append(ch.cos((i + 1) * contexts[:, 1]))
        else:
            raise ValueError('Context dimension must be 1 or 2')
        weights = ch.stack(weights, dim=1)
        log_weights = ch.log_softmax(weights, dim=1)
        return log_weights
    return get_weights


def get_chol_fn(
        n_components: int
) -> Callable:
    """
    Generates a function to compute the Cholesky decomposition of the covariance matrices of the Gaussian components.

    Parameters:
    - n_components (int): Number of Gaussian components in the mixture.

    Returns:
    - Callable: A function that takes contexts as input and returns the Cholesky decompositions.
    """
    def cat_chol(
            contexts: ch.Tensor
    ) -> ch.Tensor:
        """
        Computes the Cholesky decompositions of the covariance matrices based on the input contexts.

        Parameters:
        - contexts (ch.Tensor): The context vectors used to parameterize the Cholesky decompositions.

        Returns:
        - ch.Tensor: A tensor containing the Cholesky decompositions with shape (n_contexts, n_components, 2, 2).
        """
        chols = []
        element0 = ch.zeros_like(contexts[:, 0])
        element1 = ch.sin(3 * contexts[:, 0]) * ch.cos(3 * contexts[:, 0])
        element2 = 0.3 * ch.sin(contexts[:, 0]) * ch.cos(contexts[:, 1])
        if contexts.shape[-1] == 1:
            for i in range(n_components):
                chol = ch.stack([
                    ch.stack([0.5 * ch.sin((i + 1) * contexts[:, 0]) + 0.8, element0], dim=1),
                    ch.stack([element1, 0.5 * ch.cos((i + 1) * contexts[:, 0]) + 0.8], dim=1)], dim=1)
                chols.append(chol)
        elif contexts.shape[-1] == 2:
            for i in range(n_components):
                chol = ch.stack([
                    ch.stack([0.3 * ch.sin((i + 1) * contexts[:, 0]) + 0.5, element0], dim=1),
                    ch.stack([element2, 0.3 * ch.cos((i + 1) * contexts[:, 1]) + 0.5], dim=1)], dim=1)
                chols.append(chol)
        return ch.stack(chols, dim=1)
    return cat_chol


def get_gmm_target(
        n_components: int,
        context_dim: int
) -> ConditionalGMMTarget:
    """
    Creates a ConditionalGMMTarget instance with the specified number of components and context dimension.

    Parameters:
    - n_components (int): Number of Gaussian components in the mixture.
    - context_dim (int): The dimensionality of the context.

    Returns:
    - ConditionalGMMTarget: An instance of ConditionalGMMTarget with specified parameters.
    """
    gate_target = get_weights_fn(n_components)
    mean_target = get_mean_fn(n_components)
    chol_target = get_chol_fn(n_components)
    gmm_target = ConditionalGMMTarget(gate_target, mean_target, chol_target, context_dim)
    return gmm_target
