from typing import Callable


import torch as ch
from torch.distributions import uniform, MultivariateNormal, Categorical
import numpy as np
import matplotlib.pyplot as plt

from toy_task.GMM.targets.abstract_target import AbstractTarget


def get_curvature_fn(
        n_components: int
) -> Callable:
    def get_curvatures(
        contexts: ch.Tensor
) -> ch.Tensor:
        # contexts = contexts.expand(-1, 2)
        if n_components == 1:
            curvature = ch.stack([contexts])
        elif n_components == 2:
            curvature = ch.stack([contexts, 0.5 * contexts + 1])
        elif n_components == 5:
            curvature = ch.stack([0.2 * contexts + 1.5,
                                  -0.2 * contexts - 1.5,
                                  contexts,
                                  0.5 * contexts + 1,
                                  -0.5 * contexts - 1])
        else:
            raise ValueError("BMM target now only supports 1, 2 or 5 components")
        return curvature
    return get_curvatures


def get_weights(contexts, n_components):
    """
    Generates weight values for a given number of components.
    The number of components is flexible.
    """
    weights = []
    for i in range(n_components):
        if i % 2 == 0:
            weights.append(ch.sin((i + 1) * contexts[:, 0]))
        else:
            weights.append(ch.cos((i + 1) * contexts[:, 0]))
    weights = ch.stack(weights, dim=1)
    log_weights = ch.log_softmax(weights, dim=1)
    return log_weights


class BananaMixtureTarget(AbstractTarget, ch.nn.Module):
    """
    A conditional Banana Mixture Model target distribution.
    """
    def __init__(self,
                 curvature_fn: Callable,
                 context_dim: int,
                 context_bound_low: float = -3,
                 context_bound_high: float = 3
                 ):
        """
        Initializes the ConditionalGMMTarget with functions defining the BMM's components.

        Parameters:
        - curvature_fn (Callable): A function that generates the curvatures for the BMM.
        - context_dim (int): The dimensionality of the context.
        - context_bound_low (float, optional): Lower bound for the uniform context distribution.
        - context_bound_high (float, optional): Upper bound for the uniform context distribution.
        """
        super().__init__()
        self.context_dim = context_dim
        self.context_dist = uniform.Uniform(context_bound_low, context_bound_high)
        self.curvature_fn = curvature_fn

    def get_contexts(self,
                     n_contexts: int
                     ) -> ch.Tensor:
        """
        Generates a set of contexts for the conditional BMM.

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
        Samples from the conditional BMM given contexts.

        Parameters:
        - contexts (ch.Tensor): The context vectors parameterizing the GMM with shape (n_contexts, context_dim).
        - n_samples (int): The number of samples to draw.

        Returns:
        - ch.Tensor: shape (n_samples, n_contexts, d_z).
        """
        curvatures = self.curvature_fn(contexts) # (O, B, 1)
        n_components, batch_size, _ = curvatures.shape
        device = contexts.device

        log_gates = get_weights(contexts, n_components) # (B, O)
        means = ch.zeros(2, dtype=ch.float32, device=device)
        covs = 0.4 * ch.eye(2, dtype=ch.float32, device=device)

        indices = Categorical(ch.exp(log_gates)).sample((n_samples,)) # (S, B)
        curvatures_selected = curvatures[indices, ch.arange(batch_size).unsqueeze(0), :] # (S, B, 1)
        translation = 5 * ch.sin(3 * curvatures_selected.expand(-1, -1, 2) + 3)
        rotation = ch.tensor([[-0.9751, -0.2217], [0.2217, -0.9751]], dtype=ch.float32,
                             device=device).expand(batch_size, -1, -1)
        gaus_samples = MultivariateNormal(loc=means, covariance_matrix=covs).sample((n_samples, batch_size)) # (S, B, 2)

        x = ch.zeros_like(gaus_samples)

        # transform to banana shaped distribution
        x[..., 0] = gaus_samples[..., 0]
        x[..., 1:] = gaus_samples[..., 1:] + curvatures_selected * gaus_samples[..., 0].unsqueeze(-1) ** 2 - curvatures_selected

        # rotate samples
        x = ch.einsum('bij,ijk->bik', x, rotation) # (S, B, 2)

        # translate samples
        x = x + translation
        return x # (S, B, dz)

    def log_prob_tgt(self,
                     contexts: ch.Tensor,
                     samples: ch.Tensor
                     ) -> ch.Tensor:

        curvatures = self.curvature_fn(contexts).transpose(0, 1) # (B, O, 1)
        batch_size, n_components, _ = curvatures.shape
        device = contexts.device
        n_samples = samples.shape[0] # (S, B, O, dz) or (S, B, dz)

        log_gates = get_weights(contexts, n_components).expand(n_samples, -1, -1) # (S, B, O)
        means = ch.zeros(2, dtype=ch.float32, device=device)
        covs = 0.4 * ch.eye(2, dtype=ch.float32, device=device)

        translation = 5 * ch.sin(3 * curvatures.expand(-1, -1, 2) + 3)
        rotation = ch.tensor([[-0.9751, -0.2217], [0.2217, -0.9751]], dtype=ch.float32,
                             device=device).expand(batch_size, -1, -1)
        if samples.dim() == 3:  # (S, B, dz)
            samples = samples.unsqueeze(2)  # (S, B, 1, 2)
            samples = samples - translation # (S, B, O, 2)
            samples = ch.einsum('sboi,bji->sboj', samples, rotation) # (S, B, O, dz)
            # inverse transform
            gaus_samples = samples.clone()
            gaus_samples[..., 1:] = samples[..., 1:] - curvatures * samples[..., 0].unsqueeze(-1) ** 2 + curvatures
            log_prob_component = MultivariateNormal(loc=means, covariance_matrix=covs).log_prob(gaus_samples) # (S, B, O)
            log_prob = ch.logsumexp(log_gates + log_prob_component, dim=-1) # (S, B)
        else:
            log_prob = []
            for o in range(n_components):
                samples_o = samples[:, :, o]
                samples_o = samples_o.unsqueeze(2) # (S, B, O, dz)
                samples_o = samples_o - translation  # (S, B, O, dz)
                samples_o = ch.einsum('sboi,bji->sboj', samples_o, rotation) # (S, B, dz)
                # inverse transform
                gaus_samples = samples_o.clone()
                gaus_samples[..., 1:] = samples_o[..., 1:] - curvatures * samples_o[..., 0].unsqueeze(-1) ** 2 + curvatures
                log_prob_component = MultivariateNormal(loc=means, covariance_matrix=covs).log_prob(gaus_samples) # (S, B, O)
                log_prob_o = ch.logsumexp(log_gates + log_prob_component, dim=-1) # (S, B)
                log_prob.append(log_prob_o)
            log_prob = ch.stack(log_prob, dim=-1) # (S, B, O)
        return log_prob

    def visualize(self,
                  contexts: ch.Tensor,
                  n_samples: int = None
                  ):
        """
        Visualizes the target distribution given the contexts.

        Parameters:
        - contexts (ch.Tensor): The context vectors parameterizing the BMM.
        - n_samples (int, optional): Number of samples to draw for visualization. Default is None.
        """
        fig, axes = plt.subplots(1, contexts.shape[0], figsize=(5 * contexts.shape[0], 5))
        for i, c in enumerate(contexts):
            x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
            grid = ch.tensor(np.c_[x.ravel(), y.ravel()], dtype=ch.float32)
            pdf_values = self.log_prob_tgt(c.unsqueeze(0), grid.unsqueeze(1))
            pdf_values = pdf_values.exp().view(100, 100).numpy()

            ax = axes[i]
            ax.contourf(x, y, pdf_values, levels=50, cmap='viridis')
            if n_samples is not None:
                samples = self.sample(c.unsqueeze(0), n_samples)
                ax.scatter(samples[..., 0], samples[..., 1], color='red', alpha=0.5)
            ax.set_title(f'Target {i + 1} with context {c}')

        plt.tight_layout()
        plt.show()
