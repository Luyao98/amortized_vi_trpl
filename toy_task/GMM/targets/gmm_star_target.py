import math
import torch as ch
from torch.distributions import uniform, MultivariateNormal, Categorical
import numpy as np
import matplotlib.pyplot as plt
from toy_task.GMM.targets.abstract_target import AbstractTarget


class ConditionalSTARTarget(AbstractTarget, ch.nn.Module):
    def __init__(self, gate_fn, mean_fn, chol_fn, context_dim, context_bound_low=-3, context_bound_high=3):
        super().__init__()
        self.context_bound_low = context_bound_low
        self.context_bound_high = context_bound_high
        self.context_dim = context_dim
        self.context_dist = uniform.Uniform(context_bound_low, context_bound_high)
        self.gate_fn = gate_fn
        self.mean_fn = mean_fn
        self.chol_fn = chol_fn

    def get_contexts(self, n_contexts):
        size = ch.Size([n_contexts, self.context_dim])
        contexts = self.context_dist.sample(size)  # return shape(n_contexts, 1)
        return contexts

    def sample(self, contexts, n_samples):
        device = contexts.device
        gate = self.gate_fn(contexts).to(device)  # [n_contexts, n_components]
        means = self.mean_fn(contexts).to(device) # [n_contexts, n_components, n_features]
        chols = self.chol_fn(contexts).to(device) # [n_contexts, n_components, n_features, n_features]

        indices = Categorical(ch.exp(gate)).sample(ch.Size([n_samples])).transpose(0, 1) # [n_contexts, n_samples]
        chosen_means = ch.gather(means, 1, indices.unsqueeze(-1).expand(-1, -1, means.shape[-1]))
        chosen_chols = ch.gather(chols, 1, indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, means.shape[-1], means.shape[-1]))
        chosen_covs = chosen_chols @ chosen_chols.transpose(-1, -2)
        samples = MultivariateNormal(chosen_means, covariance_matrix=chosen_covs).sample() # [n_contexts, n_samples, n_features]
        return samples

    def log_prob_tgt(self, contexts, samples):
        device = contexts.device
        gate = self.gate_fn(contexts).to(device)  # [n_contexts, n_components]
        means = self.mean_fn(contexts).to(device) # [n_contexts, n_components, n_features]
        chols = self.chol_fn(contexts).to(device) # [n_contexts, n_components, n_features, n_features]
        batch_size, n_components, n_features = means.shape

        if samples.dim() == 3:
            n_samples = samples.shape[1]
        else:
            # for plotting
            n_samples = samples.shape[0]
            samples = samples.unsqueeze(0).expand(batch_size, -1, -1)

        gate_expanded = gate.unsqueeze(1).expand(-1, n_samples, -1)
        means_expanded = means.unsqueeze(1).expand(-1, n_samples, -1, -1)
        chols_expanded = chols.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
        covs_expanded = chols_expanded @ chols_expanded.transpose(-1, -2)
        samples_expanded = samples.unsqueeze(2).expand(-1, -1, n_components, -1)

        mvn = MultivariateNormal(means_expanded, covariance_matrix=covs_expanded)
        log_probs = mvn.log_prob(samples_expanded) + gate_expanded # [batch_size, n_samples, n_components]
        log_probs = ch.logsumexp(log_probs, dim=2)  # [batch_size, n_samples]
        return log_probs

    def visualize(self, contexts, n_samples=None):
        fig, axes = plt.subplots(1, contexts.shape[0], figsize=(5 * contexts.shape[0], 5))
        for i, c in enumerate(contexts):
            x, y = np.meshgrid(np.linspace(-5, 5, 300), np.linspace(-5, 5, 300))
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
        diag1 = ch.sin(c[:, 0]) + 1.1  # Shape: (batch_size,)
        diag2 = 0.05 * ch.cos(c[:, 1]) + 0.08
        diag3 = 0.05 * ch.cos(c[:, 0]) + 0.08
        zeros = ch.zeros_like(c[:, 0])
        if c.shape[-1] == 1:
            chol = ch.stack([ch.stack([diag1, zeros], dim=1),
                             ch.stack([zeros, diag3], dim=1)], dim=1)  # Shape: (batch_size, 2, 2)
        elif c.shape[-1] == 2:
            chol = ch.stack([ch.stack([diag1, zeros], dim=1),
                             ch.stack([zeros, diag2], dim=1)], dim=1)  # Shape: (batch_size, 2, 2)
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
        mus = [ch.tensor([2.5, 0.0], device=contexts.device)]
        # Other components generated through rotation
        theta = 2 * math.pi / n_components
        rotation = U(theta).to(contexts.device)
        for _ in range(n_components - 1):
            mus.append(rotation @ mus[-1])  # Rotate the previous mean
        mu_true = ch.stack(mus, dim=0)  # Shape: (n_components, 2)
        return mu_true.repeat(batch_size, 1, 1)
    return generate_star_means


def get_star_target(n_components, context_dim):
    gate_target = get_weights_fn(n_components)
    mean_target = get_mean_fn(n_components)
    chol_target = get_chol_fn(n_components)
    gmm_target = ConditionalSTARTarget(gate_target, mean_target, chol_target, context_dim)
    return gmm_target


if __name__ == "__main__":

    target = get_star_target(5, 2)
    contexts = target.get_contexts(3)  # (3, 1)
    # samples = target.sample(contexts, 1000)  # (3, 1000, 2)
    target.visualize(contexts, n_samples=20)
    # contexts = ch.tensor([[-0.3],
    #                       [0.7],
    #                       [-1.8]])
    # print(ch.exp(target.gate_fn(contexts)))
    # target.visualize(contexts)