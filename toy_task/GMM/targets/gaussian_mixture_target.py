import torch as ch
from torch.distributions import uniform, MultivariateNormal, Categorical
from toy_task.GMM.targets.abstract_target import AbstractTarget
import numpy as np
import matplotlib.pyplot as plt


class ConditionalGMMTarget(AbstractTarget, ch.nn.Module):
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
        samples = MultivariateNormal(chosen_means, scale_tril=chosen_chols).sample() # [n_contexts, n_samples, n_features]
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
        samples_expanded = samples.unsqueeze(2).expand(-1, -1, n_components, -1)

        mvn = MultivariateNormal(means_expanded, scale_tril=chols_expanded)
        log_probs = mvn.log_prob(samples_expanded) + gate_expanded # [batch_size, n_samples, n_components]
        log_probs = ch.logsumexp(log_probs, dim=2)  # [batch_size, n_samples]
        return log_probs

    def visualize(self, contexts, n_samples=None):
        fig, axes = plt.subplots(1, contexts.shape[0], figsize=(5 * contexts.shape[0], 5))
        for i, c in enumerate(contexts):
            x, y = np.meshgrid(np.linspace(-30, 30, 300), np.linspace(-30, 30, 300))
            grid = ch.tensor(np.c_[x.ravel(), y.ravel()], dtype=ch.float32)
            pdf_values = self.log_prob_tgt(c.unsqueeze(0), grid)
            pdf_values = pdf_values.exp().view(300, 300).numpy()

            ax = axes[i]
            ax.contourf(x, y, pdf_values, levels=50, cmap='viridis')
            if n_samples is not None:
                samples = self.sample(c.unsqueeze(1), n_samples)
                ax.scatter(samples[..., 0], samples[..., 1], color='red', alpha=0.5)
            ax.set_title(f'Target {i + 1} with context {c}')

        plt.tight_layout()
        plt.show()


def get_weights_fn(n_components):
    def get_weights(c):
        weights = []
        if c.shape[-1] == 1:
            for i in range(n_components):
                if i % 2 == 0:
                    weights.append(ch.sin((i + 1) * c[:, 0]))
                else:
                    weights.append(ch.cos((i + 1) * c[:, 0]))
        elif c.shape[-1] == 2:
            for i in range(n_components):
                if i % 2 == 0:
                    weights.append(ch.sin((i + 1) * c[:, 0]))
                else:
                    weights.append(ch.cos((i + 1) * c[:, 1]))
        else:
            raise ValueError('Context dimension must be 1 or 2')
        weights = ch.stack(weights, dim=1)
        log_weights = ch.log_softmax(weights, dim=1)
        return log_weights
    return get_weights


def get_chol_fn(n_components):
    def cat_chol(c):
        chols = []
        if c.shape[-1] == 1:
            for i in range(n_components):
                chol = ch.stack([
                    ch.stack([0.5 * ch.sin((i + 1) * c[:, 0]) + 0.8, ch.zeros_like(c[:, 0])], dim=1),
                    ch.stack([ch.sin(3 * c[:, 0]) * ch.cos(3 * c[:, 0]), 0.5 * ch.cos((i + 1) * c[:, 0]) + 0.8], dim=1)], dim=1)
                chols.append(chol)
        elif c.shape[-1] == 2:
            for i in range(n_components):
                chol = ch.stack([
                    ch.stack([0.3 * ch.sin((i + 1) * c[:, 0]) + 0.5, ch.zeros_like(c[:, 0])], dim=1),
                    ch.stack([0.3 * ch.sin(c[:, 0]) * ch.cos(c[:, 1]), 0.3 * ch.cos((i + 1) * c[:, 1]) + 0.5], dim=1)], dim=1)
                chols.append(chol)
        return ch.stack(chols, dim=1)
    return cat_chol


def spiral(t, c, a=0.3):
    if c.shape[-1] == 1:
        b = t + 0.1 * c
    elif c.shape[-1] == 2:
        b = t + 0.1 * c[:, 0].unsqueeze(1)
    else:
        raise ValueError('Context dimension must be 1 or 2')
    x = a * t * ch.cos(b)
    y = a * t * ch.sin(b)
    return ch.stack([x, y], dim=-1)


def get_mean_fn(n_components):
    def generate_spiral_means(contexts):
        t_values = np.linspace(0, 14 * np.pi, n_components, endpoint=False) # 2D 10components use 14, otherwise 35
        means = spiral(ch.tensor(t_values, dtype=ch.float32, device=contexts.device), contexts)
        return means
    return generate_spiral_means


def get_gmm_target(n_components, context_dim):
    gate_target = get_weights_fn(n_components)
    mean_target = get_mean_fn(n_components)
    chol_target = get_chol_fn(n_components)
    gmm_target = ConditionalGMMTarget(gate_target, mean_target, chol_target, context_dim)
    return gmm_target


if __name__ == "__main__":

    target = get_gmm_target(10, 2)
    contexts = target.get_contexts(3)  # (3, 1)
    # samples = target.sample(contexts, 1000)  # (3, 1000, 2)
    # target.visualize(contexts, n_samples=20)
    # contexts = ch.tensor([[-0.3],
    #                       [0.7],
    #                       [-1.8]])
    # print(ch.exp(target.gate_fn(contexts)))
    target.visualize(contexts)