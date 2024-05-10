import torch as ch
from torch.distributions import uniform, MultivariateNormal, Categorical
from toy_task.GMM.targets.abstract_target import AbstractTarget
import numpy as np
import matplotlib.pyplot as plt


class ConditionalGMMTarget(AbstractTarget, ch.nn.Module):
    def __init__(self, mean_fn, chol_fn, context_dim=1, context_bound_low=-3, context_bound_high=3):
        super().__init__()
        self.context_bound_low = context_bound_low
        self.context_bound_high = context_bound_high
        self.context_dim = context_dim
        self.context_dist = uniform.Uniform(context_bound_low, context_bound_high)

        self.mean_fn = mean_fn
        self.chol_fn = chol_fn

    def get_contexts(self, n_contexts):
        contexts = self.context_dist.sample((n_contexts, self.context_dim))  # return shape(n_contexts, 1)
        return contexts

    def sample(self, contexts, n_samples):
        means = self.mean_fn(contexts).to(contexts.device)
        chols = self.chol_fn(contexts).to(contexts.device)

        gate = get_weights(contexts)
        samples = []
        for i in range(contexts.shape[0]):
            cat = Categorical(ch.exp(gate[i]))
            indices = cat.sample((n_samples,))
            chosen_means = means[i, indices]
            chosen_chols = chols[i, indices]
            normal = MultivariateNormal(chosen_means, scale_tril=chosen_chols)
            sample = normal.sample()
            samples.append(sample)
        return ch.stack(samples)  # [n_contexts, n_samples, n_features]

    def log_prob_tgt(self, contexts, samples):
        means = self.mean_fn(contexts).to(contexts.device)
        chols = self.chol_fn(contexts).to(contexts.device)
        batch_size, n_components, n_features = means.shape

        gate = get_weights(contexts)

        if samples.dim() == 3:
            n_samples = samples.shape[1]
        else:
            # for plotting
            n_samples = samples.shape[0]
            samples = samples.unsqueeze(0).expand(batch_size, -1, -1)

        means_expanded = means.unsqueeze(1).expand(-1, n_samples, -1, -1)
        chols_expanded = chols.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
        samples_expanded = samples.unsqueeze(2).expand(-1, -1, n_components, -1)

        mvn = MultivariateNormal(means_expanded, scale_tril=chols_expanded)
        log_probs = mvn.log_prob(samples_expanded)  # [batch_size, n_samples, n_components]

        gate_expanded = gate.unsqueeze(1).expand(-1, n_samples, -1)
        log_probs += gate_expanded

        log_probs = ch.logsumexp(log_probs, dim=2)  # [batch_size, n_samples]
        return log_probs

    def visualize(self, contexts, n_samples=None):
        fig, axes = plt.subplots(1, contexts.shape[0], figsize=(5 * contexts.shape[0], 5))
        for i, c in enumerate(contexts):
            x, y = np.meshgrid(np.linspace(-15, 15, 300), np.linspace(-15, 15, 300))
            grid = ch.tensor(np.c_[x.ravel(), y.ravel()], dtype=ch.float32)
            pdf_values = ch.exp(self.log_prob_tgt(c.unsqueeze(1), grid))
            pdf_values = pdf_values.view(300, 300).numpy()

            ax = axes[i]
            ax.contourf(x, y, pdf_values, levels=50, cmap='viridis')
            if n_samples is not None:
                samples = self.sample(c.unsqueeze(1), n_samples)
                ax.scatter(samples[..., 0], samples[..., 1], color='red', alpha=0.5)
            ax.set_title(f'Target {i + 1} with context {c.item()}')

        plt.tight_layout()
        plt.show()


def get_weights(c):
    """
    only for 4 components
    """
    weights = [ch.sin(c[:, 0]), ch.cos(c[:, 0]), ch.sin(10 * c[:, 0]), ch.cos(10 * c[:, 0])]
    weights = ch.stack(weights, dim=1)
    log_weights = ch.log_softmax(weights, dim=1)
    return log_weights


def get_chol_fn(n_components):
    def cat_chol(c):
        chols = []
        for i in range(n_components):
            chol = ch.stack([
                ch.stack([0.5 * ch.sin((i + 1) * c[:, 0]) + 1.1, ch.zeros_like(c[:, 0])], dim=1),
                ch.stack([ch.sin(3 * c[:, 0]) * ch.cos(3 * c[:, 0]), 0.5 * ch.cos((i + 1) * c[:, 0]) + 1.1], dim=1)], dim=1)
            chols.append(chol)
        return ch.stack(chols, dim=1)
    return cat_chol


def get_mean_fn(n_components):
    def cat_mean(c):
        # mean = []
        # for i in range(n_components):
        #     sub_mean = ch.stack([10 * ch.sin((i + 1) * c[:, 0]), 10 * ch.cos((i + 1) * c[:, 0])], dim=1)
        #     mean.append(sub_mean)

        mean1 = ch.stack([2 + ch.sin(c[:, 0]), 2 + ch.cos(c[:, 0])], dim=1)
        mean2 = ch.stack([-6 + 3 * ch.sin(c[:, 0]), -6 + 3 * ch.cos(c[:, 0])], dim=1)
        mean3 = ch.stack([8 + 4 * ch.sin(c[:, 0]), -8 + 4 * ch.cos(c[:, 0])], dim=1)
        mean4 = ch.stack([-4 + 2 * ch.sin(c[:, 0]), 4 + 2 * ch.cos(c[:, 0])], dim=1)

        mean = [mean1, mean2, mean3, mean4]
        return ch.stack(mean, dim=1)
    return cat_mean


def get_gmm_target(n_components):
    mean_target = get_mean_fn(n_components)
    chol_target = get_chol_fn(n_components)
    target = ConditionalGMMTarget(mean_target, chol_target)
    return target


# test
# target = get_gmm_target(4)
# contexts = target.get_contexts(3)  # (3, 1)
# samples = target.sample(contexts, 1000)  # (3, 1000, 2)
# log_prob = target.log_prob_tgt(contexts, samples)  # (3, 1000)
# target.visualize(contexts, n_samples=1000)
# contexts = ch.tensor([[-2.619831], [-2.6058419], [-2.871721]])
# target.visualize(contexts)
