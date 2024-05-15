import torch as ch
from torch.distributions import MultivariateNormal, Categorical, uniform
import numpy as np
# from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt
from toy_task.GMM.targets.abstract_target import AbstractTarget


class BananaDistribution:
    def __init__(self, curvature, dim):
        self.dim = dim  # feature dimension
        self.curvature = curvature
        self.device = curvature.device
        self.batch_size = curvature.shape[0]
        self.mean = ch.zeros(self.batch_size, self.dim, device=self.device)
        self.covariance = 0.4 * ch.eye(self.dim, device=self.device).repeat(self.batch_size, 1, 1)
        self.base_dist = MultivariateNormal(self.mean, self.covariance)
        self.translation = 4 * ch.sin(10 * curvature)
        # self.rotation = ch.tensor(special_ortho_group.rvs(self.dim), dtype=ch.float32, device=self.device)
        self.rotation = ch.tensor([[-0.9751, -0.2217], [0.2217, -0.9751]], dtype=ch.float32,
                                  device=self.device).repeat(self.batch_size, 1, 1)

    def sample(self, n_samples):
        gaus_samples = self.base_dist.sample((n_samples,)).transpose(0, 1).to(self.device)
        x = ch.zeros_like(gaus_samples)

        # transform to banana shaped distribution
        curvature_expand = self.curvature.unsqueeze(-1)
        x[..., 0] = gaus_samples[..., 0]
        x[..., 1:] = gaus_samples[..., 1:] + curvature_expand * gaus_samples[..., 0].unsqueeze(-1) ** 2 - curvature_expand

        # rotate samples
        x = ch.matmul(x, self.rotation)

        # translate samples
        x = x + self.translation.unsqueeze(-1)
        return x

    def log_prob(self, samples):
        # translate and rotate back the samples
        samples = samples - self.translation.unsqueeze(-1)
        samples = ch.matmul(samples, self.rotation.transpose(-1, -2))

        # inverse transform
        # gaus_samples = samples.clone().to(self.device)
        gaus_samples = samples.clone()
        curvature_expand = self.curvature.unsqueeze(-1)
        gaus_samples[..., 1:] = samples[..., 1:] - curvature_expand * samples[..., 0].unsqueeze(-1) ** 2 + curvature_expand

        # compute log probabilities
        mean_expanded = self.mean.unsqueeze(1).expand(-1, samples.shape[1], -1)
        covariance_expanded = self.covariance.unsqueeze(1).expand(-1, samples.shape[1], -1, -1)
        mvn = MultivariateNormal(mean_expanded, covariance_expanded)
        return mvn.log_prob(gaus_samples)


class BananaMixtureTarget(AbstractTarget, ch.nn.Module):
    def __init__(self, curvature_fn, n_components=3, dim=2):
        super().__init__()
        self.curvature_fn = curvature_fn
        self.n_components = n_components
        self.ndim = dim
        self.context_dist = uniform.Uniform(-2, 2)
        # self.means = [ch.zeros(dim) for _ in range(n_components)]
        # self.covariances = [ch.eye(dim) * 0.4 for _ in range(n_components)]
        # self.translations = [ch.rand(dim) * 20 - 10 for _ in range(n_components)]
        # self.translations = [1 * (ch.sin(curvature) + i) for i in range(n_components)]
        # self.curvatures = [3.0 for _ in range(n_components)]
        # self.curvatures = [curvature for _ in range(n_components)]
        # self.mixture_weights = ch.ones(n_components, device=curvature.device) / n_components

        # self.bananas = [BananaDistribution(curvature=self.curvatures[i], mean=self.means[i],
        #                                    covariance=self.covariances[i], translation=self.translations[i])
        #                 for i in range(n_components)]

    def get_contexts(self, n_contexts):
        contexts = self.context_dist.sample((n_contexts, 1))
        return contexts

    def sample(self, contexts, n_samples):
        curvatures = self.curvature_fn(contexts).transpose(0, 1).to(contexts.device)
        log_weights = get_weights(contexts)

        samples = []
        for i in range(contexts.shape[0]):
            indices = Categorical(ch.exp(log_weights[i])).sample((n_samples,))
            chosen_curvatures = curvatures[i, indices]
            banana_i = BananaDistribution(curvature=chosen_curvatures, dim=self.ndim)
            sample = banana_i.sample(1).squeeze(1)  # 因为sample返回的是[n_contexts, n_s, 2]，squeeze一下
            samples.append(sample)
        return ch.stack(samples)
        # return samples[ch.randperm(n_samples)]  # shuffle the samples

    def log_prob_tgt(self, contexts, samples):
        curvatures = self.curvature_fn(contexts).to(contexts.device)
        log_weights = get_weights(contexts)
        banana_for_prob = [BananaDistribution(curvature=curvatures[i], dim=self.ndim)
                           for i in range(self.n_components)]
        if samples.dim() == 3:
            n_samples = samples.shape[1]
        else:
            # for plotting
            n_samples = samples.shape[0]
            samples = samples.unsqueeze(0).expand(contexts.shape[0], -1, -1)
        log_probs = ch.stack([banana.log_prob(samples) for banana in banana_for_prob])
        weights = log_weights.transpose(0, 1).unsqueeze(-1).expand(-1, -1, n_samples)
        return ch.logsumexp(weights + log_probs, dim=0)

    def visualize(self, contexts, n_samples=None):
        fig, axes = plt.subplots(1, contexts.shape[0], figsize=(5 * contexts.shape[0], 5))
        for i, c in enumerate(contexts):
            x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
            grid = ch.tensor(np.c_[x.ravel(), y.ravel()], dtype=ch.float32, device=contexts.device)
            pdf_values = ch.exp(self.log_prob_tgt(c.unsqueeze(1), grid))
            pdf_values = pdf_values.view(100, 100).numpy()

            ax = axes[i]
            ax.contourf(x, y, pdf_values, levels=50, cmap='viridis')
            if n_samples is not None:
                samples = self.sample(c.unsqueeze(1), n_samples)
                ax.scatter(samples[..., 0], samples[..., 1], color='red', alpha=0.5)
            ax.set_title(f'Target {i + 1} with context {c.item()}')

        plt.tight_layout()
        plt.show()


def get_curvature_fn(contexts, n_components=3):
    if n_components == 1:
        curvature = ch.stack([contexts])
    elif n_components == 2:
        curvature = ch.stack([contexts, 0.5 * contexts + 1])
    elif n_components == 3:
        curvature = ch.stack([contexts, 0.5 * contexts + 1, -0.5 * contexts - 1])
    else:
        raise ValueError("only support 1, 2 or 3 components")
    return curvature


def get_weights(contexts, n_components=3):
    """
    the number of components is fixed
    """
    if n_components == 1:
        weights = [ch.sin(3 * contexts[:, 0])]
    elif n_components == 2:
        weights = [ch.sin(3 * contexts[:, 0]),
                   ch.cos(3 * contexts[:, 0])]
    elif n_components == 3:
        weights = [ch.sin(3 * contexts[:, 0]),
                   ch.cos(3 * contexts[:, 0]),
                   ch.cos(2 * contexts[:, 0])]
    else:
        raise ValueError("only support 1, 2 or 3 components")
    weights = ch.stack(weights, dim=1)
    log_weights = ch.log_softmax(weights, dim=1)
    return log_weights


# # test
# target = BananaMixtureTarget(get_curvature_fn)
# # contexts_test = target.get_contexts(3)
# contexts_test = ch.tensor([[-0.3],
#                            [1.3],
#                            [-1.9]])
# print(contexts_test)
# target.visualize(contexts_test)
