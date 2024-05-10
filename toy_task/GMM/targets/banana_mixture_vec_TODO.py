import torch as ch
from torch.distributions import MultivariateNormal, Categorical, uniform
import numpy as np
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt
from toy_task.GMM.targets.abstract_target import AbstractTarget
# from toy_task.GMM.utils.network_utils import set_seed
#
# set_seed(1998)


class BananaDistribution:
    def __init__(self, curvature, dim):
        self.dim = dim  # feature dimension
        self.curvature = curvature
        self.device = curvature.device
        self.batch_size = curvature.shape[0]
        self.mean = ch.zeros(self.batch_size, self.dim, device=self.device)
        self.covariance = 0.4 * ch.eye(self.dim, device=self.device).repeat(self.batch_size, 1, 1)
        self.base_dist = MultivariateNormal(self.mean, self.covariance)
        self.translation = ch.sin(curvature)
        # self.rotation = ch.tensor(special_ortho_group.rvs(self.dim), dtype=ch.float32, device=self.device)
        self.rotation = ch.tensor([[-0.9751, -0.2217],
                                   [0.2217, -0.9751]], dtype=ch.float32, device=self.device).repeat(self.batch_size, 1, 1)

    def sample(self, num_samples):
        gaus_samples = self.base_dist.sample(ch.Size(num_samples,)).transpose(0, 1).to(self.device)
        x = ch.zeros_like(gaus_samples)

        # transform to banana shaped distribution
        x[..., 0] = gaus_samples[..., 0]
        x[..., 1:] = gaus_samples[..., 1:] + self.curvature * gaus_samples[..., 0].unsqueeze(1) ** 2 - self.curvature

        # rotate samples
        x = ch.matmul(x, self.rotation)

        # translate samples
        x += self.translation
        return x

    def log_prob(self, samples):  # 这里device应该都是samples的device
        # translate and rotate back the samples
        samples = samples - self.translation
        samples = ch.matmul(samples, self.rotation.t())

        # inverse transform
        # gaus_samples = samples.clone().to(self.device)
        gaus_samples = samples.clone()
        gaus_samples[..., 1:] = samples[..., 1:] - self.curvature * samples[..., 0].unsqueeze(1) ** 2 + self.curvature

        # compute log probabilities
        log = self.base_dist.log_prob(gaus_samples)
        return log


class BananaMixtureModel(ch.nn.Module):
    def __init__(self, get_curvature, num_components=1, dim=2):
        super().__init__()
        self.num_components = num_components
        self.ndim = dim
        self.means = [ch.zeros(dim) for _ in range(num_components)]
        self.covariances = [ch.eye(dim) * 0.4 for _ in range(num_components)]
        # self.translations = [ch.rand(dim) * 20 - 10 for _ in range(num_components)]
        self.translations = [1 * (ch.sin(curvature) + i) for i in range(num_components)]
        # self.curvatures = [3.0 for _ in range(num_components)]
        self.curvatures = [curvature for _ in range(num_components)]
        self.mixture_weights = ch.ones(num_components, device=curvature.device) / num_components

        self.bananas = [BananaDistribution(curvature=self.curvatures[i], mean=self.means[i],
                                           covariance=self.covariances[i], translation=self.translations[i])
                        for i in range(num_components)]

    def get_contexts(self, n_contexts):
        contexts = self.context_dist.sample((n_contexts, 1))
        return contexts

    def sample(self, num_samples):
        component_assignments = Categorical(self.mixture_weights).sample((num_samples,))
        samples = ch.cat([self.bananas[i].sample((ch.sum(component_assignments == i).item(),))
                          for i in range(self.num_components)])
        return samples[ch.randperm(num_samples)]  # shuffle the samples

    def log_prob(self, samples):
        log_probs = ch.stack([banana.log_prob(samples) for banana in self.bananas])
        weights = ch.log(self.mixture_weights.unsqueeze(1))
        return ch.logsumexp(weights + log_probs, dim=0)

    def visualize(self, samples=None):
        x, y = np.meshgrid(np.linspace(-15, 15, 300), np.linspace(-15, 15, 300))
        grid = ch.tensor(np.c_[x.ravel(), y.ravel()], dtype=ch.float32)
        pdf_values = ch.exp(self.log_prob(grid))
        pdf_values = pdf_values.view(300, 300).numpy()

        plt.contourf(x, y, pdf_values, levels=50, cmap='viridis')
        if samples is not None:
            plt.scatter(samples[:, 0], samples[:, 1], color='red', alpha=0.5)
        plt.show()


def get_curvature(contexts):
    curvature = contexts
    return curvature

# # test
# target = BananaMixtureTarget()
# # contexts_test = target.get_contexts(3)
# contexts_test = ch.tensor([[0.3],
#                            [-2.0],
#                            [1.0]])
# print(contexts_test)
# target.visualize(contexts_test)

