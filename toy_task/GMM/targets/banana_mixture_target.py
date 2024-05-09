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
    def __init__(self, curvature, mean, covariance, translation):
        self.device = curvature.device
        self.dim = mean.shape[0]
        self.curvature = curvature
        self.mean = mean.to(self.device)
        self.covariance = covariance.to(self.device)
        # self.base_dist = MultivariateNormal(mean, covariance)
        self.translation = translation.to(self.device)
        # self.rotation = ch.tensor(special_ortho_group.rvs(self.dim), dtype=ch.float32, device=self.device)
        # print(self.rotation)
        self.rotation = ch.tensor([[-0.9751, -0.2217],
                                   [0.2217, -0.9751]], dtype=ch.float32, device=self.device)

    def sample(self, num_samples):
        gaus_samples = MultivariateNormal(self.mean, self.covariance).sample(ch.Size(num_samples, ))
        x = ch.zeros_like(gaus_samples).to(self.device)

        # transform to banana shaped distribution
        x[:, 0] = gaus_samples[:, 0]
        x[:, 1:] = gaus_samples[:, 1:] + self.curvature * gaus_samples[:, 0].unsqueeze(1) ** 2 - self.curvature

        # rotate samples
        x = ch.matmul(x, self.rotation)

        # translate samples
        x += self.translation
        return x

    def log_prob(self, samples):
        # translate and rotate back the samples
        samples = samples - self.translation
        samples = ch.matmul(samples, self.rotation.t())

        # inverse transform
        gaus_samples = samples.clone().to(self.device)
        gaus_samples[:, 1:] = samples[:, 1:] - self.curvature * samples[:, 0].unsqueeze(1) ** 2 + self.curvature

        # compute log probabilities
        log = MultivariateNormal(self.mean, self.covariance).log_prob(gaus_samples)
        return log


class BananaMixtureModel:
    def __init__(self, curvature, num_components, dim):
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


class BananaMixtureTarget(AbstractTarget, ch.nn.Module):
    def __init__(self, num_components=1, dim=2):
        super().__init__()
        self.context_dist = uniform.Uniform(-2, 2)
        self.num_components = num_components
        self.dim = dim
        self.models = None
        self.current_contexts = None

    def initialize_models(self, contexts):
        if (self.models is None or self.current_contexts is None or
                self.current_contexts.shape[0] != contexts.shape[0] or
                self.current_contexts.device != contexts.device or
                ch.all(self.current_contexts != contexts)):
            # self.current_contexts.clone().to(contexts.device).shape[0] != contexts.shape[0] or
            # not ch.all(self.current_contexts.clone().to(contexts.device) == contexts).item()):
            # or self.current_contexts.device != contexts.device):         # only for one context test
            self.current_contexts = contexts.clone().detach().to(contexts.device)
            self.models = [BananaMixtureModel(curvature=self.current_contexts[i], num_components=self.num_components,
                                              dim=self.dim) for i in range(self.current_contexts.shape[0])]

    def get_contexts(self, n_contexts):
        contexts = self.context_dist.sample((n_contexts, 1))
        return contexts

    def sample(self, contexts, num_samples):
        self.initialize_models(contexts)
        batch_samples = [model.sample(num_samples) for model in self.models]
        return ch.stack(batch_samples)  # (num_batches, num_samples, dim)

    def log_prob_tgt(self, contexts, samples):
        self.initialize_models(contexts)
        if samples.dim() == 2:
            samples = samples.unsqueeze(0).expand(contexts.shape[0], -1, -1)
        log_probs = [model.log_prob(samples[i]) for i, model in enumerate(self.models)]
        return ch.stack(log_probs)

    def visualize(self, contexts, num_samples=None):
        self.initialize_models(contexts)
        fig, axes = plt.subplots(1, len(self.models), figsize=(5 * len(self.models), 5))
        if len(self.models) == 1:
            axes = [axes]  # ensure axes is iterable for a single target case

        for i, model in enumerate(self.models):
            x, y = np.meshgrid(np.linspace(-15, 15, 300), np.linspace(-15, 15, 300))
            grid = ch.tensor(np.c_[x.ravel(), y.ravel()], dtype=ch.float32).to(contexts.device)
            pdf_values = ch.exp(model.log_prob(grid))
            pdf_values = pdf_values.view(300, 300).numpy()

            ax = axes[i]
            ax.contourf(x, y, pdf_values, levels=50, cmap='viridis')
            if num_samples is not None:
                samples = model.sample(num_samples)
                ax.scatter(samples[:, 0], samples[:, 1], color='red', alpha=0.5)
            ax.set_title(f'Model {i + 1}')

        plt.tight_layout()
        plt.show()


# # test
# target = BananaMixtureTarget()
# # contexts_test = target.get_contexts(3)
# contexts_test = ch.tensor([[0.3],
#                            [-2.0],
#                            [1.0]])
# print(contexts_test)
# target.visualize(contexts_test)
