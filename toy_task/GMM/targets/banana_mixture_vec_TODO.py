import torch
from torch.distributions import MultivariateNormal, uniform, Categorical
import numpy as np
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt
from toy_task.GMM.targets.abstract_target import AbstractTarget

# from toy_task.GMM.utils.network_utils import set_seed
#
# set_seed(1998)


class BananaDistribution:
    def __init__(self, curvature, mean, covariance, translation):
        self.curvature = curvature.unsqueeze(-1)
        self.mean = mean
        self.covariance = covariance
        self.base_dist = MultivariateNormal(mean, covariance)
        self.translation = translation.unsqueeze(1)
        self.rotation = torch.tensor(special_ortho_group.rvs(dim=mean.shape[1]), dtype=torch.float32)

    def log_prob(self, samples):
        """
        :param samples: (batch_size, n_samples, features)
        :return: log_probs with shape (batch_size, n_samples)
        """
        # translate and rotate back the samples
        samples = samples - self.translation
        samples = torch.matmul(samples, self.rotation.t())

        # inverse transform
        gaus_samples = samples.clone()
        gaus_samples[..., 1:] = samples[..., 1:] - self.curvature * samples[..., 0].unsqueeze(
            -1) ** 2 + self.curvature

        n_samples = gaus_samples.shape[1]
        means_expanded = self.mean.unsqueeze(1).expand(-1, n_samples, -1)
        covs_expanded = self.covariance.unsqueeze(1).expand(-1, n_samples, -1, -1)
        mvn = MultivariateNormal(means_expanded, covariance_matrix=covs_expanded)
        log_probs = mvn.log_prob(gaus_samples)
        return log_probs

    def sample(self, n_samples):
        gaus_samples = self.base_dist.sample(torch.Size(n_samples,)).permute(1, 0, 2)
        x = torch.zeros_like(gaus_samples)

        # Transform to banana shaped distribution
        x[..., 0] = gaus_samples[..., 0]
        x[..., 1:] = gaus_samples[..., 1:] + self.curvature * gaus_samples[..., 0].unsqueeze(-1)**2 - self.curvature

        # Rotate samples
        x = torch.matmul(x, self.rotation)

        # Translate samples
        x += self.translation
        return x


class BananaMixtureTarget(AbstractTarget):
    def __init__(self, n_components=1, dim=2, context_bound_low=-3, context_bound_high=3):
        self.dim = dim
        self.n_components = n_components
        self.context_dist = uniform.Uniform(context_bound_low, context_bound_high)

    def get_contexts(self, n_contexts):
        contexts = self.context_dist.sample((n_contexts, 1))  # return shape(n_contexts, 1)
        return contexts

    def log_prob_tgt(self, contexts, samples):
        if samples.dim() == 3:
            print("Shape of samples is correct!")
        else:
            # for plotting
            samples = samples.unsqueeze(0).expand(contexts.shape[0], -1, -1)

        curvatures = contexts
        means = [torch.zeros(contexts.shape[0], self.dim) for _ in range(self.n_components)]
        covariances = [0.4 * torch.eye(self.dim).unsqueeze(0).repeat(contexts.shape[0], 1, 1) for _ in
                       range(self.n_components)]
        translations = [10 * torch.ones(contexts.shape[0], self.dim) - 5 for _ in range(self.n_components)]
        mixture_weights = torch.tensor(1.0 / self.n_components)
        bananas = [BananaDistribution(curvature=curvatures, mean=means[i], covariance=covariances[i],
                                      translation=translations[i]) for i in range(self.n_components)]
        log_probs = torch.stack([banana.log_prob(samples) for banana in bananas])

        weights = torch.log(mixture_weights)
        return torch.logsumexp(weights + log_probs, dim=0)  # shape(n_contexts, n_samples)

    def sample(self, contexts, n_samples):
        curvatures = contexts
        means = [torch.zeros(contexts.shape[0], self.dim) for _ in range(self.n_components)]
        covariances = [0.4 * torch.eye(self.dim).unsqueeze(0).repeat(contexts.shape[0], 1, 1) for _ in
                       range(self.n_components)]
        translations = [10 * torch.ones(contexts.shape[0], self.dim) - 5 for _ in range(self.n_components)]
        mixture_weights = torch.ones(self.n_components) / self.n_components
        bananas = [BananaDistribution(curvature=curvatures, mean=means[i], covariance=covariances[i],
                                      translation=translations[i]) for i in range(self.n_components)]

        component_assignments = Categorical(mixture_weights).sample((n_samples,))
        samples = torch.cat([bananas[i].sample((torch.sum(component_assignments == i).item(),))
                             for i in range(self.n_components)])

        # shuffle the samples
        n_batches = samples.size(0)
        random_indices = torch.stack([torch.randperm(n_samples) for _ in range(n_batches)])
        shuffled_samples = torch.gather(samples, 1, random_indices.unsqueeze(-1).expand(-1, -1, samples.size(2)))
        return shuffled_samples

    def visualize(self, contexts, samples=None):
        x, y = np.meshgrid(np.linspace(-10, 10, 300), np.linspace(-10, 10, 300))
        grid = torch.tensor(np.c_[x.ravel(), y.ravel()], dtype=torch.float32)
        pdf_values = torch.exp(self.log_prob_tgt(contexts, grid))
        fig, axes = plt.subplots(2, len(contexts), figsize=(contexts.shape[0] * 10, 10))
        for i in range(contexts.shape[0]):
            ax = axes[0, i]
            ax.clear()
            pdf_values_i = pdf_values[i].view(300, 300).numpy()
            ax.contourf(x, y, pdf_values_i, levels=100, cmap='viridis')
            if samples is not None:
                ax = axes[1, i]
                ax.scatter(samples[i, :, 0], samples[i, :, 1], color='red', alpha=0.5)
            ax.set_title(f'Target density with Context: {contexts[i].item():.2f}')
        plt.tight_layout()
        plt.show()


model = BananaMixtureTarget()
contexts = model.get_contexts(3)
# target.visualize(contexts)

samples = model.sample(contexts, 500)
model.visualize(contexts, samples=samples)
