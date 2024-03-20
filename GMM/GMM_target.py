import torch as ch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import uniform, MultivariateNormal

ch.manual_seed(37)
np.random.seed(37)


class ConditionalGMMTarget(ch.nn.Module):
    def __init__(self, mean_fn, std_fn, context_dim=1, context_bound_low=-3, context_bound_high=3):
        super().__init__()
        self.context_bound_low = context_bound_low
        self.context_bound_high = context_bound_high
        self.context_dim = context_dim

        self.context_dist = uniform.Uniform(context_bound_low, context_bound_high)

        self.mean_fn = mean_fn
        self.std_fn = std_fn

    def get_contexts(self, n_contexts):
        contexts = self.context_dist.sample((n_contexts, self.context_dim))  # return shape(n_contexts, 1)
        # print(contexts)
        return contexts

    def log_prob(self, contexts, samples):
        means = self.mean_fn(contexts)
        covs = self.std_fn(contexts)
        n_components = means.shape[1]
        gate = ch.tensor(1.0 / n_components)
        log = ch.zeros(samples.shape[0])
        for i in range(n_components):
          loc = means[:, i,]
          covariance_matrix = covs[:, i,]
          sub_log = MultivariateNormal(loc, covariance_matrix).log_prob(samples)
          log = log + ch.log(gate) + sub_log
        return log

    def visualize(self, contexts):
        x1_test = np.linspace(-10, 30, 100)
        x2_test = np.linspace(-10, 30, 100)
        x1, x2 = np.meshgrid(x1_test, x2_test)
        x1_flat = ch.from_numpy(x1.reshape(-1, 1).astype(np.float32)).detach()
        x2_flat = ch.from_numpy(x2.reshape(-1, 1).astype(np.float32)).detach()
        x = ch.cat((x1_flat, x2_flat), dim=1)

        loc = self.mean_fn(contexts)
        covariance_matrix = self.std_fn(contexts)
        for i, c in enumerate(contexts):
            fig, ax = plt.subplots(figsize=(8, 8))
            means = loc[i, ]
            covs = covariance_matrix[i, ]
            for j in range(len(means)):
                mean = means[j,]
                # print("mean", mean)
                cov = covs[j,]
                # print("cov", cov)
                mvn = MultivariateNormal(mean, covariance_matrix=cov)
                prob = mvn.log_prob(x).exp().view(x1.shape).detach() + 1e-6
                ax.contourf(x1, x2, prob, levels=100, alpha=0.3)
                # ax.contour(x1, x2, prob, levels=10, colors='k', alpha=0.5)

            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_title(f'Conditional GMM Components for Context {c}')

            plt.tight_layout()
            plt.show()


def get_cov_fn(n_components):
    def cat_cov(c):
        covs = []
        for i in range(n_components):
            sub1_chol = ch.cat([i + 0.5 * ch.sin(c) + 1.1, ch.zeros_like(c)], dim=1)
            sub2_chol = ch.cat([ch.sin(c) * ch.cos(c), i + 0.5 * ch.cos(c) + 1.1], dim=1)
            chol = ch.stack((sub1_chol, sub2_chol), dim=1)
            cov = chol @ chol.permute(0, 2, 1)
            covs.append(cov)
        return ch.stack(covs, dim=1)
    return cat_cov


def get_mean_fn(n_components):
    def cat_mean(c):
        mean = []
        for i in range(n_components):
            sub_mean = ch.cat([5 * (i + ch.sin(c)), 5 * (i + ch.cos(c))], dim=1)
            mean.append(sub_mean)
        return ch.stack(mean, dim=1)
    return cat_mean


if __name__ == '__main__':
    mean_fn = get_mean_fn(3)
    std_fn_sin = get_cov_fn(3)

    target = ConditionalGMMTarget(mean_fn, std_fn_sin)
    c = target.get_contexts(2)
    target.visualize(c)
