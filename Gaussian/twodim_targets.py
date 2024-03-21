import torch as ch
import numpy as np
import matplotlib.pyplot as plt


class ConditionalGaussianTarget(ch.nn.Module):
    def __init__(self, mean_fn, std_fn, context_dim=1, context_bound_low=-3, context_bound_high=3):
        super().__init__()
        self.context_dist = ch.distributions.uniform.Uniform(context_bound_low, context_bound_high)
        # self.context_dist = ch.distributions.Normal(0, 1)
        self.context_bound_low = context_bound_low
        self.context_bound_high = context_bound_high
        self.context_dim = context_dim
        self.mean_fn = mean_fn
        self.std_fn = std_fn

    def get_contexts(self, n_contexts):
        c = self.context_dist.sample((n_contexts, self.context_dim))  # return shape(n_contexts, 1)
        return c

    def log_prob(self, c, x):
        return ch.distributions.MultivariateNormal(loc=self.mean_fn(c), covariance_matrix=self.std_fn(c)).log_prob(x)

    def visualize(self, contexts):
        x1_test = np.linspace(self.context_bound_low, self.context_bound_high, 100)
        x2_test = np.linspace(self.context_bound_low, self.context_bound_high, 100)
        x1, x2 = np.meshgrid(x1_test, x2_test)
        x1_flat = ch.from_numpy(x1.reshape(-1, 1).astype(np.float32)).detach()
        x2_flat = ch.from_numpy(x2.reshape(-1, 1).astype(np.float32)).detach()

        fig, axs = plt.subplots(1, len(contexts), figsize=(15, 5))

        for i, c in enumerate(contexts):
            print(f"Context {i + 1}: {c}")
            c_expanded = c.unsqueeze(0).expand(x1_flat.shape[0], -1)
            x = ch.cat((x1_flat, x2_flat), dim=1)
            log_probs = self.log_prob(c_expanded, x).exp().view(x1.shape).detach() + 1e-6
            axs[i].contourf(x1, x2, log_probs, levels=100)
            axs[i].set_title(f'Context: {c}')

        for ax in axs:
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')

        plt.tight_layout()
        plt.show()


def get_cov_fn(type):
    if type == 'periodic':
        def cov_matrix(c):
            chol = ch.stack([
                ch.stack([0.5 * ch.sin(3 * c[:, 0]) + 1.1, ch.zeros_like(c[:, 0])], dim=1),
                ch.stack([ch.sin(3 * c[:, 0]) * ch.cos(3 * c[:, 0]), 0.5 * ch.cos(3 * c[:, 0]) + 1.1], dim=1)], dim=1)
            return chol @ chol.permute(0, 2, 1)
        return cov_matrix


def get_mean_fn(type):
    if type == 'constant':
        return lambda c: ch.zeros((c.shape[0], 2))
    elif type == 'periodic':
        def mean_vector(c):
            mean = ch.stack([ch.sin(3 * c[:, 0]), ch.cos(3 * c[:, 0])], dim=1)
            return mean
        return mean_vector


if __name__ == '__main__':
    mean_fn = get_mean_fn('periodic')
    std_fn_sin = get_cov_fn('periodic')

    target = ConditionalGaussianTarget(mean_fn, std_fn_sin)
    n = 5
    # n_target = target.get_contexts(n)
    # print(n_target)
    n_target = ch.tensor([[-1.1443], [-0.1062], [-0.4056], [0.9927], [1.1044]])
    target.visualize(n_target)
