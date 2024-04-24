import torch
import numpy as np
import matplotlib.pyplot as plt


class ConditionalGaussianTarget(torch.nn.Module):
    def __init__(self, mean_fn, std_fn, context_dim=1, context_bound_low=-3, context_bound_high=3):
        super().__init__()
        self.context_dist = torch.distributions.uniform.Uniform(context_bound_low, context_bound_high)
        self.context_bound_low = context_bound_low
        self.context_bound_high = context_bound_high
        self.context_dim = context_dim
        self.mean_fn = mean_fn
        self.std_fn = std_fn

    def get_contexts(self, n_contexts):
        return self.context_dist.sample(n_contexts)

    def log_prob(self, c, x):
        return torch.distributions.Normal(loc=self.mean_fn(c), scale=self.std_fn(c)).log_prob(x)

    def visualize(self):
        x1_test = np.linspace(self.context_bound_low, self.context_bound_high, 100)
        x2_test = np.linspace(-5, 5, 100)
        x1, x2 = np.meshgrid(x1_test, x2_test)
        x1_flat = torch.from_numpy(x1.reshape(-1, 1).astype(np.float32)).detach()
        x2_flat = torch.from_numpy(x2.reshape(-1, 1).astype(np.float32)).detach()
        contours = (self.log_prob(x1_flat, x2_flat).exp().view(x1.shape).detach() + 1e-6)
        fig, ax = plt.subplots()
        plt.xlim([self.context_bound_low, self.context_bound_high])
        plt.xlabel('c')
        plt.ylabel('x')
        f = ax.contourf(x1, x2, contours, levels=100)
        fig.colorbar(f)
        plt.show()


def get_std_fn(type):
    if type == 'constant':
        return lambda c: torch.ones_like(c)
    elif type == 'linear':
        return lambda c: c + 0.01
    elif type == 'periodic':
        return lambda c: .5 * torch.sin(3 * c) + 1.1


def get_mean_fn(type):
    if type == 'constant':
        return lambda c: torch.zeros_like(c)


if __name__ == '__main__':
    mean_fn = get_mean_fn('constant')
    std_fn = get_std_fn('periodic')
    ConditionalGaussianTarget(mean_fn, std_fn).visualize()