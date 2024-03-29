import torch as ch


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

    def get_contexts_1g(self, n_contexts):
        c = self.context_dist.sample((n_contexts, self.context_dim))  # return shape(n_contexts, 1)
        return c

    def log_prob_1g(self, c, x):
        return ch.distributions.MultivariateNormal(loc=self.mean_fn(c), covariance_matrix=self.std_fn(c)).log_prob(x)


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

