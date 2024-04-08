import torch as ch
from torch.distributions import uniform, MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt

# ch.manual_seed(37)
# np.random.seed(37)


class ConditionalGMMSimpleTarget(ch.nn.Module):
    def __init__(self, mean_fn, cov_fn, context_dim=1, context_bound_low=-3, context_bound_high=3):
        super().__init__()
        self.context_bound_low = context_bound_low
        self.context_bound_high = context_bound_high
        self.context_dim = context_dim

        self.context_dist = uniform.Uniform(context_bound_low, context_bound_high)

        self.mean_fn = mean_fn
        self.cov_fn = cov_fn

    def get_contexts_gmm(self, n_contexts):
        contexts = self.context_dist.sample((n_contexts, self.context_dim))  # return shape(n_contexts, 1)
        # print(contexts)
        return contexts

    def log_prob_tgt(self, contexts, samples):
        means = self.mean_fn(contexts).to(contexts.device)
        covs = self.cov_fn(contexts).to(contexts.device)
        n_contexts, n_components, _ = means.shape

        gate = ch.tensor(1.0 / n_components).to(contexts.device)
        if means.shape[0] == samples.shape[0]:
            # for easy use in plotting
            log = [MultivariateNormal(means[:, i], covs[:, i]).log_prob(samples) for i in range(n_components)]
            log = ch.stack(log, dim=1)  # batch_size, n_components
            log_sum = ch.logsumexp(ch.log(gate) + log, dim=1)  # batch_size
            return log_sum
        else:
            # log_c = []
            # for mean, cov in zip(means, covs):
            #     log = []
            #     for o in range(n_components):
            #         loc = mean[o]
            #         covariance_matrix = cov[o]
            #         log_component = MultivariateNormal(loc, covariance_matrix).log_prob(samples)
            #         log.append(ch.log(gate) + log_component)
            #     log = ch.stack(log, dim=0)
            #     # print(log.shape == ch.Size([n_components, sample_size]))
            #     log_c.append(ch.logsumexp(log, dim=0))
            #
            # return ch.stack(log_c, dim=1)  # (samples.shape[0], n_c)
            print("Not implemented yet")


def gaussian_target_simple_plot(target, contexts):
    x1_test = np.linspace(-25, 25, 100)
    x2_test = np.linspace(-25, 25, 100)
    x1, x2 = np.meshgrid(x1_test, x2_test)
    x1_flat = ch.from_numpy(x1.reshape(-1, 1).astype(np.float32)).detach()
    x2_flat = ch.from_numpy(x2.reshape(-1, 1).astype(np.float32)).detach()
    fig, axes = plt.subplots(1, len(contexts), figsize=(20, 10))
    for i, c in enumerate(contexts):
        # plot target distribution
        ax = axes[i]
        ax.clear()
        c_expanded = c.unsqueeze(0).expand(x1_flat.shape[0], -1)
        x = ch.cat((x1_flat, x2_flat), dim=1)
        log_probs = target.log_prob_tgt(c_expanded, x.to('cpu')).exp().view(x1.shape).detach() + 1e-6
        ax.contourf(x1, x2, log_probs, levels=100)
        ax.set_title(f'Target density with Context: {c}')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    plt.tight_layout()
    plt.show()


def get_cov_fns(n_components):
    def cat_cov(c):
        covs = []
        for i in range(n_components):
            chol = ch.stack([
                ch.stack([0.5 * ch.sin(c[:, 0]) + 1.1, ch.zeros_like(c[:, 0])], dim=1),
                ch.stack([ch.sin(3 * c[:, 0]) * ch.cos(3 * c[:, 0]), 0.5 * ch.cos(c[:, 0]) + 1.1], dim=1)], dim=1)
            cov = chol @ chol.permute(0, 2, 1)
            covs.append(cov)
        return ch.stack(covs, dim=1)
    return cat_cov


def get_mean_fns(n_components):
    def cat_mean(c):
        mean = []
        for i in range(n_components):
            sub_mean = ch.stack([15 * ch.sin(c[:, 0]), 15 * ch.cos(c[:, 0])], dim=1)
            mean.append(sub_mean)
        return ch.stack(mean, dim=1)
    return cat_mean


# test
# mean_fn = get_mean_fns(4)
# std_fn_sin = get_cov_fns(4)
#
# target = ConditionalGMMSimpleTarget(mean_fn, std_fn_sin)
# c = target.get_contexts_gmm(2)
# print(mean_fn(c))
# gaussian_target_simple_plot(target, c)