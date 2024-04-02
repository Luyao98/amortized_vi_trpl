import torch as ch
from torch.distributions import uniform, MultivariateNormal

# ch.manual_seed(37)
# np.random.seed(37)


class ConditionalGMMTarget(ch.nn.Module):
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

    # def get_samples_tgt(self, contexts, n_samples_per_component=100):
    #     means = self.mean_fn(contexts)  # shape: (n_contexts, n_components, 2)
    #     covs = self.cov_fn(contexts)    # shape: (n_contexts, n_components, 2, 2)
    #     n_contexts, n_components, _ = means.shape
    #     samples_list = []
    #
    #     for i in range(n_contexts):
    #         context_samples = []
    #         for j in range(n_components):
    #             mean = means[i, j]
    #             cov = covs[i, j]
    #             dist = ch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    #             samples = dist.sample((n_samples_per_component,))  # shape: (n_samples_per_component, 2)
    #             context_samples.append(samples)
    #         context_samples = ch.cat(context_samples, dim=0)  # shape: (n_components * n_samples_per_component, 2)
    #         samples_list.append(context_samples)
    #
    #     samples = ch.stack(samples_list, dim=0)  # (n_contexts, n_components * n_samples_per_component, sample_dim)
    #     return samples  # shape (n_contexts, n_samples, sample_dim)

    # def log_prob_tgt(self, contexts, samples):
    #     means = self.mean_fn(contexts)
    #     covs = self.cov_fn(contexts)
    #     n_contexts, n_components, _ = means.shape
    #     gate = ch.tensor(1.0 / n_components)
    #     log_probs = []
    #
    #     for c in range(n_contexts):
    #         log_prob_components = []
    #         for i in range(n_components):
    #             loc = means[c, i, :]
    #             covariance_matrix = covs[c, i, :, :]
    #             mvn = MultivariateNormal(loc, covariance_matrix)
    #             log_prob_component = mvn.log_prob(samples)
    #             log_prob_components.append(log_prob_component + ch.log(gate))
    #
    #         # Use LogSumExp for numerical stability when summing log probabilities
    #         # log_prob_context = ch.logsumexp(ch.stack(log_prob_components), dim=0)
    #         log_prob_context = ch.stack(log_prob_components)
    #         log_probs.append(log_prob_context)
    #     a = ch.stack(log_probs, dim=1)
    #     print("a shape", a.shape)
    #
    #     return a  # Stack to get shape [n_samples, n_contexts]
    def log_prob_tgt(self, contexts, samples):
        means = self.mean_fn(contexts)
        covs = self.cov_fn(contexts)
        n_contexts, n_components, _ = means.shape
        gate = ch.tensor(1.0 / n_components)
        log_c = []
        for c in range(n_contexts):
            log = ch.zeros(samples.shape[0])
            for i in range(n_components):
                loc = means[c, i,]
                covariance_matrix = covs[c, i,]
                sub_log = MultivariateNormal(loc, covariance_matrix).log_prob(samples)
                log = log + ch.log(gate) + sub_log
            log_c.append(log)
        return ch.stack(log_c, dim=1)  # (samples.shape[0], n_c)


def get_cov_fn(n_components):
    def cat_cov(c):
        covs = []
        for i in range(n_components):
            sub1_chol = ch.cat([i + ch.sin(c) + 3, ch.zeros_like(c)], dim=1)
            sub2_chol = ch.cat([10 * ch.sin(c) * ch.cos(c), i + ch.cos(c) + 3], dim=1)
            chol = ch.stack((sub1_chol, sub2_chol), dim=1)
            cov = chol @ chol.permute(0, 2, 1)
            covs.append(cov)
        return ch.stack(covs, dim=1)
    return cat_cov


def get_mean_fn(n_components):
    def cat_mean(c):
        mean = []
        for i in range(n_components):
            sub_mean = ch.cat([10 * (i + ch.sin(c)), 10 * (i + ch.cos(c))], dim=1)
            mean.append(sub_mean)
        return ch.stack(mean, dim=1)
    return cat_mean


# if __name__ == '__main__':
#     mean_fn = get_mean_fn(3)
#     std_fn_sin = get_cov_fn(3)
#
#     target = ConditionalGMMTarget(mean_fn, std_fn_sin)
#     c = target.get_contexts(2)
#     print(c)
#     target.visualize(c)
