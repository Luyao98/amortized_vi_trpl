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
        return contexts

    def log_prob_tgt(self, contexts, samples):
        means = self.mean_fn(contexts).to(contexts.device)
        covs = self.cov_fn(contexts).to(contexts.device)
        batch_size, n_components, n_features = means.shape

        # uniform gating
        # gate = ch.tensor(1.0 / n_components).to(contexts.device)
        # gate = ch.log(gate)

        # non-uniform but fixed gating
        # values = [0.5, 0.1, 0.3, 0.1]
        # gate = ch.stack([ch.full((n_contexts,), value) for value in values], dim=1).to(contexts.device)
        # gate = ch.log(gate)

        # non-uniform but random gating
        gate = get_weights(contexts)

        if samples.dim() == 3:
            n_samples = samples.shape[1]
        else:
            # for plotting
            n_samples = samples.shape[0]
            samples = samples.unsqueeze(0).expand(batch_size, -1, -1)

        means_expanded = means.unsqueeze(1).expand(-1, n_samples, -1, -1)
        covs_expanded = covs.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
        samples_expanded = samples.unsqueeze(2).expand(-1, -1, n_components, -1)

        mvn = MultivariateNormal(means_expanded, covariance_matrix=covs_expanded)
        log_probs = mvn.log_prob(samples_expanded)  # [batch_size, n_samples, n_components]

        gate_expanded = gate.unsqueeze(1).expand(-1, n_samples, -1)
        log_probs += gate_expanded

        log_probs = ch.logsumexp(log_probs, dim=2)  # [batch_size, n_samples]
        return log_probs


def get_weights(c):
    """
    only for 4 components
    """
    weights = [ch.sin(c[:, 0]), ch.cos(c[:, 0]), ch.sin(10 * c[:, 0]), ch.cos(10 * c[:, 0])]
    weights = ch.stack(weights, dim=1)
    log_weights = ch.log_softmax(weights, dim=1)
    return log_weights


def get_cov_fn(n_components):
    def cat_cov(c):
        covs = []
        for i in range(n_components):
            chol = ch.stack([
                ch.stack([0.5 * ch.sin((i + 1) * c[:, 0]) + 1.1, ch.zeros_like(c[:, 0])], dim=1),
                ch.stack([ch.sin(3 * c[:, 0]) * ch.cos(3 * c[:, 0]), 0.5 * ch.cos((i + 1) * c[:, 0]) + 1.1], dim=1)], dim=1)
            cov = chol @ chol.permute(0, 2, 1)
            covs.append(cov)
        return ch.stack(covs, dim=1)
    return cat_cov


def get_mean_fn(n_components):
    def cat_mean(c):
        # mean = []
        # for i in range(n_components):
        #     sub_mean = ch.stack([10 * ch.sin((i + 1) * c[:, 0]), 10 * ch.cos((i + 1) * c[:, 0])], dim=1)
        #     mean.append(sub_mean)
        # mean1 = ch.stack([10 * ch.abs(ch.sin(c[:, 0])), 10 * ch.abs(ch.cos(c[:, 0]))], dim=1)
        # mean2 = ch.stack([-10 * ch.abs(ch.sin(2 * c[:, 0])), -10 * ch.abs(ch.cos(2 * c[:, 0]))], dim=1)
        # mean3 = ch.stack([10 * ch.abs(ch.sin(6 * c[:, 0])), -10 * ch.abs(ch.cos(6 * c[:, 0]))], dim=1)
        # mean4 = ch.stack([-10 * ch.abs(ch.sin(4 * c[:, 0])), 10 * ch.abs(ch.cos(4 * c[:, 0]))], dim=1)
        mean1 = ch.stack([2 + c[:, 0], 7 + c[:, 0]], dim=1)
        mean2 = ch.stack([-9 + c[:, 0], -2 + c[:, 0]], dim=1)
        mean3 = ch.stack([10 + c[:, 0], -5 + c[:, 0]], dim=1)
        mean4 = ch.stack([-3 + c[:, 0], 3 + c[:, 0]], dim=1)
        mean = [mean1, mean2, mean3, mean4]
        return ch.stack(mean, dim=1)
    return cat_mean


def get_gmm_target(n_components):
    mean_target = get_mean_fn(n_components)
    cov_target = get_cov_fn(n_components)
    target = ConditionalGMMTarget(mean_target, cov_target)
    return target
