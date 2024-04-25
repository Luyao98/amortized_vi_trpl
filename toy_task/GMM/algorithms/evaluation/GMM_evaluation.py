import torch
from torch.distributions import MultivariateNormal, Categorical


def kl_divergence_gmm(samples, p_weights, p_means, p_covs, q_weights, q_means, q_covs):
    log_p = compute_log_prob(p_means, p_covs, p_weights, samples)
    log_q = compute_log_prob(q_means, q_covs, q_weights, samples)
    return (log_p - log_q).mean()


def compute_log_prob(means, covs, weights, samples):
    n_samples = samples.shape[1]
    n_contexts, n_components, _ = means.shape
    means_expanded = means.unsqueeze(1).expand(-1, n_samples, -1, -1)
    covs_expanded = covs.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
    samples_expanded = samples.unsqueeze(2).expand(-1, -1, n_components, -1)

    mvn = MultivariateNormal(means_expanded, covariance_matrix=covs_expanded)
    log_probs = mvn.log_prob(samples_expanded)  # [batch_size, n_samples, n_components]

    gate_expanded = weights.unsqueeze(1).expand(-1, n_samples, -1)
    log_probs += gate_expanded

    log_probs = torch.logsumexp(log_probs, dim=2)  # [batch_size, n_samples]
    return log_probs


def sample_from_gmm(weights, means, covs, num_samples):
    samples = []
    for i in range(weights.shape[0]):
        cat = Categorical(weights[i])
        indices = cat.sample((num_samples,))
        chosen_means = means[i, indices]
        chosen_covs = covs[i, indices]
        normal = MultivariateNormal(chosen_means, covariance_matrix=chosen_covs)
        samples.append(normal.sample())
    return torch.stack(samples)  # [n_contexts, n_samples, n_features]


def js_divergence_gmm(p, q, num_samples=10000):
    """
    MC estimation of JS divergence between two GMMs
    p: target distribution
    q: model distribution
    """
    p_weights, p_means, p_covs = p
    q_weights, q_means, q_covs = q
    device = p_weights.device

    p_samples = sample_from_gmm(p_weights, p_means, p_covs, num_samples).to(device)
    q_samples = sample_from_gmm(q_weights, q_means, q_covs, num_samples).to(device)

    # EUBO
    kl_pq = kl_divergence_gmm(p_samples, p_weights, p_means, p_covs, q_weights, q_means, q_covs)
    # ELBO
    kl_qp = kl_divergence_gmm(q_samples, q_weights, q_means, q_covs, p_weights, p_means, p_covs)

    js_div = 0.5 * (kl_pq + kl_qp)
    return js_div
