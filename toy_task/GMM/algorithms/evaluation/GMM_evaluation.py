import torch
from torch.distributions import MultivariateNormal, Categorical


# def kl_divergence_gmm(samples, p_weights, p_means, p_chols, q_weights, q_means, q_chols):
#     log_p = compute_log_prob(p_means, p_chols, p_weights, samples)
#     log_q = compute_log_prob(q_means, q_chols, q_weights, samples)
#     return (log_p - log_q).mean()
#
#
# def compute_log_prob(means, chols, weights, samples):
#     n_samples = samples.shape[1]
#     n_contexts, n_components, _ = means.shape
#     means_expanded = means.unsqueeze(1).expand(-1, n_samples, -1, -1)
#     chols_expanded = chols.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
#     samples_expanded = samples.unsqueeze(2).expand(-1, -1, n_components, -1)
#
#     mvn = MultivariateNormal(means_expanded, scale_tril=chols_expanded)
#     log_probs = mvn.log_prob(samples_expanded)  # [batch_size, n_samples, n_components]
#
#     gate_expanded = weights.unsqueeze(1).expand(-1, n_samples, -1)
#     log_probs += gate_expanded
#
#     log_probs = torch.logsumexp(log_probs, dim=2)  # [batch_size, n_samples]
#     return log_probs
#
#
# def sample_from_gmm(weights, means, chols, num_samples):
#     samples = []
#     for i in range(weights.shape[0]):
#         cat = Categorical(weights[i])
#         indices = cat.sample((num_samples,))
#         chosen_means = means[i, indices]
#         chosen_chols = chols[i, indices]
#         normal = MultivariateNormal(chosen_means, scale_tril=chosen_chols)
#         samples.append(normal.sample())
#     return torch.stack(samples)  # [n_contexts, n_samples, n_features]
#
#
# def js_divergence_gmm(p, q, num_samples=10000):
#     """
#     MC estimation of JS divergence between two GMMs
#     p: target distribution
#     q: model distribution
#     """
#     p_weights, p_means, p_chols = p
#     q_weights, q_means, q_chols = q
#     device = p_weights.device
#
#     p_samples = sample_from_gmm(p_weights, p_means, p_chols, num_samples).to(device)
#     q_samples = sample_from_gmm(q_weights, q_means, q_chols, num_samples).to(device)
#
#     # EUBO
#     kl_pq = kl_divergence_gmm(p_samples, p_weights, p_means, p_chols, q_weights, q_means, q_chols)
#     # ELBO
#     kl_qp = kl_divergence_gmm(q_samples, q_weights, q_means, q_chols, p_weights, p_means, p_chols)
#
#     js_div = 0.5 * (kl_pq + kl_qp)
#     return js_div


def log_prob_plot(means, chols, gates, samples):
    """
    gates are log of gate
    """
    if means.dim() == 2:
        # for evaluation
        n_components = means.shape[0]
        log = [gates[o] + MultivariateNormal(means[o], scale_tril=chols[o]).log_prob(samples) for o in range(n_components)]
        log = torch.stack(log, dim=0)  # (n_components, sample_size)
        log_c = torch.logsumexp(log, dim=0)  # (sample_size)
        return log_c
    else:
        print("Not implemented")


def kl_divergence_gmm(samples, p_weights, p_means, p_chols, q_weights, q_means, q_chols):
    """ MC estimation of KL divergence between two GMMs """
    log_p = log_prob_plot(p_means, p_chols, p_weights, samples)
    log_q = log_prob_plot(q_means, q_chols, q_weights, samples)
    return (log_p - log_q).mean()


def sample_from_gmm(weights, means, chols, num_samples):
    cat = Categorical(weights)
    indices = cat.sample((num_samples,))
    chosen_means = means[indices]
    chosen_chols = chols[indices]
    normal = MultivariateNormal(chosen_means, scale_tril=chosen_chols)
    return normal.sample()


def js_divergence_gmm(p, q, num_samples=10000):
    """
    MC estimation of JS divergence between two GMMs
    p: target distribution
    q: model distribution
    """
    batched_p_weights, batched_p_means, batched_p_chols = p
    batched_q_weights, batched_q_means, batched_q_chols = q
    device = batched_p_weights.device
    n_context = batched_p_weights.shape[0]
    js = []
    for i in range(n_context):
        p_weights = batched_p_weights[i]
        p_means = batched_p_means[i]
        p_chols = batched_p_chols[i]
        q_weights = batched_q_weights[i]
        q_means = batched_q_means[i]
        q_chols = batched_q_chols[i]

        p_samples = sample_from_gmm(p_weights, p_means, p_chols, num_samples).to(device)
        q_samples = sample_from_gmm(q_weights, q_means, q_chols, num_samples).to(device)

        # EUBO
        kl_pq = kl_divergence_gmm(p_samples, p_weights, p_means, p_chols, q_weights, q_means, q_chols)
        # ELBO
        kl_qp = kl_divergence_gmm(q_samples, q_weights, q_means, q_chols, p_weights, p_means, p_chols)

        js_div = 0.5 * (kl_pq + kl_qp)
        js.append(js_div)
    js_div = torch.stack(js).mean()
    return js_div
