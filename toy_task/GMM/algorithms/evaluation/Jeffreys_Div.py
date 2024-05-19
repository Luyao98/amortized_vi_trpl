def jeffreys_divergence(model,
                        target,
                        eval_contexts,
                        device,
                        num_samples=1000):

    eval_gate, eval_mean, eval_chol = model(eval_contexts)
    model_samples = model.get_samples_gmm(eval_gate, eval_mean, eval_chol, num_samples).to(device)
    target_samples = target.sample(eval_contexts, num_samples).to(device)
    # TODO make here more general, now only for funnel with 3D
    if target_samples.shape[-1] == 3:
        target_samples = target_samples[..., 1:]
    t_log_t = target.log_prob_tgt(eval_contexts, target_samples)  # [batch_size, n_samples]
    t_log_m = target.log_prob_tgt(eval_contexts, model_samples)
    m_log_t = model.log_prob_gmm(eval_mean, eval_chol, eval_gate, target_samples)
    m_log_m = model.log_prob_gmm(eval_mean, eval_chol, eval_gate, model_samples)

    kl_target_model = (t_log_t - m_log_t).mean()
    kl_model_target = (m_log_m - t_log_m).mean()

    js_div = 0.5 * (kl_target_model + kl_model_target)

    return js_div


# version 2: this version doesn't need any mehtod from target and target, total from sampling
# import torch
# from torch.distributions import MultivariateNormal, Categorical
# from toy_task.GMM.targets.gaussian_mixture_target import get_weights
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
# def jeffreys_divergence_2(target, target, eval_contexts, device, num_samples=1000):
#     target_mean = target.mean_fn(eval_contexts).to(device)
#     target_chol = target.chol_fn(eval_contexts).to(device)
#     target_gate = get_weights(eval_contexts).to(device)
#     model_gate, model_mean, model_chol = target(eval_contexts)
#
#     p_weights, p_means, p_chols = target_gate, target_mean, target_chol
#     q_weights, q_means, q_chols = model_gate, model_mean, model_chol
#
#     p_samples = sample_from_gmm(p_weights, p_means, p_chols, num_samples).to(device)
#     q_samples = sample_from_gmm(q_weights, q_means, q_chols, num_samples).to(device)
#
#     # EUBO
#     kl_pq = kl_divergence_gmm(p_samples, p_weights, p_means, p_chols, q_weights, q_means, q_chols)
#     # ELBO
#     kl_qp = kl_divergence_gmm(q_samples, q_weights, q_means, q_chols, p_weights, p_means, p_chols)
#
#     j_div = 0.5 * (kl_pq + kl_qp)
#     return j_div
