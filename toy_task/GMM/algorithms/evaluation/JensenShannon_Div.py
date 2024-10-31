import torch as ch
from toy_task.GMM.targets.abstract_target import AbstractTarget


def js_divergence(model,
                  target: AbstractTarget,
                  eval_contexts,
                  device,
                  num_samples=1000):
    """
    Estimates the Jensen-Shannon Divergence (JSD) and Jeffreys Divergence between two distributions
    using Monte Carlo (MC) sampling. This function computes these divergence metrics by sampling
    from a target distribution and a model distribution, providing a measure of similarity between
    the two distributions over given evaluation contexts.

    Jensen-Shannon Divergence (JSD) and Jeffreys Divergence:
        - **Jensen-Shannon Divergence** is a symmetric, smooth version of the Kullback-Leibler (KL) divergence.
          It is calculated as:

          JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)

          where M = 0.5 * (P + Q) is the midpoint distribution. The JSD is often preferred over the
          KL divergence when comparing two distributions, as it is bounded and symmetric.

        - **Jeffreys Divergence** is an alternative symmetric divergence defined as the average of the
          KL divergence in both directions:

          Jeffreys(P || Q) = 0.5 * (KL(P || Q) + KL(Q || P))

          This measure highlights the directional dissimilarity between two distributions.

    Parameters:
        model (callable): A model object with `eval_contexts` as input, returning `eval_gate`, `eval_mean`,
            and `eval_chol`. It should also provide `log_prob_gmm` and `get_samples_gmm` methods for generating
            samples and calculating log-probabilities. Expected to follow a Gaussian Mixture Model (GMM) structure.
        target (AbstractTarget): Target distribution object with `sample` and `log_prob_tgt` methods.
        eval_contexts (array-like): Contexts at which the distributions will be evaluated.
        device (str): "cpu" or "cuda".
        num_samples (int, optional): Number of samples to draw for Monte Carlo approximation. Defaults to 1000.

    Returns:
        torch.Tensor: Mean estimate of the Jensen-Shannon Divergence.
        torch.Tensor: Mean estimate of the Jeffreys Divergence.

    """

    eval_gate, eval_mean, eval_chol = model(eval_contexts)
    target_samples = target.sample(eval_contexts, num_samples)  # (n_samples, n_contexts, dz)
    model_samples = model.get_samples_gmm(eval_gate, eval_mean, eval_chol, num_samples)

    t_log_t = target.log_prob_tgt(eval_contexts, target_samples)  # (n_samples, n_contexts)
    t_log_m = target.log_prob_tgt(eval_contexts, model_samples)
    m_log_t = model.log_prob_gmm(eval_mean, eval_chol, eval_gate, target_samples)
    m_log_m = model.log_prob_gmm(eval_mean, eval_chol, eval_gate, model_samples)

    midpoint_t = ch.logsumexp(ch.stack([t_log_t, m_log_t]), dim=0) - ch.log(ch.tensor(2.0))
    midpoint_m = ch.logsumexp(ch.stack([t_log_m, m_log_m]), dim=0) - ch.log(ch.tensor(2.0))

    kl_target_midpoint = t_log_t - midpoint_t
    kl_model_midpoint = m_log_m - midpoint_m
    js_div = 0.5 * (kl_target_midpoint + kl_model_midpoint)

    kl_target_model = t_log_t - m_log_t
    kl_model_target = m_log_m - t_log_m
    jeffreys_div = 0.5 * (kl_target_model + kl_model_target)
    return js_div.mean(), jeffreys_div.mean()
