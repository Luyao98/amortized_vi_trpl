import torch as ch
from toy_task.GMM.models.GMM_model import ConditionalGMM
from toy_task.GMM.targets.banana_mixture_target import BananaMixtureTarget
from toy_task.GMM.targets.gaussian_mixture_target import ConditionalGMMTarget


def js_divergence(model,
                  target,
                  eval_contexts,
                  device,
                  num_samples=1000):
    """
    MC estimation of Jenson's Shannon Divergence between two random, unknown distributions,
    the derivation see my note: https://www.notion.so/KLD-JSD-17c7d900e3e64e2ea04776e04fdda44c?pvs=4

    """

    eval_gate, eval_mean, eval_chol = model(eval_contexts)
    model_samples = model.get_samples_gmm(eval_gate, eval_mean, eval_chol, num_samples).to(device)
    target_samples = target.sample(eval_contexts, num_samples).to(device)

    t_log_t = target.log_prob_tgt(eval_contexts, target_samples)  # [batch_size, n_samples]
    t_log_m = target.log_prob_tgt(eval_contexts, model_samples)
    m_log_t = model.log_prob_gmm(eval_mean, eval_chol, eval_gate, target_samples)
    m_log_m = model.log_prob_gmm(eval_mean, eval_chol, eval_gate, model_samples)

    midpoint_t = ch.logsumexp(ch.stack([t_log_t, m_log_t]), dim=0) - ch.log(ch.tensor(2.0))
    midpoint_m = ch.logsumexp(ch.stack([t_log_m, m_log_m]), dim=0) - ch.log(ch.tensor(2.0))

    kl_target_midpoint = (t_log_t - midpoint_t).mean()
    kl_model_midpoint = (m_log_m - midpoint_m).mean()


    js_div = 0.5 * (kl_target_midpoint + kl_model_midpoint)

    return js_div
