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

    # # Filter values
    # valid_mask = (t_log_t > -50) & (t_log_t < 50) & \
    #              (t_log_m > -50) & (t_log_m < 50) & \
    #              (m_log_t > -50) & (m_log_t < 50) & \
    #              (m_log_m > -50) & (m_log_m < 50)
    #
    # valid_rows_mask = ch.all(valid_mask, dim=1)
    #
    # # Filter data using the mask
    # t_log_t = t_log_t[valid_rows_mask]
    # t_log_m = t_log_m[valid_rows_mask]
    # m_log_t = m_log_t[valid_rows_mask]
    # m_log_m = m_log_m[valid_rows_mask]

    midpoint_t = ch.logsumexp(ch.stack([t_log_t, m_log_t]), dim=0) - ch.log(ch.tensor(2.0))
    midpoint_m = ch.logsumexp(ch.stack([t_log_m, m_log_m]), dim=0) - ch.log(ch.tensor(2.0))

    # a = ch.exp(t_log_t)
    # b = ch.exp(m_log_t)
    # c = ch.exp(t_log_m)
    # d = ch.exp(m_log_m)
    # midpoint_t = ch.log(a + b +1e-8) - ch.log(ch.tensor(2.0))
    # midpoint_m = ch.log(c + d+1e-8) - ch.log(ch.tensor(2.0))

    # inf_mask = ch.isinf(midpoint_m)
    # if inf_mask.any():
    #     inf_indices = ch.nonzero(inf_mask, as_tuple=True)
    #     print("Infinities found at positions:", inf_indices)
    #     print("Corresponding contexts:")
    #     for idx in inf_indices[0].unique():
    #         print("Context index:", idx.item(), "Value:", eval_contexts[idx].cpu().numpy())
    #         print("c:", c[idx, inf_indices[1][inf_indices[0] == idx]].cpu().numpy())
    #         print("d:", d[idx, inf_indices[1][inf_indices[0] == idx]].cpu().numpy())
    #         print("t_log_m(to calculate c)", t_log_m[idx, inf_indices[1][inf_indices[0] == idx]].cpu().numpy())
    #         print("m_log_m(to calculate d)", m_log_m[idx, inf_indices[1][inf_indices[0] == idx]].cpu().numpy())
    #         print("Target samples causing infinities:",
    #               target_samples[idx, inf_indices[1][inf_indices[0] == idx]].cpu().numpy())
    #         print("Model samples causing infinities:",
    #               model_samples[idx, inf_indices[1][inf_indices[0] == idx]].cpu().numpy())

    kl_target_midpoint = (t_log_t - midpoint_t).mean()
    kl_model_midpoint = (m_log_m - midpoint_m).mean()
    # if kl_target_midpoint < 0:

    js_div = 0.5 * (kl_target_midpoint + kl_model_midpoint)

    return js_div
