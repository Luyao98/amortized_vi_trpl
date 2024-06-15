import torch as ch
from toy_task.GMM.targets.abstract_target import AbstractTarget
from toy_task.GMM.models.GMM_model import ConditionalGMM
from toy_task.GMM.models.GMM_model_2 import ConditionalGMM2
from toy_task.GMM.models.GMM_model_3 import ConditionalGMM3

def ideal_calculated_gates(stack_loss_component):
    eva_loss_component = stack_loss_component.clone().mean(dim=-1)
    eva_sum_loss = ch.logsumexp(eva_loss_component, dim=0)  # [n_contexts]
    log_ideal_gates = eva_loss_component - eva_sum_loss
    return log_ideal_gates.transpose(0, 1)


def js_divergence_gate(stack_loss_component, model, eval_contexts, device):
    ideal_gates = ideal_calculated_gates(stack_loss_component).to(device)
    eval_gate, _, _ = model(eval_contexts)
    mid_point = ch.logsumexp(ch.stack([ideal_gates, eval_gate]), dim=0) - ch.log(ch.tensor(2.0))
    ideal_log_mid = ch.exp(ideal_gates) * (ideal_gates - mid_point)
    pred_log_mid = ch.exp(eval_gate) * (eval_gate - mid_point)
    js_div = 0.5 * (ideal_log_mid + pred_log_mid)
    return js_div.mean()

def kl_divergence_gate(stack_loss_component, model, eval_contexts, device):
    """
    KL(model_gate || ground_truth_gate)
    """
    ideal_gates = ideal_calculated_gates(stack_loss_component).to(device)
    eval_gate, _, _ = model(eval_contexts)
    kl_div = ch.exp(eval_gate) * (eval_gate - ideal_gates)
    return kl_div.mean()


def js_divergence(model: ConditionalGMM or ConditionalGMM2 or ConditionalGMM3,
                  target: AbstractTarget,
                  eval_contexts,
                  device,
                  proj_type=None,
                  num_samples=1000):
    """
    MC estimation of Jenson's Shannon Divergence between two random, unknown distributions,
    the derivation see my note: https://www.notion.so/KLD-JSD-17c7d900e3e64e2ea04776e04fdda44c?pvs=4

    """

    eval_gate, eval_mean, eval_chol = model(eval_contexts)
    target_samples = target.sample(eval_contexts, num_samples).to(device)

    if proj_type == "w2":
        eval_sqrt = eval_chol @ eval_chol.transpose(-1, -2)
        eval_cov = eval_sqrt @ eval_sqrt
        model_samples = model.get_samples_gmm_w2(eval_gate, eval_mean, eval_cov, num_samples).to(device)

        t_log_t = target.log_prob_tgt(eval_contexts, target_samples)  # [batch_size, n_samples]
        t_log_m = target.log_prob_tgt(eval_contexts, model_samples)
        m_log_t = model.log_prob_gmm_w2(eval_mean, eval_cov, eval_gate, target_samples)
        m_log_m = model.log_prob_gmm_w2(eval_mean, eval_cov, eval_gate, model_samples)
    else:
        model_samples = model.get_samples_gmm(eval_gate, eval_mean, eval_chol, num_samples).to(device)

        t_log_t = target.log_prob_tgt(eval_contexts, target_samples)  # [batch_size, n_samples]
        t_log_m = target.log_prob_tgt(eval_contexts, model_samples)
        m_log_t = model.log_prob_gmm(eval_mean, eval_chol, eval_gate, target_samples)
        m_log_m = model.log_prob_gmm(eval_mean, eval_chol, eval_gate, model_samples)

    midpoint_t = ch.logsumexp(ch.stack([t_log_t, m_log_t]), dim=0) - ch.log(ch.tensor(2.0))
    midpoint_m = ch.logsumexp(ch.stack([t_log_m, m_log_m]), dim=0) - ch.log(ch.tensor(2.0))

    kl_target_midpoint = t_log_t - midpoint_t
    kl_model_midpoint = m_log_m - midpoint_m

    js_div = 0.5 * (kl_target_midpoint + kl_model_midpoint)

    return js_div.mean()


def ideal_js_divergence(model: ConditionalGMM or ConditionalGMM2,
                        approx_reward,
                        eval_contexts,
                        device,
                        proj_type=None,
                        num_samples=1000):

    ideal_gates = ideal_calculated_gates(approx_reward).to(device)
    eval_gate, eval_mean, eval_chol = model(eval_contexts)
    ideal_samples = model.get_samples_gmm(ideal_gates, eval_mean, eval_chol, num_samples).to(device)
    if proj_type == "w2":
        eval_sqrt = eval_chol @ eval_chol.transpose(-1, -2)
        eval_cov = eval_sqrt @ eval_sqrt
        model_samples = model.get_samples_gmm_w2(eval_gate, eval_mean, eval_cov, num_samples).to(device)

        m_log_i = model.log_prob_gmm_w2(eval_mean, eval_cov, eval_gate, ideal_samples)
        m_log_m = model.log_prob_gmm_w2(eval_mean, eval_cov, eval_gate, model_samples)
        i_log_i = model.log_prob_gmm_w2(eval_mean, eval_cov, ideal_gates, ideal_samples)
        i_log_m = model.log_prob_gmm_w2(eval_mean, eval_cov, ideal_gates, model_samples)
    else:
        model_samples = model.get_samples_gmm(eval_gate, eval_mean, eval_chol, num_samples).to(device)

        m_log_i = model.log_prob_gmm(eval_mean, eval_chol, eval_gate, ideal_samples)
        m_log_m = model.log_prob_gmm(eval_mean, eval_chol, eval_gate, model_samples)
        i_log_i = model.log_prob_gmm(eval_mean, eval_chol, ideal_gates, ideal_samples)
        i_log_m = model.log_prob_gmm(eval_mean, eval_chol, ideal_gates, model_samples)

    midpoint_i = ch.logsumexp(ch.stack([i_log_i, m_log_i]), dim=0) - ch.log(ch.tensor(2.0))
    midpoint_m = ch.logsumexp(ch.stack([i_log_m, m_log_m]), dim=0) - ch.log(ch.tensor(2.0))

    kl_ideal_midpoint = (i_log_i - midpoint_i).mean()
    kl_model_midpoint = (m_log_m - midpoint_m).mean()

    js_div = 0.5 * (kl_ideal_midpoint + kl_model_midpoint)

    return js_div
