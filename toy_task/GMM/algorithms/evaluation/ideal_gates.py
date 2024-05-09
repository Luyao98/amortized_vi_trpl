import torch as ch


def ideal_calculated_gates(stack_loss_component):
    eva_loss_component = stack_loss_component.clone().mean(dim=-1)
    eva_sum_loss = ch.logsumexp(eva_loss_component, dim=0)  # [n_contexts]
    log_ideal_gates = eva_loss_component - eva_sum_loss
    gates = ch.exp(log_ideal_gates.transpose(0, 1))
    return gates
