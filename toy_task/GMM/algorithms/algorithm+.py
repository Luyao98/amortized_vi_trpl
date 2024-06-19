import torch as ch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, kl_divergence
import wandb

from toy_task.GMM.models.GMM_model import ConditionalGMM
from toy_task.GMM.models.GMM_model_2 import ConditionalGMM2
from toy_task.GMM.models.GMM_model_3 import ConditionalGMM3
from toy_task.GMM.targets.abstract_target import AbstractTarget
from toy_task.GMM.models.model_factory import get_model
from toy_task.GMM.targets.target_factory import get_target
from toy_task.GMM.algorithms.visualization.GMM_plot import plot2d_matplotlib
from toy_task.GMM.algorithms.evaluation.JensenShannon_Div import js_divergence, js_divergence_gate, kl_divergence_gate
from toy_task.GMM.algorithms.evaluation.Jeffreys_Div import jeffreys_divergence
from toy_task.GMM.projections.split_kl_projection import split_kl_projection
from toy_task.GMM.projections.split_w2_projection import split_w2_projection
# from toy_task.GMM.projections.gate_projecion import get_optimal_p_batch

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from toy_task.GMM.utils.network_utils import set_seed

def train_model_p(model: ConditionalGMM or ConditionalGMM2 or ConditionalGMM3,
                target: AbstractTarget,
                n_epochs: int,
                batch_size: int,
                n_context: int,
                n_components: int,
                n_samples: int,
                gate_lr: float,
                gaussian_lr: float,
                device,
                eps_gate: float,
                eps_mean: float,
                eps_cov: float ,
                alpha: int,
                proj_type="kl"):



    optimizer = optim.Adam([
        {'params': model.gate.parameters(), 'lr': gate_lr},
        {'params': model.gaussian_list.parameters(), 'lr': gaussian_lr}
    ], weight_decay=1e-5)
    contexts = target.get_contexts(n_context).to(device)
    eval_contexts = target.get_contexts(200).to(device)
    # bmm and gmm
    # plot_contexts = ch.tensor([[-0.3],
    #                            [0.7],
    #                            [-1.8]])
    # # funnel
    plot_contexts = ch.tensor([[-0.3],
                               [0.1],
                               [-0.8]])
    train_size = int(n_context)
    prev_loss = float('inf')

    for epoch in range(n_epochs):
        # plot initial model
        if epoch == 0:
            model.eval()
            with ch.no_grad():
                plot(model, target, proj_type=proj_type, contexts=plot_contexts)
                model.to(device)
            model.train()

        # shuffle sampled contexts, since the same sample set is used.
        indices = ch.randperm(train_size)
        shuffled_contexts = contexts[indices]

        batched_approx_reward = []
        init_contexts = shuffled_contexts[0:batch_size]
        b_gate_old, b_mean_old, b_chol_old = model(init_contexts)

        for batch_idx in range(0, train_size, batch_size):
            # get old distribution for current batch
            b_contexts = shuffled_contexts[batch_idx:batch_idx+batch_size]

            # prediction step
            gate_pred, mean_pred, chol_pred = model(b_contexts)

            # projection step
            mean_proj = []
            chol_proj = []
            for j in range(n_components):
                mean_proj_j, chol_proj_j = split_kl_projection(mean_pred[:, j] , chol_pred[:, j],
                                                               b_mean_old[:, j].clone().detach(),
                                                               b_chol_old[:, j].clone().detach(),
                                                               eps_mean, eps_cov)
                mean_proj.append(mean_proj_j)
                chol_proj.append(chol_proj_j)
            mean_proj = ch.stack(mean_proj, dim=1)
            chol_proj = ch.stack(chol_proj, dim=1)

            # gate proj
            p = cp.Variable(n_components, nonneg=True)
            q = cp.Parameter(n_components)
            r = cp.Parameter(n_components)
            epsilon = cp.Parameter(nonneg=True)

            objective = cp.Minimize(cp.sum(cp.kl_div(p, q)))
            constraints = [
                cp.sum(cp.kl_div(p, r)) <= epsilon,
                cp.sum(p) == 1
            ]
            problem = cp.Problem(objective, constraints)
            assert problem.is_dpp()

            cvxpylayer = CvxpyLayer(problem, parameters=[q, r, epsilon], variables=[p])

            # compute kl(gate_pred, gate_old), only project these whose kl > eps_gate
            kl_gates = ch.exp(gate_pred) * (gate_pred - b_gate_old)
            mask = kl_gates > eps_gate
            gate_proj = ch.zeros_like(gate_pred)
            gate_proj[~mask] = gate_pred[~mask]
            if mask.any():
                eps_gates = ch.tensor(eps_gate, dtype=ch.float32, device=device)
                exp_gate_proj = cvxpylayer(ch.exp(gate_pred), ch.exp(b_gate_old).clone().detach(), eps_gates)[0]
                gate_proj[mask] = ch.log(exp_gate_proj[mask])

            # component-wise calculation
            loss_component = []
            approx_reward_component = []
            for j in range(n_components):
                mean_pred_j = mean_pred[:, j]  # (batched_c, 2)
                chol_pred_j = chol_pred[:, j]

                mean_proj_j = mean_proj[:, j]
                chol_proj_j = chol_proj[:, j]

                # target and target log probability
                model_samples = model.get_rsamples(mean_proj_j, chol_proj_j, n_samples)
                log_model_j = model.log_prob(mean_proj_j, chol_proj_j, model_samples)
                log_target_j = target.log_prob_tgt(b_contexts, model_samples)

                # regression step
                pred_dist = MultivariateNormal(mean_pred_j, scale_tril=chol_pred_j)
                proj_dist = MultivariateNormal(mean_proj_j.clone().detach(), scale_tril=chol_proj_j.clone().detach())
                reg_loss = kl_divergence(proj_dist, pred_dist).unsqueeze(1).expand(-1, model_samples.shape[1])

                # do we need regression for gate?
                # reg_loss_gates = ch.exp(gate_proj[:, j].clone().detach()) * (gate_proj[:, j].clone().detach() - gate_pred[:, j])

                aux_loss = model.auxiliary_reward(j, gate_proj.clone().detach(), mean_proj.clone().detach(),
                                                  chol_proj.clone().detach(), model_samples)

                gate_proj_j = gate_proj[:, j].unsqueeze(1).expand(-1, model_samples.shape[1])
                approx_reward_j = log_model_j - log_target_j - aux_loss + alpha * reg_loss + gate_proj_j
                loss_j = ch.exp(gate_proj_j) * approx_reward_j

                loss_component.append(loss_j)
                approx_reward_component.append(approx_reward_j)
            if batch_idx + batch_size < train_size:
                b_next_context = shuffled_contexts[batch_idx + batch_size:batch_idx + 2 * batch_size]
                b_gate_old, b_mean_old, b_chol_old = model(b_next_context)

            batched_approx_reward.append(ch.stack(approx_reward_component))
            loss = ch.sum(ch.stack(loss_component))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            wandb.log({"train_loss": loss.item()})


        # Evaluation
        n_plot = n_epochs // 10
        if (epoch + 1) % n_plot == 0:
            model.eval()
            with ch.no_grad():
                approx_reward = ch.cat(batched_approx_reward, dim=1)  # [n_components, n_contexts, n_samples]
                js_divergence_gates = js_divergence_gate(approx_reward, model, shuffled_contexts, device)
                kl_gate = kl_divergence_gate(approx_reward, model, shuffled_contexts, device)
                js_div = js_divergence(model, target, eval_contexts, device, proj_type)
                j_div = jeffreys_divergence(model, target, eval_contexts, device, proj_type)

                plot(model, target, proj_type=proj_type, contexts=plot_contexts)
                model.to(device)

                # trick from VIPS++ paper
                current_loss = loss.item()
                if current_loss < prev_loss:
                    eps_mean *= 0.8
                    eps_cov *= 0.8
                else:
                    eps_mean *= 1.1
                    eps_cov *= 1.1
                prev_loss = current_loss

            model.train()
            wandb.log({"reverse KL between gates": kl_gate.item(),
                       "Jensen Shannon Divergence between gates": js_divergence_gates.item(),
                       "Jensen Shannon Divergence": js_div.item(),
                       "Jeffreys Divergence": j_div.item()})

    print("Training done!")


def plot(model: ConditionalGMM,
         target: AbstractTarget,
         proj_type,
         contexts=None):
    if contexts is None:
        contexts = target.get_contexts(3).to('cpu')
    else:
        contexts = contexts.clone().to('cpu')

    plot2d_matplotlib(target, model.to('cpu'), contexts, min_x=-6.5, max_x=6.5, min_y=-6.5, max_y=6.5, proj_type=proj_type)
    # plot2d_matplotlib(target, model.to('cpu'), contexts, min_x=-10, max_x=10, min_y=-10, max_y=10)


def toy_task(config):
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    n_context = config['n_context']
    n_components = config['n_components']
    num_gate_layer = config['num_gate_layer']
    num_component_layer = config['num_component_layer']
    n_samples = config['n_samples']
    gate_lr = config['gate_lr']
    gaussian_lr = config['gaussian_lr']

    model_name = config['model_name']
    dim = config['dim']
    initialization_type = config['initialization_type']

    target_name = config['target_name']
    target_components = config['target_components']

    eps_gate = config['eps_gate']
    eps_mean = config['eps_mean']
    eps_cov = config['eps_cov']
    alpha = config['alpha']

    # Device
    device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Target
    target = get_target(target_name, target_components=target_components).to(device)

    # Model
    model = get_model(model_name,
                      target_name,
                      dim,
                      device,
                      n_components,
                      num_gate_layer,
                      num_component_layer,
                      initialization_type)

    # Training
    train_model_p(model, target,
                n_epochs, batch_size, n_context, n_components, n_samples, gate_lr, gaussian_lr, device,
                eps_gate, eps_mean, eps_cov, alpha)


if __name__ == "__main__":
    # test
    set_seed(1001)
    config = {
        "n_epochs": 400,
        "batch_size": 128,
        "n_context": 1280,
        "n_components": 10,
        "num_gate_layer": 5,
        "num_component_layer": 7,
        "n_samples": 10,
        "gate_lr": 0.00001,
        "gaussian_lr": 0.00001,
        "model_name": "toy_task_model_2",
        "target_name": "funnel",
        "target_components": 10,
        "dim": 10,
        "initialization_type": "xavier",
        "eps_gate": 1e-6,
        "eps_mean": 1.0,
        "eps_cov": 0.5,
        "alpha": 50
    }
    group_name = "test"
    wandb.init(project="ELBO", group=group_name, config=config)
    toy_task(config)

