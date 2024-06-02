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
from toy_task.GMM.algorithms.evaluation.JensenShannon_Div import js_divergence, ideal_js_divergence
from toy_task.GMM.algorithms.evaluation.Jeffreys_Div import jeffreys_divergence
from toy_task.GMM.projections.split_kl_projection import split_projection


def train_model(model: ConditionalGMM or ConditionalGMM2 or ConditionalGMM3,
                target: AbstractTarget,
                n_epochs: int,
                batch_size: int,
                n_context: int,
                n_components: int,
                n_samples: int,
                gate_lr: float,
                gaussian_lr: float,
                device,
                project,
                eps_mean: float or None,
                eps_cov: float or None,
                alpha: int or None,
                responsibility=True):

    # optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-5)
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
    # funnel
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
                plot(model, target, plot_contexts)
                model.to(device)
            model.train()

        # shuffle sampled contexts, since the same sample set is used.
        indices = ch.randperm(train_size)
        shuffled_contexts = contexts[indices]

        # get old distribution for projection
        gate_old, mean_old, chol_old = model(shuffled_contexts)
        gate_old = gate_old.clone().detach()
        mean_old = mean_old.clone().detach()
        chol_old = chol_old.clone().detach()

        batched_approx_reward = []
        for batch_idx in range(0, train_size, batch_size):
            # get old distribution for current batch
            b_contexts = shuffled_contexts[batch_idx:batch_idx+batch_size]
            b_gate_old = gate_old[batch_idx:batch_idx+batch_size]
            b_mean_old = mean_old[batch_idx:batch_idx+batch_size]
            b_chol_old = chol_old[batch_idx:batch_idx+batch_size]

            # prediction step
            gate_pred, mean_pred, chol_pred = model(b_contexts)

            # component-wise calculation
            loss_component = []
            approx_reward_component = []
            for j in range(n_components):
                mean_pred_j = mean_pred[:, j]  # (batched_c, 2)
                chol_pred_j = chol_pred[:, j]
                mean_old_j = b_mean_old[:, j]
                chol_old_j = b_chol_old[:, j]

                if project:
                    mean_proj_j, chol_proj_j = split_projection(mean_pred_j, chol_pred_j, mean_old_j, chol_old_j,
                                                                eps_mean, eps_cov)

                    # target and target log probability
                    model_samples = model.get_rsamples(mean_proj_j, chol_proj_j, n_samples)  # shape (n_c, n_samples, 2)
                    log_model_j = model.log_prob(mean_proj_j, chol_proj_j, model_samples)
                    log_target_j = target.log_prob_tgt(b_contexts, model_samples)

                    # regression step
                    pred_dist = MultivariateNormal(mean_pred_j, scale_tril=chol_pred_j)
                    proj_dist = MultivariateNormal(mean_proj_j, scale_tril=chol_proj_j)
                    reg_loss = kl_divergence(pred_dist, proj_dist).unsqueeze(1).expand(-1, model_samples.shape[1])

                    aux_loss = model.auxiliary_reward(j, gate_pred.detach(), mean_pred.detach(), chol_pred.detach(),
                                                      model_samples)
                    gate_pred_j = gate_pred[:, j].unsqueeze(1).expand(-1, model_samples.shape[1])
                    approx_reward_j = log_model_j - log_target_j - aux_loss + alpha * reg_loss + gate_pred_j
                    loss_j = ch.exp(gate_pred_j) * approx_reward_j
                else:
                    model_samples = model.get_rsamples(mean_pred_j, chol_pred_j, n_samples)
                    log_model_j = model.log_prob(mean_pred_j, chol_pred_j, model_samples)
                    log_target_j = target.log_prob_tgt(b_contexts, model_samples)
                    if responsibility:
                        # loss: with log responsibility but without projection
                        aux_loss = model.auxiliary_reward(j, gate_pred.detach(), mean_pred.detach(), chol_pred.detach(),
                                                          model_samples)
                        gate_pred_j = gate_pred[:, j].unsqueeze(1).expand(-1, model_samples.shape[1])
                        approx_reward_j = log_model_j - log_target_j - aux_loss + gate_pred_j
                        loss_j = ch.exp(gate_pred_j) * approx_reward_j
                    else:
                        # loss: without log responsibility or projection
                        loss_j = log_model_j - log_target_j
                        approx_reward_j = loss_j
                loss_component.append(loss_j)
                approx_reward_component.append(approx_reward_j)

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
                ideal_js_div = ideal_js_divergence(model, approx_reward, shuffled_contexts, device)
                js_div = js_divergence(model, target, eval_contexts, device)
                j_div = jeffreys_divergence(model, target, eval_contexts, device)

                plot(model, target, plot_contexts)
                model.to(device)

                # trick from VIPS++ paper
                if project:
                    current_loss = loss.item()
                    if current_loss < prev_loss:
                        eps_mean *= 0.8
                        eps_cov *= 0.8
                    else:
                        eps_mean *= 1.1
                        eps_cov *= 1.1
                    prev_loss = current_loss
            model.train()
            wandb.log({"ideal Jensen Shannon Divergence": ideal_js_div.item(),
                       "Jensen Shannon Divergence": js_div.item(),
                       "Jeffreys Divergence": j_div.item()})

    print("Training done!")


def plot(model: ConditionalGMM,
         target: AbstractTarget,
         contexts=None):
    if contexts is None:
        contexts = target.get_contexts(3).to('cpu')
    else:
        contexts = contexts.clone().to('cpu')

    plot2d_matplotlib(target, model.to('cpu'), contexts, min_x=-6.5, max_x=6.5, min_y=-6.5, max_y=6.5)
    # plot2d_matplotlib(target, model.to('cpu'), contexts, min_x=-10, max_x=10, min_y=-10, max_y=10)


def toy_task(config):
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    n_context = config['n_context']
    n_components = config['n_components']
    n_samples = config['n_samples']
    gate_lr = config['gate_lr']
    gaussian_lr = config['gaussian_lr']

    model_name = config['model_name']
    dim = config['dim']
    initialization_type = config['initialization_type']

    target_name = config['target_name']
    target_components = config['target_components']

    project = config['project']
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
                      initialization_type)

    # Training
    train_model(model, target,
                n_epochs, batch_size, n_context, n_components, n_samples, gate_lr, gaussian_lr, device,
                project, eps_mean, eps_cov, alpha)


# # test
# config = {
#     "n_epochs": 400,
#     "batch_size": 64,
#     "n_context": 640,
#     "n_components": 4,
#     "n_samples": 10,
#     "gate_lr": 0.01,
#     "gaussian_lr": 0.01,
#     "model_name": "toy_task_model_3",
#     "target_name": "gmm",
#     "target_components": "4",
#     "dim": 2,
#     "initialization_type": "xavier",
#     "project": False,
#     "eps_mean": 0.5,
#     "eps_cov": 0.1,
#     "alpha": 2
# }
#
# toy_task(config)
