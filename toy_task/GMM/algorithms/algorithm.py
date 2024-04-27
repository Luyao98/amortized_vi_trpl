import torch as ch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, kl_divergence
import wandb

from toy_task.GMM.models.GMM_model import ConditionalGMM
from toy_task.GMM.models.model_factory import get_model
from toy_task.GMM.targets.GMM_target import ConditionalGMMTarget, get_weights, get_gmm_target
from toy_task.GMM.algorithms.visualization.GMM_plot import plot2d_matplotlib
from toy_task.GMM.algorithms.evaluation.GMM_evaluation import js_divergence_gmm
from toy_task.GMM.projections.split_kl_projection import split_projection


def train_model(model: ConditionalGMM,
                target: ConditionalGMMTarget,
                n_epochs: int,
                batch_size: int,
                n_context: int,
                n_components: int,
                n_samples: int,
                init_lr,
                device,
                project,
                eps_mean: float or None,
                eps_cov: float or None,
                alpha: int or None,
                responsibility=True):

    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-5)
    contexts = target.get_contexts(n_context).to(device)
    train_size = int(n_context)
    # prev_js = float('inf')

    for epoch in range(n_epochs):
        # shuffle sampled contexts, since I use the same sample set
        indices = ch.randperm(train_size)
        shuffled_contexts = contexts[indices]

        # get old distribution for auxiliary reward
        gate_old, mean_old, chol_old = model(shuffled_contexts)
        gate_old = gate_old.clone().detach()
        mean_old = mean_old.clone().detach()
        chol_old = chol_old.clone().detach()

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
            for j in range(n_components):
                mean_pred_j = mean_pred[:, j]  # (batched_c, 2)
                chol_pred_j = chol_pred[:, j]
                mean_old_j = b_mean_old[:, j]
                chol_old_j = b_chol_old[:, j]

                if project:
                    mean_proj_j, chol_proj_j = split_projection(mean_pred_j, chol_pred_j, mean_old_j, chol_old_j, eps_mean, eps_cov)

                    # model and target log probability
                    model_samples = model.get_rsamples(mean_proj_j, chol_proj_j, n_samples)  # shape (n_c, n_samples, 2)
                    log_model_j = model.log_prob(mean_proj_j, chol_proj_j, model_samples)
                    log_target_j = target.log_prob_tgt(b_contexts, model_samples)

                    # regression step
                    pred_dist = MultivariateNormal(mean_pred_j, scale_tril=chol_pred_j)
                    proj_dist = MultivariateNormal(mean_proj_j, scale_tril=chol_proj_j)
                    reg_loss = kl_divergence(pred_dist, proj_dist).unsqueeze(1).expand(-1, model_samples.shape[1])

                    aux_loss = model.auxiliary_reward(j, b_gate_old, b_mean_old, b_chol_old, model_samples)
                    gate_pred_j = gate_pred[:, j].unsqueeze(1).expand(-1, model_samples.shape[1])
                    loss_j = ch.exp(gate_pred_j) * (log_model_j - log_target_j - aux_loss + alpha * reg_loss + gate_pred_j)
                else:
                    model_samples = model.get_rsamples(mean_pred_j, chol_pred_j, n_samples)
                    log_model_j = model.log_prob(mean_pred_j, chol_pred_j, model_samples)
                    log_target_j = target.log_prob_tgt(b_contexts, model_samples)
                    if responsibility:
                        # loss: with log responsibility but without projection
                        aux_loss = model.auxiliary_reward(j, b_gate_old, b_mean_old, b_chol_old, model_samples)
                        # loss_j = log_model_j - log_target_j - auxiliary_loss
                        gate_pred_j = gate_pred[:, j].unsqueeze(1).expand(-1, model_samples.shape[1])
                        loss_j = ch.exp(gate_pred_j) * (log_model_j - log_target_j - aux_loss + gate_pred_j)
                    else:
                        # loss: without log responsibility or projection
                        loss_j = log_model_j - log_target_j
                loss_component.append(loss_j.mean())
            loss = ch.sum(ch.stack(loss_component))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            wandb.log({"train_loss": loss.item()})

        # Evaluation
        if (epoch + 1) % 10 == 0:
            model.eval()
            eval_contexts = target.get_contexts(200).to(device)
            target_mean = target.mean_fn(eval_contexts)
            target_chol = target.chol_fn(eval_contexts)

            # for non-uniform but random gating
            target_gate = get_weights(eval_contexts)
            p = (target_gate, target_mean, target_chol)

            model_gate, model_mean, model_chol = model(eval_contexts)
            q = (model_gate, model_mean, model_chol)

            js_div = js_divergence_gmm(p, q)
            # # trick from TRPL paper
            # current_js = js_div.item()
            # if current_js < prev_js:
            #     eps_mean *= 0.8
            #     eps_cov *= 0.8
            # else:
            #     eps_mean *= 1.1
            #     eps_cov *= 1.1
            # prev_js = current_js
            model.train()
            wandb.log({"JS Divergence": js_div.item()})

    print("Training done!")


def plot(model: ConditionalGMM,
         target: ConditionalGMMTarget):
    contexts = target.get_contexts(3).to('cpu')
    plot2d_matplotlib(target, model.to('cpu'), contexts, min_x=-15, max_x=15, min_y=-15, max_y=15)


def toy_task(n_epochs: int,
             batch_size: int,
             n_context: int,
             n_components: int,
             n_samples: int,
             fc_layer_size: int,
             init_lr: float,
             model_name: str,
             initialization_type: str,
             project: bool,
             eps_mean: float,
             eps_cov: float,
             alpha: int):
    # Device
    device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Wandb
    wandb.init(project="ELBOopt_GMM", config={
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "n_context": n_context,
        "n_components": n_components,
        "fc_layer_size": fc_layer_size,
        "init_lr": init_lr,
        "eps_mean": eps_mean,
        "eps_cov": eps_cov,
        "alpha": alpha,
        "project": project,
        "model_name": model_name,
        "initialization_type": initialization_type})

    # Target
    target = get_gmm_target(n_components)

    # Model
    model = get_model(model_name,
                      device,
                      fc_layer_size,
                      n_components,
                      initialization_type)

    # Training
    train_model(model, target,
                n_epochs, batch_size, n_context, n_components, n_samples, init_lr, device,  # training hyperparameter
                project, eps_mean, eps_cov, alpha)  # projection hyperparameter

    # Plotting
    plot(model, target)
