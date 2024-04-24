import torch as ch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, kl_divergence
import wandb
# import time

from toy_task.GMM.targets.GMM_target import ConditionalGMMTarget, get_gmm_target
from toy_task.GMM.targets.GMM_simple_target import ConditionalGMMSimpleTarget
from toy_task.GMM.algorithms.visualization import plot2d_matplotlib
from toy_task.Gaussian import GaussianNN, covariance, get_samples, log_prob, log_prob_gmm
from toy_task.GMM.projections.split_kl_projection import split_projection
from toy_task.GMM.utils.network_utils import initialize_weights
from toy_task.GMM.algorithms.evaluation import js_divergence_gmm

ch.autograd.set_detect_anomaly(True)


class ConditionalGMM(nn.Module):
    def __init__(self,
                 fc_layer_size,
                 n_components,
                 init_bias_mean_list=None,
                 init_bias_chol_list=None):
        super(ConditionalGMM, self).__init__()
        self.gaussian_list = nn.ModuleList()

        for i in range(n_components):
            init_bias_mean = init_bias_mean_list[i] if init_bias_mean_list is not None else None
            init_bias_chol = init_bias_chol_list[i] if init_bias_chol_list is not None else None
            gaussian = GaussianNN(fc_layer_size, init_bias_mean=init_bias_mean, init_bias_chol=init_bias_chol)
            self.gaussian_list.append(gaussian)

    def forward(self, x):
        # p(x|o,c)
        means, chols = [], []
        for gaussian in self.gaussian_list:
            mean, chol = gaussian(x)
            means.append(mean)
            chols.append(chol)
        # shape (n_contexts, n_components), (n_contexts, n_components, 2), (n_contexts, n_components, 2, 2)
        return ch.stack(means, dim=1), ch.stack(chols, dim=1)

    @staticmethod
    def auxiliary_reward(j, gate_old, mean_old, chol_old, samples):
        numerator = gate_old[:, j] + MultivariateNormal(mean_old[:, j], scale_tril=chol_old[:, j]).log_prob(samples)

        denominator = log_prob_gmm(mean_old, chol_old, gate_old, samples)
        denominator = ch.sum(denominator, dim=0)

        aux_reward = numerator - denominator
        return aux_reward

    @staticmethod
    def auxiliary_reward2(j, mean_old, chol_old, samples):
        """
        used for the model without gating network. However, to calculate the auxiliary reward, the gate value is needed.
        To use the closed-form solution to calculate the gate value, the auxiliary reward is needed.
        So here I just set the gate value to 1/n_components.
        """
        gate_old = (1 / mean_old.shape[1]) * ch.ones(mean_old.shape[0], mean_old.shape[1])
        log_gate_old = ch.log(gate_old).detach().to(mean_old.device)
        numerator = log_gate_old[:, j] + MultivariateNormal(mean_old[:, j], scale_tril=chol_old[:, j]).log_prob(samples)
        denominator = log_prob_gmm(mean_old, chol_old, log_gate_old, samples)
        aux_reward = numerator - denominator
        return aux_reward


def train_model(model: ConditionalGMM,
                target: ConditionalGMMTarget or ConditionalGMMSimpleTarget,
                n_epochs: int,
                batch_size: int,
                n_context: int,
                n_components: int,
                eps_mean: float,
                eps_cov: float,
                alpha: int,
                optimizer,
                device,
                project,
                responsibility):

    contexts = target.get_contexts_gmm(n_context).to(device)
    # eval_contexts = target.get_contexts_gmm(300).to(device)
    train_size = int(n_context)
    # prev_kl_divergence = float('inf')

    for epoch in range(n_epochs):
        # start_time = time.time()

        # shuffle sampled contexts, since I use the same sample set
        indices = ch.randperm(train_size)
        shuffled_contexts = contexts[indices]

        # get old distribution for auxiliary reward
        mean_old, chol_old = model(shuffled_contexts)
        mean_old = mean_old.clone().detach()
        chol_old = chol_old.clone().detach()

        for batch_idx in range(0, train_size, batch_size):
            # get old distribution for current batch
            b_contexts = shuffled_contexts[batch_idx:batch_idx+batch_size]
            b_mean_old = mean_old[batch_idx:batch_idx+batch_size]
            b_chol_old = chol_old[batch_idx:batch_idx+batch_size]

            # prediction step
            mean_pred, chol_pred = model(b_contexts)

            # component-wise calculation
            loss_component = []
            auxiliary_loss = []
            for j in range(n_components):
                mean_pred_j = mean_pred[:, j]  # (batched_c, 2)
                chol_pred_j = chol_pred[:, j]
                mean_old_j = b_mean_old[:, j]
                chol_old_j = b_chol_old[:, j]

                if project:
                    mean_proj_j, chol_proj_j = split_projection(mean_pred_j, chol_pred_j, mean_old_j, chol_old_j, eps_mean, eps_cov)
                    cov_proj_j = covariance(chol_proj_j)

                    # model and target log probability
                    model_samples = get_samples(mean_proj_j, cov_proj_j)  # shape (n_c, 2)
                    log_model_j = log_prob(mean_proj_j, cov_proj_j, model_samples)
                    log_target_j = target.log_prob_tgt(b_contexts, model_samples)

                    # regression step
                    pred_dist = MultivariateNormal(mean_pred_j, scale_tril=chol_pred_j)
                    proj_dist = MultivariateNormal(mean_proj_j, scale_tril=chol_proj_j)
                    reg_loss = kl_divergence(pred_dist, proj_dist)
                    # regression_loss.append(reg_loss)

                    aux_loss = model.auxiliary_reward2(j, b_mean_old, b_chol_old, model_samples)
                    loss_j = log_model_j - log_target_j - aux_loss + alpha * reg_loss
                else:
                    cov_pred_j = covariance(chol_pred_j)
                    model_samples = get_samples(mean_pred_j, cov_pred_j)
                    log_model_j = log_prob(mean_pred_j, cov_pred_j, model_samples)
                    log_target_j = target.log_prob_tgt(b_contexts, model_samples)
                    if responsibility:
                        # loss: with log responsibility but without projection
                        aux_loss = model.auxiliary_reward2(j, b_mean_old, b_chol_old, model_samples)
                        loss_j = log_model_j - log_target_j - aux_loss
                    else:
                        # loss: without log responsibility or projection
                        loss_j = log_model_j - log_target_j
                auxiliary_loss.append(aux_loss.mean())
                loss_component.append(loss_j.mean())
            loss = ch.sum(ch.stack(loss_component))
            auxiliary_loss = ch.mean(ch.stack(auxiliary_loss))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            wandb.log({"train_loss": loss.item(),
                       f"auxiliary loss": auxiliary_loss.item()
                       })

        # end_time = time.time()
        # epoch_duration = end_time - start_time
        # print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds.")

        # # evaluation step, only consider kl divergence between components
        # if (epoch + 1) % 5 == 0:
        #     model.eval()
        #
        #     eval_contexts = target.get_contexts_gmm(500).to(device)
        #     target_mean = target.mean_fn(eval_contexts)
        #     target_cov = target.cov_fn(eval_contexts)
        #     model_mean, model_chol = model(eval_contexts)
        #     model_cov = covariance(model_chol)
        #
        #     kl = []
        #     for i in range(500):
        #         model_dist = MultivariateNormal(model_mean[i], model_cov[i])
        #         target_dist = MultivariateNormal(target_mean[i], target_cov[i])
        #         kl_i = kl_divergence(model_dist, target_dist)
        #         kl.append(kl_i)
        #     kl = ch.stack(kl, dim=0)
        #     kl = ch.mean(kl)
        #
        #     # # trick from TRPL paper
        #     # current_kl_divergence = kl.item()
        #     # if current_kl_divergence < prev_kl_divergence:
        #     #     # if eps_mean > 0.5:
        #     #         eps_mean *= 0.8
        #     #     # if eps_cov > 0.05:
        #     #         eps_cov *= 0.8
        #     # else:
        #     #     eps_mean *= 1.1
        #     #     eps_cov *= 1.1
        #     # prev_kl_divergence = current_kl_divergence
        #
        #     print(f'Epoch {epoch+1}: KL Divergence = {kl.item()}')
        #     model.train()
        #
        #     wandb.log({"kl_divergence": kl.item()})
        # evaluation step, using JS divergence
        # if (epoch + 1) % 10 == 0:
        #     model.eval()
        #
        #     target_mean = target.mean_fn(eval_contexts)
        #     target_cov = target.cov_fn(eval_contexts)
        #     target_chol = ch.linalg.cholesky(target_cov)
        #     target_gate = (1 / n_components) * ch.ones(eval_contexts.shape[0], n_components).to(device)
        #     p = (target_gate, target_mean, target_chol)
        #
        #     model_mean, model_chol = model(eval_contexts)
        #     model_gate = (1 / n_components) * ch.ones(eval_contexts.shape[0], n_components).to(device)
        #     q = (model_gate, model_mean, model_chol)
        #
        #     js_div = js_divergence_gmm(p, q)
        #     print(f'Epoch {epoch+1}: JS Divergence = {js_div.item()}')
        #     model.train()
        #     wandb.log({"JS Divergence": js_div.item()})
    print("Training done!")


if __name__ == "__main__":
    # Device
    device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Training parameters
    n_epochs = 150
    batch_size = 64
    n_context = 1280
    n_components = 4
    fc_layer_size = 128
    init_lr = 0.01
    weight_decay = 1e-5
    eps_mean = 0.1       # mean projection bound
    eps_cov = 0.5       # cov projection bound
    alpha = 10            # regression penalty

    project = False        # calling projection or not
    responsibility = True  # consider responsibility or not

    init_bias_mean_list = [
        [10.0, 10.0],
        [-10.0, -10.0],
        [10.0, -10.0],
        [-10.0, 10.0]
    ]
    # Wandb
    wandb.init(project="ELBOopt_GMM", config={
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "n_context": n_context,
        "n_components": n_components,
        "fc_layer_size": fc_layer_size,
        "init_lr": init_lr,
        "weight_decay": weight_decay,
        "eps_mean": eps_mean,
        "eps_cov": eps_cov,
        "alpha": alpha
    })
    config = wandb.config

    # Target
    target = get_gmm_target(n_components)

    # Simple Target
    # mean_target = get_mean_fns(n_components)
    # cov_target = get_cov_fns(n_components)
    # target = ConditionalGMMSimpleTarget(mean_target, cov_target)

    # Model
    model = ConditionalGMM(fc_layer_size, n_components, init_bias_mean_list).to(device)
    initialize_weights(model, initialization_type="xavier", preserve_bias_layers=['fc_mean'])
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)

    # Training
    train_model(model, target, n_epochs, batch_size, n_context, n_components,
                eps_mean, eps_cov, alpha, optimizer, device,
                project, responsibility)

    # Evaluation
    model.eval()
    eval_contexts = target.get_contexts_gmm(1000).to(device)
    target_mean = target.mean_fn(eval_contexts)
    target_cov = target.cov_fn(eval_contexts)
    target_chol = ch.linalg.cholesky(target_cov)
    target_gate = (1 / n_components) * ch.ones(eval_contexts.shape[0], n_components).to(device)
    p = (target_gate, target_mean, target_chol)

    model_mean, model_chol = model(eval_contexts)
    model_gate = (1 / n_components) * ch.ones(eval_contexts.shape[0], n_components).to(device)
    q = (model_gate, model_mean, model_chol)

    js_div = js_divergence_gmm(p, q)
    print(f"JS Divergence: {js_div.item()}")
    wandb.log({"JS Divergence": js_div.item()})

    # Plot
    contexts = target.get_contexts_gmm(3).to('cpu')
    # mean = target.mean_fn(contexts)
    # print("target mean:", mean)
    # plot2d_matplotlib(target, model.to('cpu'), contexts)
    plot2d_matplotlib(target, model.to('cpu'), contexts, min_x=-15, max_x=15, min_y=-15, max_y=15)
    # plot2d_matplotlib(target, model.to('cpu'), contexts, min_x=-10, max_x=10, min_y=-10, max_y=10)
    # gaussian_simple_plot(model.to('cpu'), target, contexts)