import torch as ch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, kl_divergence
import wandb

from GMM.GMM_target import ConditionalGMMTarget, get_cov_fn, get_mean_fn
from GMM.GMM_plot import plot2d_matplotlib
from Gaussian.Gaussian_model import covariance, get_samples, log_prob
from Gaussian.split_kl_projection import split_projection
from Gaussian.utils import fill_triangular_gmm, initialize_weights

ch.autograd.set_detect_anomaly(True)


class ConditionalGMM_test(nn.Module):
    def __init__(self, fc_layer_size, n_components):
        super(ConditionalGMM_test, self).__init__()
        self.n_components = n_components

        self.fc1 = nn.Linear(1, fc_layer_size)
        self.fc2 = nn.Linear(fc_layer_size, fc_layer_size)
        self.fc3 = nn.Linear(fc_layer_size, fc_layer_size)
        self.gating = nn.Linear(fc_layer_size, n_components)
        self.fc_mean = nn.Linear(fc_layer_size, n_components * 2)
        self.fc_chol = nn.Linear(fc_layer_size, n_components * 3)

    def forward(self, x):
        x = ch.relu(self.fc1(x))
        x = ch.relu(self.fc2(x))
        x = ch.relu(self.fc3(x))
        gating = ch.log_softmax(self.gating(x), dim=1)
        mean = self.fc_mean(x)
        mean = mean.view(-1, self.n_components, 2)
        flat_chol = self.fc_chol(x)
        chol = fill_triangular_gmm(flat_chol, self.n_components)

        return gating, mean, chol


    @staticmethod
    def log_prob_gmm(means, chols, gates, samples):
        n_contexts, n_components, _ = means.shape
        log_c = []
        for c in range(n_contexts):
            log = ch.zeros(samples.shape[0])
            for i in range(n_components):
                loc = means[c, i,]
                chol = chols[c, i,]
                gate = ch.tensor(gates[c, i])
                sub_log = MultivariateNormal(loc, scale_tril=chol).log_prob(samples)
                log = log + gate + sub_log
            log_c.append(log)
        return ch.stack(log_c, dim=1)   # shape(samples.shape[0], n_contexts)

    @staticmethod
    def auxiliary_reward(j, gate_old, mean_old, cov_old, samples):
        n_contexts, n_components, _ = mean_old.shape
        numerator = gate_old[:, j] + log_prob(mean_old[:, j, ], cov_old[:, j, ], samples)

        denominator = ch.zeros(n_contexts)
        for i in range(n_components):
            o_value = gate_old[:, i] + log_prob(mean_old[:, i, ], cov_old[:, i, ], samples)
            denominator = denominator + o_value
        auxiliary_reward = numerator - denominator
        return auxiliary_reward
    # def auxiliary_reward(j, gate_old, mean_old, cov_old, samples):
    #     n_contexts, n_components, _ = mean_old.shape
    #     numerator = (ch.log(gate_old[:, j]) +
    #                  MultivariateNormal(loc=mean_old[:, j], covariance_matrix=cov_old[:, j]).log_prob(samples))
    #     log_probs = []
    #     for i in range(n_components):
    #         log_prob_i = ch.log(gate_old[:, i]) + MultivariateNormal(loc=mean_old[:, i],
    #                                                                     covariance_matrix=cov_old[:, i]).log_prob(samples)
    #         log_probs.append(log_prob_i)
    #
    #     # shape [n_contexts, n_samples, n_components]
    #     log_probs_tensor = ch.stack(log_probs, dim=-1)
    #
    #     # Use logsumexp to calculate the log of the sum of exponentials across the components dimension
    #     log_sum_exp = ch.logsumexp(log_probs_tensor, dim=-1)
    #
    #     auxiliary_reward = numerator - log_sum_exp
    #
    #     return auxiliary_reward


def train_model(model, target, n_epochs, batch_size, n_context, n_components, eps_mean, eps_cov, alpha, optimizer):
    contexts = target.get_contexts_gmm(n_context)
    train_size = int(n_context)

    for epoch in range(n_epochs):
        # shuffle sampled contexts, since I use the same sample set
        indices = ch.randperm(train_size)
        shuffled_contexts = contexts[indices]

        # get old distribution for auxiliary reward
        gate_old, mean_old, chol_old = model(shuffled_contexts)
        cov_old = covariance(chol_old)
        gate_old = gate_old.clone().detach()
        mean_old = mean_old.clone().detach()
        chol_old = chol_old.clone().detach()
        cov_old = cov_old.clone().detach()

        for batch_idx in range(0, train_size, batch_size):
            # get old distribution for current batch
            b_contexts = shuffled_contexts[batch_idx:batch_idx+batch_size]
            b_mean_old = mean_old[batch_idx:batch_idx+batch_size, ]
            b_chol_old = chol_old[batch_idx:batch_idx+batch_size, ]
            b_cov_old = cov_old[batch_idx:batch_idx+batch_size, ]
            b_gate_old = gate_old[batch_idx:batch_idx+batch_size, ]

            # prediction step
            gate_pred, mean_pred, chol_pred = model(b_contexts)
            cov_pred = covariance(chol_pred)  # shape (n_c, n_o, 2, 2)

            # component-wise calculation
            # projection step
            # loss_component = []
            log_model, log_target, regres_loss = [], [], []
            for j in range(n_components):
                mean_pred_j = mean_pred[:, j, ]  # (batched_c, 2)
                chol_pred_j = chol_pred[:, j, ]
                cov_pred_j = cov_pred[:, j, ]
                mean_old_j = b_mean_old[:, j, ]
                chol_old_j = b_chol_old[:, j, ]

                mean_proj_j, chol_proj_j = split_projection(mean_pred_j, chol_pred_j, mean_old_j, chol_old_j, eps_mean, eps_cov)
                cov_proj_j = covariance(chol_proj_j)

                # track gradient of samples
                model_samples = get_samples(mean_proj_j, cov_proj_j)  # shape (n_c, 2)
                log_model_j = log_prob(mean_proj_j, cov_proj_j, model_samples)
                log_target_j = ch.sum(target.log_prob_tgt(b_contexts, model_samples), dim=0)

                # regeression step
                pred_dist = MultivariateNormal(mean_pred_j, cov_pred_j)
                proj_dist = MultivariateNormal(mean_proj_j, cov_proj_j)
                reg_loss = kl_divergence(pred_dist, proj_dist)

                # auxiliary reward
                # auxiliary_loss = model.auxiliary_reward(j, b_gate_old, b_mean_old, b_cov_old, model_samples)

                # loss choice 1: weighted sum of component loss, but without H(p(o|c))
                #                -> model tends to set all other gates to 0
                # loss_j = gate_pred[:, j] * (log_model_j - log_target_j - auxiliary_loss + alpha * reg_loss)

                # loss choice 2: weighted sum of component loss, with H(p(o|c))
                #                -> model still tends to set all other gates to 0
                # loss_j = gate_pred[:, j] * (log_model_j - log_target_j - auxiliary_loss + gate_pred[:, j] + alpha * reg_loss)

                # loss choice 3: sum of component loss
                # loss_j = log_model_j - log_target_j - auxiliary_loss + alpha * reg_loss

                # loss choice 4: without log responsibility
                # loss_j = log_model_j - log_target_j + alpha * reg_loss
                # loss_component.append(loss_j.mean())
                log_model.append(log_model_j)
                log_target.append(log_target_j)
                regres_loss.append(reg_loss)
            log_model = ch.stack(log_model, dim=1)
            log_target = ch.stack(log_target, dim=1)
            regres_loss = ch.stack(regres_loss, dim=1)
            loss = ch.sum(log_model - log_target + alpha * regres_loss)

            # loss = ch.sum(ch.stack(loss_component))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            wandb.log({"train_loss": loss.item(),
                       "log_model": ch.mean(log_model).item(),
                       "log_target": ch.mean(log_target).item(),
                       "regression_loss": ch.mean(regres_loss).item()})

        # evaluation step, only consider kl divergence between components
        if (epoch + 1) % 5 == 0:  # Evaluate every 5 epochs
            model.eval()  # Set model to evaluation mode

            eval_contexts = target.get_contexts_gmm(1000)
            target_mean = target.mean_fn(eval_contexts)
            target_cov = target.cov_fn(eval_contexts)
            _, model_mean, model_chol = model(eval_contexts)
            model_cov = covariance(model_chol)

            kl = []
            for i in range(1000):
                model_dist = MultivariateNormal(model_mean[i], model_cov[i])
                target_dist = MultivariateNormal(target_mean[i], target_cov[i])
                kl_i = kl_divergence(model_dist, target_dist)
                kl.append(kl_i)
            kl = ch.stack(kl, dim=0)
            kl = ch.sum(kl) / 1000
            print(f'Epoch {epoch+1}: KL Divergence = {kl.item()}')
            model.train()

            wandb.log({"kl_divergence": kl.item()})

    print("Training done!")


if __name__ == "__main__":
    # Device
    device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Training parameters
    n_epochs = 50
    batch_size = 64
    n_context = 1280
    n_components = 1
    fc_layer_size = 64
    init_lr = 0.01
    weight_decay = 1e-5
    eps_mean = 0.05       # mean projection bound
    eps_cov = 0.005       # cov projection bound
    alpha = 75  # regression penalty

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
    mean_target = get_mean_fn(n_components)
    cov_target = get_cov_fn(n_components)
    target = ConditionalGMMTarget(mean_target, cov_target)

    # Model
    model = ConditionalGMM_test(fc_layer_size, n_components).to(device)
    initialize_weights(model, initialization_type="orthogonal")
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)

    # Training
    train_model(model, target, n_epochs, batch_size, n_context, n_components, eps_mean, eps_cov, alpha, optimizer)

    # Plot
    contexts = target.get_contexts_gmm(3)
    print("contexts:", contexts)
    mean = target.mean_fn(contexts)
    print("target mean:", mean)
    # for loss choice 3
    plot2d_matplotlib(target, model, contexts)