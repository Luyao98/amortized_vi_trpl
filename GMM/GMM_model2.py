import torch as ch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, kl_divergence
import wandb

from GMM.GMM_target import ConditionalGMMTarget, get_cov_fn, get_mean_fn
from GMM.GMM_simple_target import ConditionalGMMSimpleTarget, get_cov_fns, get_mean_fns
from GMM.GMM_plot import plot2d_matplotlib
from GMM.GMM_simple_plot import gaussian_simple_plot
from Gaussian.Gaussian_model import GaussianNN, covariance, get_samples, log_prob, log_prob_gmm
from Gaussian.split_kl_projection import split_projection
from Gaussian.utils import initialize_weights

ch.autograd.set_detect_anomaly(True)

# class GateNN(nn.Module):
#     def __init__(self, fc_layer_size, n_components):
#         super(GateNN, self).__init__()
#         self.fc1 = nn.Linear(1, fc_layer_size)
#         # self.fc2 = nn.Linear(fc_layer_size, fc_layer_size)
#         self.fc3 = nn.Linear(fc_layer_size, n_components)
#
#     def forward(self, x):
#         x = ch.relu(self.fc1(x))
#         # x = ch.relu(self.fc2(x))
#         x = ch.log_softmax(self.fc3(x), dim=-1)
#         return x


class ConditionalGMM(nn.Module):
    def __init__(self, fc_layer_size, n_components):
        super(ConditionalGMM, self).__init__()
        # self.gate = GateNN(fc_layer_size, n_components)
        self.gaussian_list = nn.ModuleList([GaussianNN(fc_layer_size) for _ in range(n_components)])
        self.optimizers = [optim.Adam(gaussian.parameters(), lr=init_lr, weight_decay=weight_decay) for gaussian in
                           self.gaussian_list]

    def forward(self, x):
        # gate = self.gate(x)

        means, chols = [], []
        for gaussian in self.gaussian_list:
            mean, chol = gaussian(x)
            means.append(mean)
            chols.append(chol)
        return ch.stack(means, dim=1), ch.stack(chols, dim=1)

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
                device):
    contexts = target.get_contexts_gmm(n_context).to(device)
    train_size = int(n_context)

    for epoch in range(n_epochs):
        # shuffle sampled contexts, since I use the same sample set
        indices = ch.randperm(train_size)
        shuffled_contexts = contexts[indices]

        # get old distribution for auxiliary reward
        mean_old, chol_old = model(shuffled_contexts)
        mean_old = mean_old.clone().detach()
        chol_old = chol_old.clone().detach()

        for batch_idx in range(0, train_size, batch_size):
            # get old distribution for current batch
            b_contexts = shuffled_contexts[batch_idx:batch_idx + batch_size]
            b_mean_old = mean_old[batch_idx:batch_idx + batch_size]
            b_chol_old = chol_old[batch_idx:batch_idx + batch_size]

            for optimizer in model.optimizers:
                optimizer.zero_grad()
            # prediction step
            # mean_pred, chol_pred = model(b_contexts)
            for j, (gaussian, optimizer) in enumerate(zip(model.gaussian_list, model.optimizers)):
                mean_pred_j, chol_pred_j = gaussian(b_contexts)

                # component-wise calculation

                # # with TRPL
                # eps_mean = 0.005       # mean projection bound
                # eps_cov = 0.001       # cov projection bound
                # alpha = 75            # regression penalty
                # mean_proj_j, chol_proj_j = split_projection(mean_pred_j, chol_pred_j, b_mean_old[:, j], b_chol_old[:, j], eps_mean, eps_cov)
                # cov_proj_j = covariance(chol_proj_j)
                #
                # # model and target log probability
                # model_samples = get_samples(mean_proj_j, cov_proj_j)  # shape (n_c, 2)
                # log_model_j = log_prob(mean_proj_j, cov_proj_j, model_samples)
                # log_target_j = target.log_prob_tgt(b_contexts, model_samples)
                #
                # # regression step
                # pred_dist = MultivariateNormal(mean_pred_j, scale_tril=chol_pred_j)
                # proj_dist = MultivariateNormal(mean_proj_j, scale_tril=chol_proj_j)
                # reg_loss = kl_divergence(pred_dist, proj_dist)
                #
                # auxiliary_loss = model.auxiliary_reward2(j, b_mean_old, b_chol_old, model_samples)
                # loss_j = (log_model_j - log_target_j - auxiliary_loss + alpha * reg_loss).mean()

                cov_pred_j = covariance(chol_pred_j)
                model_samples = get_samples(mean_pred_j, cov_pred_j)
                log_model_j = log_prob(mean_pred_j, cov_pred_j, model_samples)
                log_target_j = target.log_prob_tgt(b_contexts, model_samples)
                # loss_j = (log_model_j - log_target_j).mean()
                auxiliary_loss = model.auxiliary_reward2(j, b_mean_old, b_chol_old, model_samples)
                loss_j = (log_model_j - log_target_j - auxiliary_loss).mean()

                loss_j.backward(retain_graph=j < n_components - 1)
                optimizer.step()
                wandb.log({f"training loss {j}": loss_j.item()})

    print("Training done!")


if __name__ == "__main__":
    # Device
    device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Training parameters
    n_epochs = 150
    batch_size = 64
    n_context = 1280
    n_components = 2
    fc_layer_size = 64
    init_lr = 0.001
    weight_decay = 1e-5

    # Wandb
    wandb.init(project="ELBOopt_GMM", config={
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "n_context": n_context,
        "n_components": n_components,
        "fc_layer_size": fc_layer_size,
        "init_lr": init_lr,
        "weight_decay": weight_decay,
    })
    config = wandb.config

    # Target
    mean_target = get_mean_fn(n_components)
    cov_target = get_cov_fn(n_components)
    target = ConditionalGMMTarget(mean_target, cov_target)

    # Model
    model = ConditionalGMM(fc_layer_size, n_components).to(device)
    initialize_weights(model, initialization_type="xavier")
    #  optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)

    # Training
    train_model(model, target, n_epochs, batch_size, n_context, n_components, device)

    # Plot
    contexts = target.get_contexts_gmm(3).to('cpu')
    mean = target.mean_fn(contexts)
    print("target mean:", mean)
    plot2d_matplotlib(target, model.to('cpu'), contexts, min_x=-20, max_x=20, min_y=-20, max_y=20)