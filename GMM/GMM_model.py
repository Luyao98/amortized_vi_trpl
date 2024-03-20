import torch as ch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
import wandb

from GMM.GMM_target import ConditionalGMMTarget, get_cov_fn, get_mean_fn
from Gaussian.model import GaussianNN
from Gaussian.kl_projection import KLProjection
from Gaussian.utils import initialize_weights, likelihood

ch.autograd.set_detect_anomaly(True)


class GateNN(nn.Module):
    def __init__(self, fc_layer_size, n_components):
        super(GateNN, self).__init__()
        self.fc1 = nn.Linear(1, fc_layer_size)
        self.fc2 = nn.Linear(fc_layer_size, fc_layer_size)
        self.fc3 = nn.Linear(fc_layer_size, n_components)

    def forward(self, x):
        x = ch.relu(self.fc1(x))
        x = ch.relu(self.fc2(x))
        x = ch.softmax(self.fc3(x), dim=1)
        return x


class ConditionalGMM(nn.Module):
    def __init__(self, fc_layer_size, n_components):
        super(ConditionalGMM, self).__init__()
        self.gate = GateNN(fc_layer_size, n_components)
        self.gaussian_list = nn.ModuleList([GaussianNN(fc_layer_size) for _ in range(n_components)])

    def forward(self, x):
        # p(o|c)
        gate = self.gate(x)
        # p(x|o,c)
        means, chols = [], []
        for gaussian in self.gaussian_list:
            mean, chol = gaussian(x)
            means.append(mean)
            chols.append(chol)
        # shape (n_contexts, n_components, 2) (n_c, n_o, 2, 2)
        return gate, ch.stack(means, dim=1), ch.stack(chols, dim=1)

    @staticmethod
    def covariance_gmm(chol):
        cov_matrix = chol @ chol.transpose(-1, -2)
        return cov_matrix

    @staticmethod
    def get_samples(mean, chol, n=1):
        eps = ch.randn((n,) + mean.shape).to(dtype=chol.dtype, device=chol.device)[..., None]
        samples = (chol @ eps).squeeze(-1) + mean
        return samples.squeeze(0)


def train_model(model, target, n_epochs, batch_size, n_context, n_components, eps, optimizer):
    contexts = target.get_contexts(n_context).clone().detach()
    train_size = int(n_context)

    for epoch in range(n_epochs):
        # shuffle sampled contexts, since I use the same sample set
        indices = ch.randperm(train_size)
        shuffled_contexts = contexts[indices]

        # get old distribution for auxiliary reward
        gate_old, mean_old, chol_old = model(shuffled_contexts)
        cov_old = model.covariance_gmm(chol_old)

        for batch_idx in range(0, train_size, batch_size):
            # get old distribution for current batch
            b_contexts = shuffled_contexts[batch_idx:batch_idx+batch_size]
            b_mean_old = mean_old[batch_idx:batch_idx+batch_size, ]
            b_cov_old = cov_old[batch_idx:batch_idx+batch_size, ]
            b_gate_old = gate_old[batch_idx:batch_idx+batch_size, ]

            # prediction step
            gate_pred, mean_pred, chol_pred = model(b_contexts)
            cov_pred = model.covariance_gmm(chol_pred)  # shape (n_c, n_o, 2, 2)
            model_samples = model.get_samples(mean_pred, chol_pred)  # shape (n_c, n_o, 2)

            # component-wise calculation
            # projection step
            loss_component = []
            for j in range(n_components):
                mean_pred_j = mean_pred[:, j, ]  # (batched_c, 2)
                cov_pred_j = cov_pred[:, j, ]
                mean_old_j = b_mean_old[:, j, ]
                cov_old_j = b_cov_old[:, j, ]

                mean_proj_j, cov_proj_j = KLProjection.apply((mean_pred_j, cov_pred_j), (mean_old_j, cov_old_j), eps)
                log_model_j = likelihood(mean_proj_j, cov_proj_j, model_samples[:, j, ])
                # log_model_j = likelihood(mean_pred_j, cov_pred_j, model_samples[:, j,])
                log_target_j = target.log_prob(b_contexts, model_samples[:, j, ])

                # auxiliary reward
                with ch.no_grad():
                    numerator = ch.log(b_gate_old[:, j]) + MultivariateNormal(loc=b_mean_old[:, j,], covariance_matrix=b_cov_old[:, j,]).log_prob(model_samples[:, j, ])
                    denominator = ch.zeros(batch_size)
                    for i in range(n_components):
                        o_value = ch.log(b_gate_old[:, i]) + MultivariateNormal(loc=b_mean_old[:, i,], covariance_matrix=b_cov_old[:, i,]).log_prob(model_samples[:, j, ])
                        denominator = denominator + o_value
                    auxiliary_reward = numerator - denominator
                loss_j = gate_pred[:, j] * (log_model_j - log_target_j - auxiliary_reward + ch.log(gate_pred[:, j]))
                # loss_j = log_model_j - log_target_j - auxiliary_reward
                loss_component.append(loss_j.mean())

            loss = ch.sum(ch.stack(loss_component))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            wandb.log({"train_loss": loss.item()})

    print("Training done!")


if __name__ == "__main__":
    # Device
    device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Training parameters
    n_epochs = 50
    batch_size = 64
    n_context = 640
    n_components = 3
    fc_layer_size = 64
    init_lr = 0.01
    weight_decay = 1e-5
    eps = 0.1  # projection

    # Wandb
    wandb.init(project="ELBOopt_GMM", config={
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "n_context": n_context,
        "n_components": n_components,
        "fc_layer_size": fc_layer_size,
        "init_lr": init_lr,
        "weight_decay": weight_decay,
        "eps": eps,
    })
    config = wandb.config

    # Target
    mean_target = get_mean_fn(n_components)
    cov_target = get_cov_fn(n_components)
    target = ConditionalGMMTarget(mean_target, cov_target)

    # Model
    model = ConditionalGMM(fc_layer_size, n_components).to(device)
    initialize_weights(model, initialization_type="xavier")
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)

    # Training
    train_model(model, target, n_epochs, batch_size, n_context, n_components, eps, optimizer)
