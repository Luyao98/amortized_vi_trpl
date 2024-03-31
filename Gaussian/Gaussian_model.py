import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, kl_divergence

import wandb
# from torchviz import make_dot

from Gaussian.utils import fill_triangular, diag_bijector, inverse_softplus, initialize_weights
from Gaussian.Gaussian_targets import ConditionalGaussianTarget, get_cov_fn, get_mean_fn
from Gaussian.Gaussian_plot import gaussian_plot
from Gaussian.kl_projection import KLProjection
from Gaussian.split_kl_projection import split_projection


# np.random.seed(37)
# torch.manual_seed(37)

# torch.autograd.set_detect_anomaly(True)


class GaussianNN(nn.Module):
    def __init__(self, fc_layer_size):
        super(GaussianNN, self).__init__()
        self.fc1 = nn.Linear(1, fc_layer_size)
        self.fc2 = nn.Linear(fc_layer_size, fc_layer_size)
        self.fc3 = nn.Linear(fc_layer_size, fc_layer_size)
        self.fc_mean = nn.Linear(fc_layer_size, 2)
        self.fc_chol = nn.Linear(fc_layer_size, 3)

        self.diag_activation = nn.Softplus()
        self.diag_activation_inv = inverse_softplus
        self.init_std = torch.tensor(1.0)
        self.minimal_std = 1e-3

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mean = self.fc_mean(x)
        flat_chol = self.fc_chol(x)
        chol_matrix = fill_triangular(flat_chol)
        chol = diag_bijector(lambda z: self.diag_activation(z + self.get_activation_shift()) + self.minimal_std, chol_matrix)
        return mean, chol

    def get_activation_shift(self):
        init_std = self.init_std
        minimal_std = self.minimal_std
        return self.diag_activation_inv(init_std - minimal_std)


def covariance(chol):
    cov_matrix = chol @ chol.permute(0, 2, 1)
    return cov_matrix


def get_samples(mean, cov):
    return MultivariateNormal(loc=mean, covariance_matrix=cov).rsample()


def log_prob(mean, cov, samples):
    return torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov).log_prob(samples)


def train_model(model,
                target,
                n_epochs,
                batch_size,
                n_context,
                eps_mean,
                eps_cov,
                alpha,
                optimizer,
                split_proj,
                sample_dist
                ):

    train_size = int(n_context)
    contexts = target.get_contexts_1g(n_context)

    for epoch in range(n_epochs):
        indices = torch.randperm(train_size)
        shuffled_contexts = contexts[indices]
        mean_old, chol_old = model(shuffled_contexts)
        mean_old = mean_old.clone().detach()
        chol_old = chol_old.clone().detach()

        for batch_idx in range(0, train_size, batch_size):
            b_contexts = shuffled_contexts[batch_idx:batch_idx+batch_size]
            b_mean_old = mean_old[batch_idx:batch_idx+batch_size, :]
            b_chol_old = chol_old[batch_idx:batch_idx+batch_size, :, :]

            # prediction step
            mean_pred, chol_pred = model(b_contexts)
            cov_pred = covariance(chol_pred)

            # projection step
            if split_proj:
                # project mean and cov separately, with TRPL
                mean_proj, chol_proj = split_projection(mean_pred, chol_pred, b_mean_old, b_chol_old, eps_mean, eps_cov)
                cov_proj = covariance(chol_proj)
            else:
                # project together with MORE
                b_cov_old = covariance(b_chol_old)
                b_cov_old = b_cov_old.clone().detach()
                mean_proj, cov_proj = KLProjection.apply((mean_pred, cov_pred), (b_mean_old, b_cov_old), 0.1)

            if sample_dist:
                # draw samples from projected distribution
                model_samples = get_samples(mean_proj, cov_proj)
            else:
                # draw samples from predicted distribution
                model_samples = get_samples(mean_pred, cov_pred)
            log_model = log_prob(mean_proj, cov_proj, model_samples)
            log_target = target.log_prob_1g(b_contexts, model_samples)
            proj_loss = (log_model - log_target).mean()

            # regression step
            pred_dist = MultivariateNormal(mean_pred, cov_pred)
            proj_dist = MultivariateNormal(mean_proj, cov_proj)
            reg_loss = (kl_divergence(pred_dist, proj_dist)).mean()

            loss = proj_loss + alpha * reg_loss

            # # without projection
            # model_samples = get_samples(mean_pred, cov_pred)
            # log_model = log_prob(mean_pred, cov_pred, model_samples)
            # log_target = target.log_prob_g(b_contexts, model_samples)
            # loss = (log_model - log_target).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            # scheduler.step()

            wandb.log({"training loss": loss.item(),
                       "regression loss": reg_loss.item(),
                       "projection loss": proj_loss.item()})
            # make_dot(loss, params=dict(model.named_parameters())).render("full_graph", format="png")
    print("Training done!")


if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Training parameters
    n_epochs = 100
    batch_size = 64
    n_context = 1280
    fc_layer_size = 64
    init_lr = 0.01
    weight_decay = 1e-5
    eps_mean = 0.05       # mean projection bound
    eps_cov = 0.005       # cov projection bound
    alpha = 75           # regression penalty
    split_proj = True    # True: projection separately else together
    sample_dist = True    # True: draw samples from projected dist. else from predicted dist.

    # Wandb
    wandb.init(project="ELBOopt_2D", config={
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "n_context": n_context,
        "fc_layer_size": fc_layer_size,
        "init_lr": init_lr,
        "eps_mean": eps_mean,
        "eps_cov": eps_cov,
        "alpha": alpha,
        "split_proj": split_proj,
        "sample_dist": sample_dist
    })
    config = wandb.config

    # Target
    mean_target = get_mean_fn('periodic')
    cov_target = get_cov_fn('periodic')
    target = ConditionalGaussianTarget(mean_target, cov_target)

    # Model
    model = GaussianNN(fc_layer_size).to(device)
    initialize_weights(model, initialization_type="xavier")
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # Training
    train_model(model, target, n_epochs, batch_size, n_context, eps_mean, eps_cov, alpha, optimizer, split_proj, sample_dist)

    # Plot
    contexts = target.get_contexts_1g(5)
    gaussian_plot(model, target, contexts)
