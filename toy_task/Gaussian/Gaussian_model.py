import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, kl_divergence

import wandb
# from torchviz import make_dot

from toy_task.GMM.utils.network_utils import initialize_weights
from toy_task.GMM.utils.torch_utils import diag_bijector, fill_triangular, inverse_softplus
from toy_task.Gaussian.Gaussian_targets import ConditionalGaussianTarget, get_cov_fn, get_mean_fn
from toy_task.Gaussian.Gaussian_plot import gaussian_plot
from toy_task.GMM.projections.kl_projection import KLProjection
from toy_task.GMM.projections.split_kl_projection import split_projection


# np.random.seed(37)
# torch.manual_seed(37)

# torch.autograd.set_detect_anomaly(True)


class GaussianNN(nn.Module):
    def __init__(self, fc_layer_size, init_bias_mean=None, init_bias_chol=None):
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

        if init_bias_mean is not None:
            with torch.no_grad():
                self.fc_mean.bias.copy_(torch.tensor(init_bias_mean, dtype=self.fc_mean.bias.dtype))

        if init_bias_chol is not None:
            with torch.no_grad():
                self.fc_chol.bias.copy_(torch.tensor(init_bias_chol, dtype=self.fc_chol.bias.dtype))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mean = self.fc_mean(x)
        flat_chol = self.fc_chol(x)
        chol_matrix = fill_triangular(flat_chol)
        chol = diag_bijector(lambda z: self.diag_activation(z + self.get_activation_shift()) + self.minimal_std,
                             chol_matrix)
        return mean, chol

    def get_activation_shift(self):
        init_std = self.init_std
        minimal_std = self.minimal_std
        return self.diag_activation_inv(init_std - minimal_std)

    def covariance(self, chol):
        cov_matrix = chol @ chol.transpose(-1, -2)
        return cov_matrix

    def get_rsamples(self, mean, chol, n_samples):
        rsamples = MultivariateNormal(loc=mean, scale_tril=chol).rsample(torch.Size([n_samples]))
        return rsamples.transpose(0, 1)

    def log_prob(self, mean, chol, samples):
        log = [MultivariateNormal(loc=mean[i], scale_tril=chol[i]).log_prob(samples[i]).sum() for i in range(mean.shape[0])]
        return torch.stack(log, dim=0).unsqueeze(-1)


def train_model(model: GaussianNN,
                target: ConditionalGaussianTarget,
                n_epochs: int,
                batch_size: int,
                n_context: int,
                eps_mean: float,
                eps_cov: float,
                alpha: int,
                optimizer: optim.Optimizer,
                split_proj: bool,
                sample_dist: bool,
                device
                ):

    train_size = int(n_context)
    contexts = target.get_contexts_1g(n_context).to(device)

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

            # projection step
            if split_proj:
                # project mean and cov separately, with TRPL
                mean_proj, chol_proj = split_projection(mean_pred, chol_pred, b_mean_old, b_chol_old, eps_mean, eps_cov)
            else:
                # project together with MORE
                cov_pred = model.covariance(chol_pred)
                b_cov_old = model.covariance(b_chol_old)
                b_cov_old = b_cov_old.clone().detach()
                mean_proj, chol_proj = KLProjection.apply((mean_pred, cov_pred), (b_mean_old, b_cov_old), 0.1)

            if sample_dist:
                # draw samples from projected distribution
                model_samples = model.get_rsamples(mean_proj, chol_proj, n_samples=1)
            else:
                # draw samples from predicted distribution
                model_samples = model.get_rsamples(mean_pred, chol_pred, n_samples=1)
            log_model = model.log_prob(mean_proj, chol_proj, model_samples)
            log_target = target.log_prob_1g(b_contexts, model_samples)
            proj_loss = (log_model - log_target).mean()

            # regression step
            pred_dist = MultivariateNormal(mean_pred, scale_tril=chol_pred)
            proj_dist = MultivariateNormal(mean_proj, scale_tril=chol_proj)
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
    n_epochs = 150
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

    init_bias_mean = [1.0, -1.0]
    # init_bias_chol = [5.0, 0.0, 5.0]

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
    model = GaussianNN(fc_layer_size, init_bias_mean).to(device)
    initialize_weights(model, initialization_type="orthogonal", preserve_bias_layers=['fc_mean'])
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)

    # Training
    train_model(model, target, n_epochs, batch_size, n_context, eps_mean, eps_cov, alpha, optimizer, split_proj, sample_dist, device)

    # Plot
    contexts = target.get_contexts_1g(5).to('cpu')
    gaussian_plot(model.to('cpu'), target, contexts)
