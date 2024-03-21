import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
import wandb
# from torchviz import make_dot


from Gaussian.utils import fill_triangular, diag_bijector, inverse_softplus, initialize_weights, likelihood
from Gaussian.twodim_targets import ConditionalGaussianTarget, get_cov_fn, get_mean_fn
from Gaussian.kl_projection import KLProjection
from Gaussian.split_kl_projection import mean_projection, CovKLProjection


np.random.seed(37)
torch.manual_seed(37)

torch.autograd.set_detect_anomaly(True)


class GaussianNN(nn.Module):
    def __init__(self, fc_layer_size):
        super(GaussianNN, self).__init__()
        self.fc1 = nn.Linear(1, fc_layer_size)
        self.fc2 = nn.Linear(fc_layer_size, fc_layer_size)
        self.fc3_mean = nn.Linear(fc_layer_size, 2)
        self.fc3_chol = nn.Linear(fc_layer_size, 3)

        self.diag_activation = nn.Softplus()
        self.diag_activation_inv = inverse_softplus
        self.init_std = torch.tensor(1.0)
        self.minimal_std = 1e-5

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc3_mean(x)
        flat_chol = self.fc3_chol(x)
        chol_matrix = fill_triangular(flat_chol)
        chol = diag_bijector(lambda z: self.diag_activation(z + self.get_activation_shift()) + self.minimal_std, chol_matrix)
        return mean, chol

    def get_activation_shift(self):
        init_std = self.init_std
        minimal_std = self.minimal_std
        return self.diag_activation_inv(init_std - minimal_std)

    @staticmethod
    def covariance(chol):
        cov_matrix = chol @ chol.permute(0, 2, 1)
        return cov_matrix

    @staticmethod
    def get_samples(mean, chol, n=1):
        eps = torch.randn((n,) + mean.shape).to(dtype=chol.dtype, device=chol.device)[..., None]
        samples = (chol @ eps).squeeze(-1) + mean
        return samples.squeeze(0)


def train_model(model, target, n_epochs, batch_size, n_context, eps, alpha, optimizer, split_proj):
    train_size = int(n_context)
    contexts = target.get_contexts(n_context)

    for epoch in range(n_epochs):
        indices = torch.randperm(train_size)
        shuffled_contexts = contexts[indices]
        mean_old, chol_old = model(shuffled_contexts)
        mean_old = mean_old.detach()
        chol_old = chol_old.detach()

        for batch_idx in range(0, train_size, batch_size):
            b_contexts = shuffled_contexts[batch_idx:batch_idx+batch_size]
            b_mean_old = mean_old[batch_idx:batch_idx+batch_size, :]
            b_chol_old = chol_old[batch_idx:batch_idx+batch_size, :, :]
            b_cov_old = model.covariance(b_chol_old)

            # prediction step
            mean_pred, chol_pred = model(b_contexts)
            cov_pred = model.covariance(chol_pred)
            model_samples = model.get_samples(mean_pred, chol_pred)

            # projection step
            if split_proj:
                # split project mean and cov
                mean_proj = mean_projection(mean_pred, b_mean_old, b_chol_old, 0.1)
                cov_proj = CovKLProjection.apply(b_chol_old, chol_pred, cov_pred, eps)
            else:
                mean_proj, cov_proj = KLProjection.apply((mean_pred, cov_pred), (b_mean_old, b_cov_old), eps)

            log_model = likelihood(mean_proj, cov_proj, model_samples)
            log_target = target.log_prob(b_contexts, model_samples)
            proj_loss = (log_model - log_target).mean()

            # regeression step
            pred_dist = MultivariateNormal(mean_pred, cov_pred)
            proj_dist = MultivariateNormal(mean_proj, cov_proj)
            reg_loss = (torch.distributions.kl.kl_divergence(pred_dist, proj_dist)).mean()

            loss = proj_loss + alpha * reg_loss
            # loss = proj_loss

            # without projection
            # log_model = likelihood(mean_pred, cov_pred, model_samples)
            # log_target = target.log_prob(batch_contexts, model_samples)
            # loss = (log_model - log_target).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            # scheduler.step()

            wandb.log({"train_loss": loss.item()})
            # make_dot(loss, params=dict(model.named_parameters())).render("full_graph", format="png")
    print("Training done!")


def plot_contexts(model, contexts_test, x1_test, x2_test):
    x1, x2 = np.meshgrid(x1_test, x2_test)
    x1_flat = torch.from_numpy(x1.reshape(-1, 1).astype(np.float32)).detach()
    x2_flat = torch.from_numpy(x2.reshape(-1, 1).astype(np.float32)).detach()

    fig, axs = plt.subplots(1, len(contexts_test), figsize=(15, 5))

    for i, c in enumerate(contexts_test):
        c_expanded = c.unsqueeze(0).expand(x1_flat.shape[0], -1)
        mean, chol = model(c_expanded)
        x = torch.cat((x1_flat, x2_flat), dim=1)
        log_probs = MultivariateNormal(mean, scale_tril=chol).log_prob(x).exp().view(x1.shape).detach() + 1e-6
        axs[i].contourf(x1, x2, log_probs, levels=100)
        axs[i].set_title(f'Context: {c}')

    for ax in axs:
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Training parameters
    n_epochs = 50
    batch_size = 64
    n_context = 960
    fc_layer_size = 64
    init_lr = 0.01
    weight_decay = 1e-5
    eps = 0.1  # projection
    alpha = 25  # regerssion
    split_proj = False

    # Wandb
    wandb.init(project="ELBOopt_2D", save_code=False, config={
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "n_context": n_context,
        "fc_layer_size": fc_layer_size,
        "init_lr": init_lr,
        "weight_decay": weight_decay,
        "eps": eps,
        "alpha": alpha
    })
    config = wandb.config

    # Target
    mean_target = get_mean_fn('periodic')
    cov_target = get_cov_fn('periodic')
    target = ConditionalGaussianTarget(mean_target, cov_target)

    # Model
    model = GaussianNN(fc_layer_size).to(device)
    initialize_weights(model, initialization_type="orthogonal")
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)

    # Training
    train_model(model, target, n_epochs, batch_size, n_context, eps, alpha, optimizer, split_proj)

    # Plot
    contexts_test = torch.tensor([[-1.1443], [-0.1062], [-0.4056], [0.9927], [1.1044]])
    x1_test = np.linspace(-3, 3, 100)
    x2_test = np.linspace(-3, 3, 100)
    plot_contexts(model, contexts_test, x1_test, x2_test)
