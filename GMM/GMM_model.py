import torch as ch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, kl_divergence
import wandb

from GMM.GMM_target import ConditionalGMMTarget, get_cov_fn, get_mean_fn
from GMM.GMM_plot import plot2d_matplotlib
from Gaussian.model import GaussianNN
from Gaussian.kl_projection import KLProjection
from Gaussian.split_kl_projection import mean_projection, CovKLProjection
from Gaussian.utils import initialize_weights

ch.autograd.set_detect_anomaly(True)


class GateNN(nn.Module):
    def __init__(self, fc_layer_size, n_components):
        super(GateNN, self).__init__()
        self.fc1 = nn.Linear(1, fc_layer_size)
        # self.fc2 = nn.Linear(fc_layer_size, fc_layer_size)
        self.fc3 = nn.Linear(fc_layer_size, n_components)

    def forward(self, x):
        x = ch.relu(self.fc1(x))
        # x = ch.relu(self.fc2(x))
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
        # shape (n_contexts, n_components), (n_contexts, n_components, 2), (n_contexts, n_components, 2, 2)
        return gate, ch.stack(means, dim=1), ch.stack(chols, dim=1)

    @staticmethod
    def covariance_gmm(chol):
        cov_matrix = chol @ chol.transpose(-1, -2)
        return cov_matrix

    @staticmethod
    def get_samples(mean, cov):
        return MultivariateNormal(loc=mean, covariance_matrix=cov).rsample()

    @staticmethod
    def log_prob(mean, cov, samples):
        return MultivariateNormal(loc=mean, covariance_matrix=cov).log_prob(samples)

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
                log = log + ch.log(gate) + sub_log
            log_c.append(log)
        return ch.stack(log_c, dim=1)   # shape(samples.shape[0], n_contexts)

    @staticmethod
    def auxiliary_reward(j, gate_old, mean_old, cov_old, samples):
        n_contexts, n_components, _ = mean_old.shape
        numerator = (ch.log(gate_old[:, j]) +
                     MultivariateNormal(loc=mean_old[:, j, ], covariance_matrix=cov_old[:, j, ]).log_prob(samples))
        denominator = ch.zeros(n_contexts)
        for i in range(n_components):
            o_value = (ch.log(gate_old[:, i]) +
                       MultivariateNormal(loc=mean_old[:, i, ], covariance_matrix=cov_old[:, i, ]).log_prob(samples))
            denominator = denominator + o_value
        auxiliary_reward = numerator - denominator
        return auxiliary_reward


def train_model(model, target, n_epochs, batch_size, n_context, n_components, eps, alpha, optimizer, split):
    contexts = target.get_contexts(n_context).clone().detach()
    train_size = int(n_context)

    for epoch in range(n_epochs):
        # shuffle sampled contexts, since I use the same sample set
        indices = ch.randperm(train_size)
        shuffled_contexts = contexts[indices]

        # get old distribution for auxiliary reward
        gate_old, mean_old, chol_old = model(shuffled_contexts)
        cov_old = model.covariance_gmm(chol_old)
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
            cov_pred = model.covariance_gmm(chol_pred)  # shape (n_c, n_o, 2, 2)

            # component-wise calculation
            # projection step
            loss_component = []
            for j in range(n_components):
                mean_pred_j = mean_pred[:, j, ]  # (batched_c, 2)
                chol_pred_j = chol_pred[:, j, ]
                cov_pred_j = cov_pred[:, j, ]
                mean_old_j = b_mean_old[:, j, ]
                chol_old_j = b_chol_old[:, j, ]
                cov_old_j = b_cov_old[:, j, ]

                if split:
                    mean_proj_j = mean_projection(mean_pred_j, mean_old_j, chol_old_j, 0.1)
                    cov_proj_j = CovKLProjection.apply(chol_old_j, chol_pred_j, cov_pred_j, eps)
                    # because of numerical unstable need it to guarantee semi-pos. defi.
                    nan_mask = ch.isnan(cov_proj_j)
                    cov_proj_j = ch.where(nan_mask, cov_pred_j, cov_proj_j)
                else:
                    mean_proj_j, cov_proj_j = KLProjection.apply((mean_pred_j, cov_pred_j), (mean_old_j, cov_old_j), eps)

                # track gradient of samples
                model_samples = model.get_samples(mean_proj_j, cov_proj_j).requires_grad_(True)  # shape (n_c, 2)
                log_model_j = model.log_prob(mean_proj_j, cov_proj_j, model_samples)
                log_target_j = ch.sum(target.log_prob(b_contexts, model_samples), dim=0)

                # regeression step
                pred_dist = MultivariateNormal(mean_pred_j, cov_pred_j)
                proj_dist = MultivariateNormal(mean_proj_j, cov_proj_j)
                reg_loss = ch.distributions.kl.kl_divergence(pred_dist, proj_dist)

                # auxiliary reward
                auxiliary_loss = model.auxiliary_reward(j, b_gate_old, b_mean_old, b_cov_old, model_samples)

                # loss choice 1: weighted sum of component loss, but without H(p(o|c))
                #                -> model tends to set all other gates to 0
                # loss_j = gate_pred[:, j] * (log_model_j - log_target_j - auxiliary_loss + alpha * reg_loss)

                # loss choice 2: weighted sum of component loss, with H(p(o|c))
                #                -> model still tends to set all other gates to 0
                # loss_j = gate_pred[:, j] * (log_model_j - log_target_j - auxiliary_loss + gate_pred[:, j] + alpha * reg_loss)

                # loss choice 3: sum of component loss
                loss_j = log_model_j - log_target_j - auxiliary_loss + alpha * reg_loss
                loss_component.append(loss_j.mean())

            loss = ch.sum(ch.stack(loss_component))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            wandb.log({"train_loss": loss.item()})

        # evaluation step, only consider kl divergence between components
        if epoch % 5 == 0:  # Evaluate every 5 epochs
            model.eval()  # Set model to evaluation mode

            eval_contexts = target.get_contexts(1000)
            target_mean = target.mean_fn(eval_contexts)
            target_cov = target.cov_fn(eval_contexts)
            _, model_mean, model_chol = model(eval_contexts)
            model_cov = model.covariance_gmm(model_chol)

            kl =[]
            for i in range(1000):
                model_dist = MultivariateNormal(model_mean[i], model_cov[i])
                target_dist = MultivariateNormal(target_mean[i], target_cov[i])
                kl_i = kl_divergence(model_dist, target_dist)
                kl.append(kl_i)
            kl = ch.stack(kl, dim=0)
            kl = ch.sum(kl) / 1000
            print(f'Epoch {epoch}: KL Divergence = {kl.item()}')
            model.train()

            wandb.log({"kl_divergence": kl.item()})

    print("Training done!")


if __name__ == "__main__":
    # Device
    device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Training parameters
    n_epochs = 20
    batch_size = 32
    n_context = 640
    n_components = 2
    fc_layer_size = 64
    init_lr = 0.01
    weight_decay = 1e-5
    eps = 0.1  # projection
    alpha = 50  # regression
    split = True

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
        "alpha": alpha
    })
    config = wandb.config

    # Target
    mean_target = get_mean_fn(n_components)
    cov_target = get_cov_fn(n_components)
    target = ConditionalGMMTarget(mean_target, cov_target)

    # Model
    model = ConditionalGMM(fc_layer_size, n_components).to(device)
    initialize_weights(model, initialization_type="orthogonal")
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)

    # Training
    train_model(model, target, n_epochs, batch_size, n_context, n_components, eps, alpha, optimizer, split)

    # Plot
    contexts = target.get_contexts(2)
    # for loss choice 3
    # plot2d_matplotlib(target, model, contexts)
    # for loss 1/2
    plot2d_matplotlib(target, model, contexts, min_x=-10, max_x=10, min_y=-10, max_y=10)
