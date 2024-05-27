import torch as ch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical

from toy_task.GMM.models.abstract_gmm_model import AbstractGMM
from toy_task.GMM.utils.torch_utils import diag_bijector, fill_triangular, inverse_softplus


class GateNN(nn.Module):
    def __init__(self, n_components, gate_size=128, num_layers=5, init_bias_gate=None):
        super(GateNN, self).__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList()

        self.fc_layers.append(nn.Linear(1, gate_size))
        for _ in range(1, num_layers):
            self.fc_layers.append(nn.Linear(gate_size, gate_size))
        self.fc_gate = nn.Linear(gate_size, n_components)

        # Set init uniform bias for gates
        if init_bias_gate is not None:
            with ch.no_grad():
                self.fc_gate.bias.copy_(ch.tensor(init_bias_gate, dtype=self.fc_gate.bias.dtype))

    def forward(self, x):
        for fc in self.fc_layers:
            x = ch.relu(fc(x))
        x = ch.log_softmax(self.fc_gate(x), dim=-1)
        return x


class GaussianNN(nn.Module):
    def __init__(self, fc_layer_size, dim_mean, dim_chol, num_layers=8, init_bias_mean=None, init_bias_chol=None):
        super(GaussianNN, self).__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList()

        self.fc_layers.append(nn.Linear(1, fc_layer_size))
        for _ in range(1, num_layers):
            self.fc_layers.append(nn.Linear(fc_layer_size, fc_layer_size))
        self.fc_mean = nn.Linear(fc_layer_size, dim_mean)
        self.fc_chol = nn.Linear(fc_layer_size, dim_chol)

        self.diag_activation = nn.Softplus()
        self.diag_activation_inv = inverse_softplus
        self.init_std = ch.tensor(3.5)  # 1 for gmm target, 3.5 for banana target
        self.minimal_std = 1e-3

        if init_bias_mean is not None:
            with ch.no_grad():
                self.fc_mean.bias.copy_(ch.tensor(init_bias_mean, dtype=self.fc_mean.bias.dtype))

        if init_bias_chol is not None:
            with ch.no_grad():
                self.fc_chol.bias.copy_(ch.tensor(init_bias_chol, dtype=self.fc_chol.bias.dtype))

    def forward(self, x):
        for fc in self.fc_layers:
            x = ch.relu(fc(x))
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


class ConditionalGMM(AbstractGMM, nn.Module):
    def __init__(self,
                 fc_layer_size,
                 n_components,
                 dim,
                 init_bias_gate=None,
                 init_bias_mean_list=None,
                 init_bias_chol_list=None):
        super(ConditionalGMM, self).__init__()
        self.gate = GateNN(n_components, init_bias_gate=init_bias_gate)
        self.gaussian_list = nn.ModuleList()
        self.dim_mean = dim
        self.dim_chol = (1 + dim) * dim // 2

        for i in range(n_components):
            init_bias_mean = init_bias_mean_list[i] if init_bias_mean_list is not None else None
            init_bias_chol = init_bias_chol_list[i] if init_bias_chol_list is not None else None
            gaussian = GaussianNN(fc_layer_size,
                                  dim_mean=self.dim_mean,
                                  dim_chol=self.dim_chol,
                                  init_bias_mean=init_bias_mean,
                                  init_bias_chol=init_bias_chol)
            self.gaussian_list.append(gaussian)

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

    # def covariance(self, chol):
    #     cov_matrix = chol @ chol.transpose(-1, -2)
    #     return cov_matrix
    #
    # def get_rsamples(self, mean, chol, n_samples):
    #     rsamples = MultivariateNormal(loc=mean, scale_tril=chol).rsample(ch.Size([n_samples]))
    #     return rsamples.transpose(0, 1)
    #
    # def get_samples_gmm(self, log_gates, means, chols, num_samples):
    #     if log_gates.shape[1] == 1:
    #         # print("target has only one component")
    #         samples = MultivariateNormal(means.squeeze(1), scale_tril=chols.squeeze(1)).sample((num_samples,))
    #         return samples.transpose(0, 1)
    #     else:
    #         samples = []
    #         for i in range(log_gates.shape[0]):
    #             cat = Categorical(ch.exp(log_gates[i]))
    #             indices = cat.sample((num_samples,))
    #             chosen_means = means[i, indices]
    #             chosen_chols = chols[i, indices]
    #             normal = MultivariateNormal(chosen_means, scale_tril=chosen_chols)
    #             samples.append(normal.sample())
    #         return ch.stack(samples)  # [n_contexts, n_samples, n_features]
    #
    # def log_prob(self, mean, chol, samples):
    #     if samples.dim() == 3:
    #         batch_size, n_samples, _ = samples.shape
    #         mean_expanded = mean.unsqueeze(1).expand(-1, n_samples, -1)  # [batch_size, n_samples, n_features]
    #         chol_expanded = chol.unsqueeze(1).expand(-1, n_samples, -1, -1)
    #
    #         mvn = MultivariateNormal(loc=mean_expanded, scale_tril=chol_expanded)
    #         log_probs = mvn.log_prob(samples)  # [batch_size, n_samples]
    #         # return log_probs.mean(dim=1)  # [batch_size]
    #         return log_probs
    #     else:
    #         raise ValueError("Shape of samples should be [batch_size, n_samples, n_features]")
    #
    # def log_prob_gmm(self, means, chols, log_gates, samples):
    #     n_samples = samples.shape[1]
    #     n_contexts, n_components, _ = means.shape
    #
    #     means_expanded = means.unsqueeze(1).expand(-1, n_samples, -1, -1)
    #     chols_expanded = chols.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
    #     samples_expanded = samples.unsqueeze(2).expand(-1, -1, n_components, -1)
    #
    #     # since I only plot 2D, I need to modify here into right shape. This if only happens in plotting
    #     if means_expanded.shape[-1] != samples_expanded.shape[-1]:
    #         mvn = MultivariateNormal(means_expanded[..., :2], scale_tril=chols_expanded[..., :2, :2])
    #         log_probs = mvn.log_prob(samples_expanded)
    #     else:
    #         mvn = MultivariateNormal(means_expanded, scale_tril=chols_expanded)
    #         log_probs = mvn.log_prob(samples_expanded)  # [batch_size, n_samples, n_components]
    #
    #     gate_expanded = log_gates.unsqueeze(1).expand(-1, n_samples, -1)
    #     log_probs += gate_expanded
    #
    #     log_probs = ch.logsumexp(log_probs, dim=2)  # [batch_size, n_samples]
    #     # return ch.sum(log_probs, dim=1)
    #     return log_probs
    #
    # def auxiliary_reward(self, j, gate_old, mean_old, chol_old, samples):
    #     gate_old_expanded = gate_old.unsqueeze(1).expand(-1, samples.shape[1], -1)
    #     numerator = gate_old_expanded[:, :, j] + self.log_prob(mean_old[:, j], chol_old[:, j], samples)
    #     denominator = self.log_prob_gmm(mean_old, chol_old, gate_old, samples)
    #     aux_reward = numerator - denominator
    #     return aux_reward
