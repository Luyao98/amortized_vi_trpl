import torch as ch
import torch.nn as nn

from toy_task.GMM.models.abstract_gmm_model import AbstractGMM
from toy_task.GMM.utils.torch_utils import fill_triangular_gmm


def embedding(contexts, n_components):
    batch_size, feature_size = contexts.shape
    context_reshape = contexts.unsqueeze(1).repeat(1, n_components, 1)  # (b, n_com, f)
    eye = ch.eye(n_components).unsqueeze(0).repeat(batch_size, 1, 1).to(contexts.device)
    new_contexts = ch.cat([context_reshape, eye], dim=-1)
    return new_contexts.view(-1, feature_size + n_components)  # (b*n_com, f+n_com)


class GateNN3(nn.Module):
    def __init__(self, n_components, num_layers, gate_size, dropout_prob, init_bias_gate=None):
        super(GateNN3, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(1, gate_size))
        for _ in range(1, num_layers):
            self.fc_layers.append(nn.Linear(gate_size, gate_size))
        self.fc_gate = nn.Linear(gate_size, n_components)

        # set initial uniform bias for gates if provided
        if init_bias_gate is not None:
            with ch.no_grad():
                self.fc_gate.bias.copy_(ch.tensor(init_bias_gate, dtype=self.fc_gate.bias.dtype))

    def forward(self, x):
        for fc in self.fc_layers:
            x = ch.relu(fc(x))
            # x = self.dropout(x)

        return self.fc_gate(x)


class GaussianNN3(nn.Module):
    def __init__(self, num_layers, gaussian_size, n_components, dim, dropout_prob, init_bias_mean=None):
        super(GaussianNN3, self).__init__()
        self.n_components = n_components
        self.mean_dim = dim
        self.chol_dim = dim * (dim + 1) // 2
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(1 + n_components, gaussian_size))
        for _ in range(1, num_layers):
            self.fc_layers.append(nn.Linear(gaussian_size, gaussian_size))

        # Output layers
        self.fc_mean = nn.Linear(gaussian_size, self.mean_dim)
        self.fc_chol = nn.Linear(gaussian_size, self.chol_dim)

        # Initialize mean bias if provided
        if init_bias_mean is not None:
            with ch.no_grad():
                self.fc_mean.bias.copy_(ch.tensor(init_bias_mean, dtype=self.fc_mean.bias.dtype))

        self.init_std = ch.tensor(10)

    def forward(self, x):
        for fc in self.fc_layers:
            x = ch.relu(fc(x))
            # x = self.dropout(x)
        flat_means = self.fc_mean(x)
        flat_chols = self.fc_chol(x)
        means = flat_means.view(-1, self.n_components, self.mean_dim)
        chols_reshape = flat_chols.view(-1, self.n_components, self.chol_dim)
        chols = fill_triangular_gmm(chols_reshape, self.n_components, self.init_std)
        return means, chols


class ConditionalGMM3(AbstractGMM, nn.Module):
    def __init__(self, num_layers_gate, gate_size, num_layers_gaussian, gaussian_size, max_components, init_components,
                 dim, init_bias_gate=None, init_bias_mean=None, dropout_prob=0.0):
        super(ConditionalGMM3, self).__init__()
        self.dim = dim
        self.max_components = max_components
        self.active_component_indices = list(range(init_components))

        self.embedded_mean_bias = nn.Parameter(ch.zeros((max_components, dim)), requires_grad=False)
        self.embedded_chol_bias = nn.Parameter(ch.zeros((max_components, dim, dim)), requires_grad=False)

        self.gate = GateNN3(max_components, num_layers_gate, gate_size, dropout_prob, init_bias_gate)
        self.gaussian_list = GaussianNN3(num_layers_gaussian, gaussian_size, max_components, dim, dropout_prob,
                                         init_bias_mean)

    def forward(self, x):
        gates = self.gate(x)  # (b, n_com)

        active_gates = gates[:, self.active_component_indices]
        log_gates = ch.log_softmax(active_gates, dim=-1)

        x = embedding(x, self.max_components)  # (b*n_com, f+n_com)
        means, chols = self.gaussian_list(x)

        active_means = means[:, self.active_component_indices] + self.embedded_mean_bias[self.active_component_indices]
        active_chols = chols[:, self.active_component_indices] + self.embedded_chol_bias[self.active_component_indices]

        return log_gates, active_means, active_chols

    def add_component(self, component_index):
        if component_index not in self.active_component_indices:
            self.active_component_indices.append(component_index)
            self.active_component_indices.sort()
        else:
            raise ValueError(f"Component index {component_index} is already active.")
