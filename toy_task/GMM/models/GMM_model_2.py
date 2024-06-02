import torch as ch
import torch.nn as nn

from toy_task.GMM.models.abstract_gmm_model import AbstractGMM
from toy_task.GMM.utils.torch_utils import fill_triangular_gmm


class GateNN(nn.Module):
    def __init__(self, n_components, num_layers, gate_size, init_bias_gate=None):
        super(GateNN, self).__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList()

        self.fc_layers.append(nn.Linear(1, gate_size))
        for _ in range(1, num_layers):
            self.fc_layers.append(nn.Linear(gate_size, gate_size))
        self.fc_gate = nn.Linear(gate_size, n_components)

        # set init uniform bias for gates
        if init_bias_gate is not None:
            with ch.no_grad():
                self.fc_gate.bias.copy_(ch.tensor(init_bias_gate, dtype=self.fc_gate.bias.dtype))

    def forward(self, x):
        for fc in self.fc_layers:
            x = ch.relu(fc(x))
        x = ch.log_softmax(self.fc_gate(x), dim=-1)
        return x


class GaussianNN2(nn.Module):
    def __init__(self,
                 num_layers,
                 gaussian_size,
                 n_components,
                 dim,
                 init_bias_mean=None
                 ):
        super(GaussianNN2, self).__init__()
        self.num_layers = num_layers
        self.layer_size = gaussian_size
        self.n_components = n_components
        self.dim = dim
        self.mean_dim = n_components * dim
        self.chol_dim = n_components * dim * (dim + 1) // 2

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(1, gaussian_size))
        for _ in range(1, num_layers):
            self.fc_layers.append(nn.Linear(gaussian_size, gaussian_size))
        self.fc_mean = nn.Linear(gaussian_size, self.mean_dim)
        self.fc_chol = nn.Linear(gaussian_size, self.chol_dim)

        # init model output
        if init_bias_mean is not None:
            with ch.no_grad():
                self.fc_mean.bias.copy_(ch.tensor(init_bias_mean, dtype=self.fc_mean.bias.dtype))
        self.init_std = ch.tensor(3.5)

    def forward(self, x):
        for fc in self.fc_layers:
            x = ch.relu(fc(x))
        flat_means = self.fc_mean(x)
        flat_chols = self.fc_chol(x)
        means = flat_means.view(-1, self.n_components, self.dim)
        chols_reshape = flat_chols.view(-1, self.n_components, self.dim * (self.dim + 1) // 2)
        chols = fill_triangular_gmm(chols_reshape, self.n_components, self.init_std)
        return means, chols


class ConditionalGMM2(AbstractGMM, nn.Module):
    def __init__(self,
                 num_layers_gate,
                 gate_size,
                 num_layers_gaussian,
                 gaussian_size,
                 n_components,
                 dim,
                 init_bias_gate=None,
                 init_bias_mean=None
                 ):
        super(ConditionalGMM2, self).__init__()
        self.gate = GateNN(n_components, num_layers_gate, gate_size, init_bias_gate)
        self.gaussian_list = GaussianNN2(num_layers_gaussian, gaussian_size, n_components, dim, init_bias_mean)

    def forward(self, x):
        log_gates = self.gate(x)
        means, chols = self.gaussian_list(x)
        return log_gates, means, chols
