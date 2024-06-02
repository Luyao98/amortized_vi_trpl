import torch as ch
import torch.nn as nn

from toy_task.GMM.models.abstract_gmm_model import AbstractGMM
from toy_task.GMM.utils.torch_utils import diag_bijector, fill_triangular, inverse_softplus


class GateNN(nn.Module):
    def __init__(self, n_components, gate_size, num_layers, init_bias_gate=None):  # gmm target:2, others:5
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
    def __init__(self, fc_layer_size, num_layers, dim_mean, dim_chol, init_bias_mean=None, init_bias_chol=None): # gmm target:3, others:8
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
        self.init_std = ch.tensor(3.5)
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
                 num_layers_gate,
                 gate_size,
                 num_layers_gaussian,
                 gaussian_size,
                 n_components,
                 dim,
                 init_bias_gate=None,
                 init_bias_mean_list=None,
                 init_bias_chol_list=None):
        super(ConditionalGMM, self).__init__()
        self.gate = GateNN(n_components, gate_size, num_layers_gate, init_bias_gate=init_bias_gate)
        self.gaussian_list = nn.ModuleList()
        self.dim_mean = dim
        self.dim_chol = (1 + dim) * dim // 2

        for i in range(n_components):
            init_bias_mean = init_bias_mean_list[i] if init_bias_mean_list is not None else None
            init_bias_chol = init_bias_chol_list[i] if init_bias_chol_list is not None else None
            gaussian = GaussianNN(gaussian_size,
                                  num_layers_gaussian,
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
