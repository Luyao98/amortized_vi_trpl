import torch as ch
import torch.nn as nn

from toy_task.GMM.models.abstract_gmm_model import AbstractGMM
from toy_task.GMM.utils.torch_utils import diag_bijector, fill_triangular, inverse_softplus


class GateNN(nn.Module):
    def __init__(self, in_size, n_components, num_layers, num_neuron_gate_layer, init_bias_gate=None):
        super(GateNN, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(in_size, num_neuron_gate_layer))
        for _ in range(1, num_layers):
            self.fc_layers.append(nn.Linear(num_neuron_gate_layer, num_neuron_gate_layer))
        self.fc_gate = nn.Linear(num_neuron_gate_layer, n_components)

        # set initial uniform bias for gates if provided
        if init_bias_gate is not None:
            with ch.no_grad():
                self.fc_gate.bias.copy_(ch.tensor(init_bias_gate, dtype=self.fc_gate.bias.dtype))

    def forward(self, x):
        for fc in self.fc_layers:
            x = ch.relu(fc(x))

        return self.fc_gate(x)


class GaussianNN(nn.Module):
    def __init__(self, in_size, num_layers, num_neuron_component_layer, dim, init_bias_mean=None):
        super(GaussianNN, self).__init__()
        self.mean_dim = dim
        self.chol_dim = dim * (dim + 1) // 2
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(in_size, num_neuron_component_layer))
        for _ in range(1, num_layers):
            self.fc_layers.append(nn.Linear(num_neuron_component_layer, num_neuron_component_layer))
        self.fc_mean = nn.Linear(num_neuron_component_layer, self.mean_dim)
        self.fc_chol = nn.Linear(num_neuron_component_layer, self.chol_dim)

        self.diag_activation = nn.Softplus()
        self.diag_activation_inv = inverse_softplus
        self.init_std = ch.tensor(1.0)
        self.minimal_std = 1e-3

        if init_bias_mean is not None:
            with ch.no_grad():
                self.fc_mean.bias.copy_(init_bias_mean.clone().detach())

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
                 num_gate_layer,
                 num_neuron_gate_layer,
                 num_component_layer,
                 num_neuron_component_layer,
                 n_components,
                 dim,
                 context_dim,
                 random_init,
                 init_scale,
                 init_bias_gate=None):
        super(ConditionalGMM, self).__init__()
        self.dim = dim
        self.init_bias_mean = ch.zeros(dim)
        self.active_component_indices = list(range(n_components))
        self.previous_deleted_indices = None
        self.delete_counter = 0
        if random_init:
            mean_bias = 2 * init_scale * ch.rand(n_components, dim) - init_scale
            self.embedded_mean_bias = nn.Parameter(mean_bias, requires_grad=False)
        else:
            self.embedded_mean_bias = nn.Parameter(ch.zeros((n_components, dim)), requires_grad=False)
        self.embedded_chol_bias = nn.Parameter(ch.zeros((n_components, dim, dim)), requires_grad=False)

        self.gate = GateNN(context_dim, n_components, num_gate_layer, num_neuron_gate_layer, init_bias_gate)
        self.gaussian_list = nn.ModuleList()
        for i in range(n_components):
            gaussian = GaussianNN(context_dim, num_component_layer, num_neuron_component_layer, dim, self.init_bias_mean)
            self.gaussian_list.append(gaussian)

    def forward(self, x):
        # p(o|c)
        gate = self.gate(x)
        log_gates = ch.log_softmax(gate, dim=1)
        # p(x|o,c)
        means, chols = [], []
        for gaussian in self.gaussian_list:
            mean, chol = gaussian(x)
            means.append(mean)
            chols.append(chol)
        active_means = ch.stack(means, dim=1) + self.embedded_mean_bias
        active_chols = ch.stack(chols, dim=1) + self.embedded_chol_bias
        return log_gates, active_means, active_chols
