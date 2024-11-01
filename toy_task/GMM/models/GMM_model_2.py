import torch as ch
import torch.nn as nn

from toy_task.GMM.models.abstract_gmm_model import AbstractGMM
from toy_task.GMM.models.GMM_model import GateNN
from toy_task.GMM.utils.torch_utils import fill_triangular_gmm


class GaussianNN2(nn.Module):
    def __init__(self, in_size, num_layers, num_neuron_component_layer, n_components, dim, init_bias_mean=None):
        super(GaussianNN2, self).__init__()
        self.n_components = n_components
        self.mean_dim = dim
        self.chol_dim = dim * (dim + 1) // 2

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(in_size, num_neuron_component_layer))
        for _ in range(1, num_layers):
            self.fc_layers.append(nn.Linear(num_neuron_component_layer, num_neuron_component_layer))
        self.fc_mean = nn.Linear(num_neuron_component_layer, n_components * self.mean_dim)
        self.fc_chol = nn.Linear(num_neuron_component_layer, n_components * self.chol_dim)

        # init model output
        if init_bias_mean is not None:
            with ch.no_grad():
                self.fc_mean.bias.copy_(init_bias_mean.clone().detach())
        self.init_std = ch.tensor(1.0)

    def forward(self, x):
        for fc in self.fc_layers:
            x = ch.relu(fc(x))

        flat_means = self.fc_mean(x)
        flat_chols = self.fc_chol(x)
        means = flat_means.view(-1, self.n_components, self.mean_dim)
        chols_reshape = flat_chols.view(-1, self.n_components, self.chol_dim)
        chols = fill_triangular_gmm(chols_reshape, self.n_components, self.init_std)
        return means, chols


class ConditionalGMM2(AbstractGMM, nn.Module):
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
                 init_bias_gate=None
                 ):
        super(ConditionalGMM2, self).__init__()
        self.dim = dim
        self.init_bias_mean = ch.zeros(n_components * dim)
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
        self.gaussian_list = GaussianNN2(context_dim, num_component_layer, num_neuron_component_layer, n_components,
                                         dim, self.init_bias_mean)


    def forward(self, x):
        gates = self.gate(x)
        log_gates = ch.log_softmax(gates, dim=-1)
        means, chols = self.gaussian_list(x)
        return log_gates, means, chols
