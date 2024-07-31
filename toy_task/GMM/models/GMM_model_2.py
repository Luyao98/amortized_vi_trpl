import torch as ch
import torch.nn as nn

from toy_task.GMM.models.abstract_gmm_model import AbstractGMM
from toy_task.GMM.utils.torch_utils import fill_triangular_gmm


class GateNN(nn.Module):
    def __init__(self, n_components, num_layers, gate_size, dropout_prob, init_bias_gate=None):
        super(GateNN, self).__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_prob)

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
            # x = self.dropout(x)
        x = self.fc_gate(x)
        # x = nn.functional.gumbel_softmax(self.fc_gate(x), tau=1.0, hard=False, dim=-1)
        return x


class GaussianNN2(nn.Module):
    def __init__(self,
                 num_layers,
                 gaussian_size,
                 n_components,
                 dim,
                 dropout_prob,
                 init_bias_mean=None
                 ):
        super(GaussianNN2, self).__init__()
        self.num_layers = num_layers
        self.layer_size = gaussian_size
        self.n_components = n_components
        self.dim = dim
        self.mean_dim = n_components * dim
        self.chol_dim = n_components * dim * (dim + 1) // 2
        self.dropout = nn.Dropout(dropout_prob)

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
        # print("Initialized fc_mean.bias:", self.fc_mean.bias)
        self.init_std = ch.tensor(3.5)

    def forward(self, x):
        for fc in self.fc_layers:
            x = ch.relu(fc(x))
            # x = self.dropout(x)
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
                 active_components,
                 dim,
                 init_bias_gate=None,
                 init_bias_mean=None,
                 dropout_prob=0.0
                 ):
        super(ConditionalGMM2, self).__init__()
        self.dim = dim
        self.active_components = active_components
        self.gate = GateNN(n_components, num_layers_gate, gate_size, dropout_prob, init_bias_gate)
        self.gaussian_list = GaussianNN2(num_layers_gaussian, gaussian_size, n_components, dim, dropout_prob, init_bias_mean)

        self.hooks = []

    def forward(self, x):
        gates = self.gate(x)
        log_gates = ch.log_softmax(gates[:,:self.active_components], dim=-1)
        means, chols = self.gaussian_list(x)
        return log_gates, means[:,:self.active_components], chols[:,:self.active_components]

    def register_hooks(self):
        def hook_fn(grad):
            grad_copy = grad.clone()
            # print("grad_copy.shape:", grad_copy.shape)
            grad_copy[:, :self.active_components] = 0
            return grad_copy

        # for param in self.gate.fc_gate.parameters():
        #     hook = param.register_hook(hook_fn)
        #     self.hooks.append(hook)

        for param in self.gaussian_list.fc_mean.parameters():
            if param.ndim > 1 and param.shape[0] > self.active_components:
                hook = param.register_hook(hook_fn)
                self.hooks.append(hook)

        for param in self.gaussian_list.fc_chol.parameters():
            if param.ndim > 1 and param.shape[0] > self.active_components:
                hook = param.register_hook(hook_fn)
                self.hooks.append(hook)

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []