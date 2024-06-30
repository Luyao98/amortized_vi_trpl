import torch as ch
import torch.nn as nn
from torch.distributions import Normal

# from toy_task.GMM.models.GMM_model_2 import ConditionalGMM2
from toy_task.GMM.models.abstract_gmm_model import AbstractGMM

class GateNN(nn.Module):
    def __init__(self, n_components, num_layers, gate_size, init_bias_gate=None):
        super(GateNN, self).__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList()

        self.fc_layers.append(nn.Linear(784, gate_size))
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
                 dim):
        super(GaussianNN2, self).__init__()
        self.num_layers = num_layers
        self.layer_size = gaussian_size
        self.n_components = n_components
        self.dim = dim
        self.mean_dim = n_components * dim
        # self.chol_dim = n_components * dim * (dim + 1) // 2

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(784, gaussian_size))
        for _ in range(1, num_layers):
            self.fc_layers.append(nn.Linear(gaussian_size, gaussian_size))
        self.fc_mean = nn.Linear(gaussian_size, self.mean_dim)
        self.fc_chol = nn.Linear(gaussian_size, self.mean_dim)

    def forward(self, x):
        for fc in self.fc_layers:
            x = ch.relu(fc(x))
        means = self.fc_mean(x).view(-1, self.n_components, self.dim)
        flat_chols = ch.exp(self.fc_chol(x)).view(-1, self.n_components, self.dim)
        chols = ch.diag_embed(flat_chols)

        return means, chols


class ConditionalGMM2(AbstractGMM, nn.Module):
    def __init__(self,
                 num_layers_gate,
                 gate_size,
                 num_layers_gaussian,
                 gaussian_size,
                 n_components,
                 dim,
                 init_bias_gate=None):
        super(ConditionalGMM2, self).__init__()
        self.gate = GateNN(n_components, num_layers_gate, gate_size, init_bias_gate)
        self.gaussian_list = GaussianNN2(num_layers_gaussian, gaussian_size, n_components, dim)

    def forward(self, x):
        log_gates = self.gate(x)
        means, chols = self.gaussian_list(x)
        return log_gates, means, chols

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        # self.fc_logvar = nn.Linear(hidden_dim, output_dim)


    def forward(self, z):
        h = ch.relu(self.fc1(z))
        # h = ch.relu(self.fc2(h))
        # h = ch.relu(self.fc3(h))
        recon_mean = ch.sigmoid(self.fc_mu(h))
        # recon_logvar = self.fc_logvar(h)
        return recon_mean


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_components, n_samples=1, projection=False):
        super(VAE, self).__init__()
        self.encoder = ConditionalGMM2(num_layers_gate=1,
                                       gate_size=hidden_dim,
                                       num_layers_gaussian=2,
                                       gaussian_size=hidden_dim,
                                       n_components=n_components,
                                       dim=latent_dim,
                                       init_bias_gate=[0.0] * n_components)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.n_samples = n_samples
        self.projection = projection

    def forward(self, x):
        # encoder
        log_gates, means, chols = self.encoder(x)
        if self.projection:
            pass
        z = self.encoder.get_rsamples(means, chols, self.n_samples).permute(0, 2, 1, 3) # (b,o,s=1,l)
        # log_z = self.encoder.log_prob(means, chols, z) # (b,o,s=1)
        log_responsibility = self.encoder.log_responsibility(log_gates.clone().detach(), means.clone().detach(),
                                                        chols.clone().detach(), z) # (b,o,s=1)
        kld = self.kl_divergence(means, chols) # (b,o)
        # # test
        # covs = ch.diagonal(chols @ chols.transpose(-1, -2), dim1=-2, dim2=-1) # (b,o,l)
        # kl_loss = lambda means, covs: -0.5 * ch.sum(1 + means - means.pow(2) - covs.exp())
        # recon_loss = lambda recon_x, x: nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')


        # decoder
        # cov != I
        # result = [self.decoder(z[:,i].squeeze(-2)) for i in range(self.n_components)]
        # recon_mean, recon_logvar = zip(*result)
        # likelihood = [Normal(recon_mean[i], ch.exp(recon_logvar[i])).log_prob(x).sum(-1) for i in range(self.n_components)] # (b,o)
        # likelihood = ch.stack(likelihood, dim=1)

        # cov = I
        recon_mean = [self.decoder(z[:,i].squeeze(-2)) for i in range(self.n_components)]
        recon_mean = ch.stack(recon_mean, dim=1) # (b,o,l)
        likelihood = [-0.5 * ((x - recon_mean[:,i]) ** 2).sum(-1) for i in range(self.n_components)]
        # likelihood = [Normal(recon_mean[:,i], ch.ones_like(recon_mean[:,i])).log_prob(x).sum(-1) for i in range(self.n_components)]
        likelihood = ch.stack(likelihood, dim=1) # (b,o)

        # kld =kl_loss(means.squeeze(1), covs.squeeze(1))
        # likelihood = recon_loss(recon_mean.squeeze(1), x)
        # test_loss = -likelihood.sum() + kld.sum()
        return  log_gates, z.squeeze(-2), log_responsibility.squeeze(-1), kld, likelihood, recon_mean

    def kl_divergence(self, mean, chol):
        """
        Compute KL divergence between component and normal Gaussian
        :param mean: (b,o,l)
        :param chol: (b,o,l,l)
        :return:
        """
        batch_size, n_component, latent_dim = mean.shape

        cov = chol @ chol.transpose(-1, -2)
        trace_term = ch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)
        mean_term = (mean ** 2).sum(-1)
        log_det_cov = 2 * ch.log(ch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)

        kl_div = 0.5 * (trace_term + mean_term - latent_dim - log_det_cov)

        return kl_div

    def generate(self, z):
        return self.decoder(z)
