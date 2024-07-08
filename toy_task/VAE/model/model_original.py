import torch as ch
import torch.nn as nn
from torch.distributions import MultivariateNormal, kl_divergence

from toy_task.GMM.models.abstract_gmm_model import AbstractGMM
from toy_task.GMM.projections.split_kl_projection import split_kl_projection


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
        x = ch.softmax(self.fc_gate(x), dim=-1)
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
        self.logvar = nn.Linear(gaussian_size, self.mean_dim)
        self.fc0 = nn.Linear(gaussian_size, dim)

    def forward(self, x):
        for fc in self.fc_layers:
            x = ch.relu(fc(x))
        means = self.fc_mean(x).view(-1, self.n_components, self.dim)
        # flat_chols = ch.exp(self.logvar(x)).view(-1, self.n_components, self.dim)
        # chols = ch.diag_embed(flat_chols)
        logvar = self.logvar(x).view(-1, self.n_components, self.dim)

        return means, logvar


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
        gates = self.gate(x)
        means, logvar = self.gaussian_list(x)
        return gates, means, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        # self.fc0 = nn.Linear(2, latent_dim)
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(latent_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        # self.fc_logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        for fc in self.fc_layers:
            z = ch.relu(fc(z))
        recon_mean = ch.sigmoid(self.fc_mu(z))
        # recon_logvar = self.fc_logvar(h)
        return recon_mean


class VAE_original(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_components, encoder_layer_1, encoder_layer_2, decoder_layer,
                 n_samples=1, projection=False, eps_means=None, eps_chols=None, alpha=None):
        super(VAE_original, self).__init__()
        self.encoder = ConditionalGMM2(num_layers_gate=encoder_layer_1,
                                       gate_size=hidden_dim,
                                       num_layers_gaussian=encoder_layer_2,
                                       gaussian_size=hidden_dim,
                                       n_components=n_components,
                                       dim=latent_dim,
                                       init_bias_gate=[0.0] * n_components)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, decoder_layer)
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.n_samples = n_samples

        self.projection = projection
        self.eps_means = eps_means
        self.eps_chols = eps_chols
        self.alpha = alpha
        self.old_means = None
        self.old_chols = None

    def forward(self, x, temperature=1.0, hard=False):
        # encoder
        gates, means, chols = self.encoder(x)
        gates = self.gumbel_softmax(gates, temperature, hard)
        log_gates = ch.log(gates)
        if self.projection:
            if self.old_means is None:
                means_proj = means
                chols_proj = chols
            else:
                means_proj, chols_proj = self.projection_layer(means, chols, self.eps_means, self.eps_chols)
            pred_dist = MultivariateNormal(means, scale_tril=chols)
            proj_dist = MultivariateNormal(means_proj.clone().detach(), scale_tril=chols_proj.clone().detach())
            regression = kl_divergence(proj_dist, pred_dist)

            z = self.encoder.get_rsamples(means_proj, chols_proj, self.n_samples).permute(0, 2, 1, 3)  # (b,o,s=1,l)
            log_responsibility = self.encoder.log_responsibility(log_gates.clone().detach(),
                                                                 means_proj.clone().detach(),
                                                                 chols_proj.clone().detach(), z)  # (b,o,s=1)
            log_responsibility = log_responsibility - self.alpha * regression.unsqueeze(-1)  # (b,o,s=1)
            # kld = self.kld(means_proj, chols_proj)  # (b,o)
        else:
            # z = self.encoder.get_rsamples(means, chols, self.n_samples).permute(0, 2, 1, 3) # (b,o,s=1,l)
            z = ch.distributions.Normal(means, ch.exp(chols)).rsample() # (b,o,l)
            # log_z = self.encoder.log_prob(means, chols, z) # (b,o,s=1)
            # log_responsibility = self.encoder.log_responsibility(log_gates.clone().detach(),
            #                                                      means.clone().detach(),
            #                                                      chols.clone().detach(), z) # (b,o,s=1)
            log_responsibility = None
            # kld = self.kld(means, chols) # (b,o)
        # test
        recon_loss = lambda recon_x, x: nn.functional.binary_cross_entropy(recon_x, x, reduction="none")

        recon_mean = [self.decoder(z[:,i]) for i in range(self.n_components)]
        recon_mean = ch.stack(recon_mean, dim=1) # (b,o,f)

        kld = self.kld_test(means, chols)
        likelihood = [recon_loss(recon_mean[:,i], x) for i in range(self.n_components)]
        likelihood = ch.stack(likelihood, dim=1).sum(-1)

        return  log_gates, z, log_responsibility, kld, likelihood, recon_mean

    def kld(self, mean, chol):
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

    def kld_test(self,mean, logvar):
        # var = ch.diagonal(chol @ chol.transpose(-1, -2), dim1=-2, dim2=-1)
        kld = -0.5 * ch.sum(1 + logvar - mean.pow(2) - ch.exp(logvar), dim=-1)
        return kld

    def projection_layer(self, means, chols, eps_means, eps_chols):
        proj_dist = [split_kl_projection(means[:, j], chols[:, j],
                                         self.old_means[:, j].clone().detach(),
                                         self.old_chols[:, j].clone().detach(),
                                         eps_means, eps_chols) for j in range(self.n_components)]
        mean_proj, chol_proj = zip(*proj_dist)
        return ch.stack(mean_proj, dim=1), ch.stack(chol_proj, dim=1)

    def gumbel_softmax_sample(self, logits, temperature=1.0):
        gumbel_noise = -ch.log(-ch.log(ch.rand_like(logits) + 1e-20) + 1e-20)
        y = logits + gumbel_noise
        return ch.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature=1.0, hard=False):
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = ch.zeros_like(y)
            y_hard.scatter_(1, y.argmax(dim=1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y
        return y
