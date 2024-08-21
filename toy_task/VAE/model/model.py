import numpy as np
import torch as ch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal, kl_divergence

from toy_task.GMM.models.GMM_model_2 import ConditionalGMM2
from toy_task.GMM.models.abstract_gmm_model import AbstractGMM
from toy_task.GMM.projections.split_kl_projection import split_kl_projection
from toy_task.GMM.utils.network_utils import generate_init_biases

# class GateNN(nn.Module):
#     def __init__(self, n_components, num_layers, gate_size, init_bias_gate=None):
#         super(GateNN, self).__init__()
#         self.num_layers = num_layers
#         self.fc_layers = nn.ModuleList()
#
#         self.fc_layers.append(nn.Linear(784, gate_size))
#         for _ in range(1, num_layers):
#             self.fc_layers.append(nn.Linear(gate_size, gate_size))
#         self.fc_gate = nn.Linear(gate_size, n_components)
#
#         # set init uniform bias for gates
#         if init_bias_gate is not None:
#             with ch.no_grad():
#                 self.fc_gate.bias.copy_(ch.tensor(init_bias_gate, dtype=self.fc_gate.bias.dtype))
#
#     def forward(self, x):
#         for fc in self.fc_layers:
#             x = ch.relu(fc(x))
#         x = ch.log_softmax(self.fc_gate(x), dim=-1)
#         return x
#
#
# class GaussianNN2(nn.Module):
#     def __init__(self,
#                  num_layers,
#                  gaussian_size,
#                  n_components,
#                  dim):
#         super(GaussianNN2, self).__init__()
#         self.num_layers = num_layers
#         self.layer_size = gaussian_size
#         self.n_components = n_components
#         self.dim = dim
#         self.mean_dim = n_components * dim
#         # self.chol_dim = n_components * dim * (dim + 1) // 2
#
#         self.fc_layers = nn.ModuleList()
#         self.fc_layers.append(nn.Linear(784, gaussian_size))
#         for _ in range(1, num_layers):
#             self.fc_layers.append(nn.Linear(gaussian_size, gaussian_size))
#         self.fc_mean = nn.Linear(gaussian_size, self.mean_dim)
#         self.fc_chol = nn.Linear(gaussian_size, self.mean_dim)
#         self.fc0 = nn.Linear(gaussian_size, dim)
#
#     def forward(self, x):
#         for fc in self.fc_layers:
#             x = ch.relu(fc(x))
#         means = self.fc_mean(x).view(-1, self.n_components, self.dim)
#         flat_chols = ch.exp(self.fc_chol(x)).view(-1, self.n_components, self.dim)
#         chols = ch.diag_embed(flat_chols)
#
#         return means, chols
#
#
# class ConditionalGMM2(AbstractGMM, nn.Module):
#     def __init__(self,
#                  num_layers_gate,
#                  gate_size,
#                  num_layers_gaussian,
#                  gaussian_size,
#                  n_components,
#                  dim,
#                  init_bias_gate=None):
#         super(ConditionalGMM2, self).__init__()
#         self.gate = GateNN(n_components, num_layers_gate, gate_size, init_bias_gate)
#         self.gaussian_list = GaussianNN2(num_layers_gaussian, gaussian_size, n_components, dim)
#
#     def forward(self, x):
#         log_gates = self.gate(x)
#         means, chols = self.gaussian_list(x)
#         return log_gates, means, chols

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(latent_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        # self.fc_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        for fc in self.fc_layers:
            z = ch.nn.functional.leaky_relu(fc(z))
            # z= ch.relu(fc(z))
        recon_mean = ch.sigmoid(self.fc_mu(z))
        # recon_var = ch.exp(self.fc_var(z))
        return recon_mean


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_components, encoder_layer_1, encoder_layer_2, decoder_layer,
                 n_samples, projection=False, eps_means=None, eps_chols=None, alpha=None):
        super(VAE, self).__init__()
        self.encoder = ConditionalGMM2(num_layers_gate=encoder_layer_1,
                                       gate_size=hidden_dim,
                                       num_layers_gaussian=encoder_layer_2,
                                       gaussian_size=hidden_dim,
                                       n_components=n_components,
                                       dim=latent_dim,
                                       init_bias_gate=[0.0] * n_components,
                                       init_bias_mean=np.array(generate_init_biases(n_components, latent_dim, 2.0)).flatten())
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, decoder_layer)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.n_samples = n_samples

        self.projection = projection
        self.eps_means = eps_means
        self.eps_chols = eps_chols
        self.alpha = alpha
        self.old_means = None
        self.old_chols = None

    def forward(self, x):
        # encoder
        log_gates, means, chols = self.encoder(x)
        # log_gates = self.gumbel_softmax(log_gates)
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
                                                                 chols_proj.clone().detach(), z)  # (b,o,s)
            log_responsibility = log_responsibility - self.alpha * regression.unsqueeze(-1)  # (b,o,s)
            kld = self.kld(means_proj, chols_proj)  # (b,o)
        else:
            z = self.encoder.get_rsamples(means, chols, self.n_samples).permute(0, 2, 1, 3) # (b,o,s,l)
            # log_z = self.encoder.log_prob(means, chols, z) # (b,o,s)
            # log_prior = Normal(0, 1).log_prob(z).sum(-1) # (b,o,s)
            log_responsibility = self.encoder.log_responsibility(log_gates.clone().detach(),
                                                                 means.clone().detach(),
                                                                 chols.clone().detach(), z) # (b,o,s)
            kld = self.kld(means, chols) # (b,o)
            # kld = log_z - log_prior

        # decoder
        # cov != I
        # recon_mean, recon_var  = self.decoder(z.reshape(-1, self.latent_dim))
        # recon_mean = recon_mean.reshape(-1, self.n_components, self.n_samples, self.input_dim) # (b,o,s,f)
        # recon_var = recon_var.reshape(-1, self.n_components, self.n_samples, self.input_dim) # (b,o,s,f)
        # likelihood = MultivariateNormal(recon_mean, ch.diag_embed(recon_var)).log_prob(x.unsqueeze(1).unsqueeze(1)) # (b,o,s)

        # cov = I
        recon_mean = self.decoder(z.reshape(-1, self.latent_dim)).reshape(-1, self.n_components, self.n_samples, self.input_dim) # (b,o,s,f)
        likelihood = -0.5 * ((x.unsqueeze(1).unsqueeze(1) - recon_mean) ** 2).sum(-1) # (b,o,s)
        # likelihood = Normal(recon_mean, ch.ones_like(recon_mean)).log_prob(x.unsqueeze(1).unsqueeze(1)).sum(-1)

        # if self.projection:
        #     return  log_gates, means_proj, chols_proj, log_responsibility, kld, likelihood, recon_mean
        # else:
        return log_gates, means, chols, log_responsibility, kld, likelihood, recon_mean

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

    def kld_test(self,mean, chol):
        var = ch.diagonal(chol @ chol.transpose(-1, -2), dim1=-2, dim2=-1)
        kld = -0.5 * ch.sum(1 + ch.log(var) - mean.pow(2) - var, dim=-1)
        return kld

    def projection_layer(self, means, chols, eps_means, eps_chols):
        proj_dist = [split_kl_projection(means[:, j], chols[:, j],
                                         self.old_means[:, j].clone().detach(),
                                         self.old_chols[:, j].clone().detach(),
                                         eps_means, eps_chols) for j in range(self.n_components)]
        mean_proj, chol_proj = zip(*proj_dist)
        return ch.stack(mean_proj, dim=1), ch.stack(chol_proj, dim=1)