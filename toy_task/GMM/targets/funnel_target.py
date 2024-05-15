import torch as ch
from torch.distributions import uniform, Normal, MultivariateNormal
import matplotlib.pyplot as plt


class FunnelTarget(ch.nn.Module):
    def __init__(self, sig_fn, dim=3):
        super().__init__()
        self.dim = dim
        self.sig = sig_fn
        self.context_dist = uniform.Uniform(-1, 1)

    def get_contexts(self, n_contexts):
        contexts = self.context_dist.sample((n_contexts, 1))  # return shape(n_contexts, 1)
        return contexts

    def sample(self, contexts, n_samples):
        n_contexts = contexts.shape[0]
        sigs = self.sig(contexts).to(contexts.device)
        samples = []

        for i in range(n_contexts):
            v_samples = Normal(loc=0., scale=sigs[i]).sample((n_samples,))

            variance_other = ch.exp(v_samples)
            other_dim = self.dim - 1

            # For each sample of 'v', sample the remaining dimensions from their respective normal distributions
            other_samples = []
            for var in variance_other:
                cov_other = ch.eye(other_dim, device=contexts.device) * (var + 1e-8)  # for numerical stability
                normal_dist = MultivariateNormal(loc=ch.zeros(other_dim, device=contexts.device), covariance_matrix=cov_other)
                other_samples.append(normal_dist.sample())

            # Stack and concatenate the samples from 'v' and other dimensions
            other_samples = ch.stack(other_samples)
            full_samples = ch.cat((v_samples.unsqueeze(-1), other_samples), dim=-1)
            samples.append(full_samples)

        return ch.stack(samples)  # [n_contexts, n_samples, dim]

    def log_prob_tgt(self, contexts, samples):
        n_contexts = contexts.shape[0]
        sigs = self.sig(contexts).to(contexts.device)
        means = ch.zeros(n_contexts).to(contexts.device)

        if samples.dim() == 2:
            samples = samples.unsqueeze(0).expand(n_contexts, -1, -1)
        if samples.shape[-1] == self.dim - 1:
            sample_v = Normal(loc=means, scale=sigs).sample((samples.shape[1],))
            sample_v = sample_v.transpose(0, 1).unsqueeze(-1)
            samples = ch.cat([sample_v, samples], dim=-1)

        log_prob = []
        for i in range(samples.shape[1]):
            x = samples[:, i]
            v = x[:, 0]
            log_density_v = Normal(loc=0., scale=sigs[i]).log_prob(v)
            variance_other = ch.exp(v)
            other_dim = self.dim - 1
            cov_other = ch.eye(other_dim, device=contexts.device).unsqueeze(0).repeat(x.shape[0], 1, 1) * (variance_other.view(-1, 1, 1) + 1e-8)
            mean_other = ch.zeros(other_dim, device=contexts.device).unsqueeze(0).repeat(x.shape[0], 1)
            log_density_other = MultivariateNormal(loc=mean_other, covariance_matrix=cov_other).log_prob(x[:, 1:])
            log_prob.append(log_density_v + log_density_other)
        log_prob_tgt = ch.stack(log_prob, dim=1)
        return log_prob_tgt

    def visualize(self, contexts, n_samples=None):
        fig, axes = plt.subplots(1, contexts.shape[0], figsize=(5 * contexts.shape[0], 5))
        for i, c in enumerate(contexts):
            v_range = ch.linspace(-2.5 * 3, 2.5 * 3, 100)
            other_range = ch.linspace(-10, 10, 100)

            V, O = ch.meshgrid(v_range, other_range, indexing='ij')
            samples = ch.stack([V, O], dim=-1).view(-1, 2)
            samples = ch.cat([samples, ch.zeros(samples.shape[0], self.dim-2)], dim=1)  # Pad with zeros for unused dimensions

            # Compute log probabilities
            log_probs = self.log_prob_tgt(c.unsqueeze(1), samples).squeeze(0).view(100, 100)

            # Convert log probabilities to probabilities for better visualization
            probs = ch.exp(log_probs)

            ax = axes[i]
            ax.contourf(V.numpy(), O.numpy(), probs.numpy(), levels=50, cmap='viridis')
            ax.axis("scaled")
            ax.set_title("Funnel Distribution")
            ax.set_xlabel("$v$")
            ax.set_ylabel("$x[0]$")
            ax.grid(True)
        plt.tight_layout()
        plt.show()


def get_sig_fn(c):
    sig = ch.exp(c[:, 0])
    return sig


# target = FunnelTarget(get_sig_fn)
# c = target.get_contexts(2)
# target.visualize(c)
