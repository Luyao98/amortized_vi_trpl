import torch as ch
from torch.distributions import uniform, Normal, MultivariateNormal
import matplotlib.pyplot as plt

from toy_task.GMM.targets.abstract_target import AbstractTarget


class FunnelTarget(AbstractTarget, ch.nn.Module):
    def __init__(self, sig_fn, dim):
        super().__init__()
        self.dim = dim
        self.sig = sig_fn
        self.context_dist = uniform.Uniform(-1, 1)

    def get_contexts(self,
                     n_contexts: int
                     ) -> ch.Tensor:
        """
        Generates a set of contexts for the funnel.

        Parameters:
        - n_contexts (int): Number of contexts to generate.

        Returns:
        - ch.Tensor: shape (n_contexts, context_dim).
        """
        size = ch.Size([n_contexts, self.context_dim])
        contexts = self.context_dist.sample(size)
        return contexts

    def sample(self, contexts, n_samples):
        n_contexts = contexts.shape[0]
        sigs = self.sig(contexts).to(contexts.device)

        samples_v = Normal(loc=ch.zeros(n_contexts).to(contexts.device), scale=sigs.squeeze(-1)).sample((n_samples,))
        samples_v = samples_v.unsqueeze(-1)  # (S, C, 1)
        other_dim = self.dim - 1
        variance_other = ch.exp(samples_v).expand(-1, -1, other_dim)  # [S, C, 9]
        cov_other = ch.diag_embed(variance_other)
        mean_other = ch.zeros_like(variance_other).to(contexts.device)
        samples_other = MultivariateNormal(loc=mean_other, covariance_matrix=cov_other).sample()  # [S, C, 9]
        full_samples = ch.cat((samples_v, samples_other), dim=-1)  # (S, C, 10)
        return full_samples

    def log_prob_tgt(self, contexts, samples):
        n_contexts = contexts.shape[0]
        sigs = self.sig(contexts).to(contexts.device)

        if samples.dim() == 2:
            # for plotting
            if samples.shape[-1] == 2:
                # samples = ch.cat([samples, ch.zeros(samples.shape[0], self.dim - 2)], dim=1)
                samples = samples.unsqueeze(0).expand(n_contexts, -1, -1)
                v = samples[:, :, 0]
                log_density_v = Normal(loc=ch.zeros(n_contexts, 1).to(contexts.device), scale=sigs).log_prob(
                    v)  # [n_contexts, n_samples]
                other_dim = 1
                variance_other = ch.exp(v).unsqueeze(-1).expand(-1, -1, other_dim)
                cov_other = ch.diag_embed(variance_other)
                mean_other = ch.zeros(n_contexts, samples.shape[1], other_dim).to(
                    contexts.device)  # [n_contexts, n_samples, other_dim]
                log_density_other = MultivariateNormal(loc=mean_other, covariance_matrix=cov_other).log_prob(
                    samples[:, :, 1:])
                log_prob = log_density_v + log_density_other
        else:
            v = samples[:, :, 0]
            log_density_v = Normal(loc=ch.zeros(n_contexts, 1).to(contexts.device), scale=sigs).log_prob(v)  # [n_contexts, n_samples]
            other_dim = self.dim - 1
            variance_other = ch.exp(v).unsqueeze(-1).expand(-1, -1, other_dim)
            cov_other = ch.diag_embed(variance_other)
            mean_other = ch.zeros(n_contexts, samples.shape[1], other_dim).to(contexts.device)  # [n_contexts, n_samples, other_dim]
            log_density_other = MultivariateNormal(loc=mean_other, covariance_matrix=cov_other).log_prob(samples[:, :, 1:])
            log_prob = log_density_v + log_density_other
        return log_prob

    def visualize(self, contexts, n_samples=None):
        fig, axes = plt.subplots(1, contexts.shape[0], figsize=(5 * contexts.shape[0], 5))
        for i, c in enumerate(contexts):
            v_range = ch.linspace(-2.5 * 3, 2.5 * 3, 100)
            other_range = ch.linspace(-10, 10, 100)

            V, O = ch.meshgrid(v_range, other_range, indexing='ij')
            samples = ch.stack([V, O], dim=-1).view(-1, 2)
            log_probs = self.log_prob_tgt(c.unsqueeze(1), samples).squeeze(0).view(100, 100)

            probs = ch.exp(log_probs)

            ax = axes[i]
            ax.contourf(V.numpy(), O.numpy(), probs.numpy(), levels=50, cmap='viridis')
            if n_samples is not None:
                samples = self.sample(c.unsqueeze(1), n_samples)
                ax.scatter(samples[..., 0], samples[..., 1], color='red', alpha=0.5)
            ax.axis("scaled")
            ax.set_title("Funnel Distribution")
            ax.set_xlabel("$v$")
            ax.set_ylabel("$x[0]$")
            ax.grid(True)
        plt.tight_layout()
        plt.show()


def get_sig_fn(contexts):
    sig = ch.sin(3 * contexts+1) + 1.1
    return sig


# # test
# target = FunnelTarget(get_sig_fn, dim=10)
# # contexts_test = target.get_contexts(3)
# contexts_test = ch.tensor([[-0.3],
#                            [0.1],
#                            [-0.9]])
# target.visualize(contexts_test, 100)
# # s = target.sample(contexts_test, 100)
