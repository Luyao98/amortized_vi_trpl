from abc import ABC, abstractmethod
import torch as ch
from torch.distributions import MultivariateNormal, Categorical


class AbstractGMM(ABC):
    """
    Abstract class representing a Gaussian Mixture Model.
    """

    @abstractmethod
    def forward(self, x: ch.Tensor) -> tuple:
        """
        Forward pass to compute parameters of the Gaussian components based on input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Tuple containing component weights, means, and covariance matrices.
        """
        pass

    def covariance(self, chol: ch.Tensor) -> ch.Tensor:
        """
        Compute covariance matrices from Cholesky decompositions.

        Args:
            chol (torch.Tensor): Cholesky decompositions of covariance matrices.

        Returns:
            torch.Tensor: Covariance matrices.
        """
        cov_matrix = chol @ chol.transpose(-1, -2)
        return cov_matrix

    def get_rsamples(self, mean: ch.Tensor, chol: ch.Tensor, n_samples: int) -> ch.Tensor:
        """
        Generate reparameterized samples from the Gaussian distributions.

        Args:
            mean (torch.Tensor): Means of the Gaussian components.
            chol (torch.Tensor): Cholesky decompositions of the covariance matrices.
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Reparameterized samples.
        """
        rsamples = MultivariateNormal(loc=mean, scale_tril=chol).rsample(ch.Size([n_samples]))
        return rsamples.transpose(0, 1)

    def get_samples_gmm(self, log_gates: ch.Tensor, means: ch.Tensor, chols: ch.Tensor, num_samples: int) -> ch.Tensor:
        """
        Generate samples from the Gaussian Mixture Model.

        Args:
            log_gates (torch.Tensor): Logarithm of the mixing weights for the components.
            means (torch.Tensor): Means of the Gaussian components.
            chols (torch.Tensor): Cholesky decompositions of the covariance matrices of the components.
            num_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Samples generated from the GMM.
        """
        if log_gates.shape[1] == 1:
            # print("target has only one component")
            samples = MultivariateNormal(means.squeeze(1), scale_tril=chols.squeeze(1)).sample((num_samples,))
            return samples.transpose(0, 1)
        else:
            samples = []
            for i in range(log_gates.shape[0]):
                cat = Categorical(ch.exp(log_gates[i]))
                indices = cat.sample((num_samples,))
                chosen_means = means[i, indices]
                chosen_chols = chols[i, indices]
                normal = MultivariateNormal(chosen_means, scale_tril=chosen_chols)
                samples.append(normal.sample())
            return ch.stack(samples)  # [n_contexts, n_samples, n_features]


    def log_prob(self, mean: ch.Tensor, chol: ch.Tensor, samples: ch.Tensor) -> ch.Tensor:
        """
        Calculate the log probabilities of given samples under the Gaussian distributions.

        Args:
            mean (torch.Tensor): Means of the Gaussian components.
            chol (torch.Tensor): Cholesky decompositions of the covariance matrices.
            samples (torch.Tensor): Samples for which to compute the log probabilities.

        Returns:
            torch.Tensor: Log probabilities of the samples.
        """
        if samples.dim() == 4:
            _, _, n_samples, _ = samples.shape
            mean_expanded = mean.unsqueeze(2).expand(-1, -1, n_samples, -1)  # [b, o, s,l]
            chol_expanded = chol.unsqueeze(2).expand(-1, -1, n_samples, -1, -1)

            mvn = MultivariateNormal(loc=mean_expanded, scale_tril=chol_expanded)
            log_probs = mvn.log_prob(samples)  # [b, o, s]
            return log_probs
        elif samples.dim() == 3:
            batch_size, n_samples, _ = samples.shape
            mean_expanded = mean.unsqueeze(1).expand(-1, n_samples, -1)  # [batch_size, n_samples, n_features]
            chol_expanded = chol.unsqueeze(1).expand(-1, n_samples, -1, -1)

            mvn = MultivariateNormal(loc=mean_expanded, scale_tril=chol_expanded)
            log_probs = mvn.log_prob(samples)  # [batch_size, n_samples]
            # return log_probs.mean(dim=1)  # [batch_size]
            return log_probs
        else:
            raise ValueError("Shape of samples should be [batch_size, n_samples, n_features]")

    def log_prob_gmm(self, means: ch.Tensor, chols: ch.Tensor, log_gates: ch.Tensor, samples: ch.Tensor) -> ch.Tensor:
        """
        Calculate the log probabilities of given samples under the Gaussian Mixture Model.

        Args:
            means (torch.Tensor): Means of the Gaussian components.
            chols (torch.Tensor): Cholesky decompositions of the covariance matrices of the components.
            log_gates (torch.Tensor): Logarithm of the mixing weights.
            samples (torch.Tensor): Samples for which to compute the log probabilities.

        Returns:
            torch.Tensor: Log probabilities of the samples under the GMM.
        """
        n_samples = samples.shape[1]
        n_contexts, n_components, _ = means.shape

        means_expanded = means.unsqueeze(1).expand(-1, n_samples, -1, -1)
        chols_expanded = chols.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
        samples_expanded = samples.unsqueeze(2).expand(-1, -1, n_components, -1)

        # since I only plot 2D, I need to modify here into right shape. This if only happens in plotting
        if means_expanded.shape[-1] != samples_expanded.shape[-1]:
            mvn = MultivariateNormal(means_expanded[..., :2], scale_tril=chols_expanded[..., :2, :2])
            log_probs = mvn.log_prob(samples_expanded)
        else:
            mvn = MultivariateNormal(means_expanded, scale_tril=chols_expanded)
            log_probs = mvn.log_prob(samples_expanded)  # [batch_size, n_samples, n_components]

        gate_expanded = log_gates.unsqueeze(1).expand(-1, n_samples, -1)
        log_probs += gate_expanded

        log_probs = ch.logsumexp(log_probs, dim=2)  # [batch_size, n_samples]
        # return ch.sum(log_probs, dim=1)
        return log_probs


    def auxiliary_reward(self, j: int, gate_old: ch.Tensor, mean_old: ch.Tensor, chol_old: ch.Tensor,
                         samples: ch.Tensor) -> ch.Tensor:
        """
        Compute the auxiliary reward for reinforcement learning applications.

        Args:
            j (int): Index of the component.
            gate_old (torch.Tensor): Previous mixing weights.
            mean_old (torch.Tensor): Previous means of the Gaussian components.
            chol_old (torch.Tensor): Previous Cholesky decompositions of the covariance matrices.
            samples (torch.Tensor): Samples for which to compute the auxiliary reward.

        Returns:
            torch.Tensor: Computed auxiliary rewards.
        """
        gate_old_expanded = gate_old.unsqueeze(1).expand(-1, samples.shape[1], -1)
        numerator = gate_old_expanded[:, :, j] + self.log_prob(mean_old[:, j], chol_old[:, j], samples)
        denominator = self.log_prob_gmm(mean_old, chol_old, gate_old, samples)
        aux_reward = numerator - denominator
        return aux_reward

    def log_responsibility(self,gate_old: ch.Tensor, mean_old: ch.Tensor, chol_old: ch.Tensor,
                         samples: ch.Tensor) -> ch.Tensor:
        """
        Compute responsibility for each component in the mixture model
        :param gate_old: (b,o)
        :param mean_old: (b,o,l)
        :param chol_old: (b,o,l,l)
        :param samples: (b,o,s,l)
        :return:
        """
        log_res = [self.auxiliary_reward(i, gate_old, mean_old, chol_old, samples[:,i]) for i in range(mean_old.shape[1])]
        return ch.stack(log_res, dim=1)


    def get_rsamples_w2(self, mean: ch.Tensor, cov: ch.Tensor, n_samples: int) -> ch.Tensor:
        """
        Generate reparameterized samples from the Gaussian distributions.

        Args:
            mean (torch.Tensor): Means of the Gaussian components.
            cov (torch.Tensor): the covariance matrices.
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Reparameterized samples.
        """
        rsamples = MultivariateNormal(loc=mean, covariance_matrix=cov).rsample(ch.Size([n_samples]))
        return rsamples.transpose(0, 1)


    def get_samples_gmm_w2(self, log_gates: ch.Tensor, means: ch.Tensor, covs: ch.Tensor, num_samples: int) -> ch.Tensor:
        """
        Generate samples from the Gaussian Mixture Model.

        Args:
            log_gates (torch.Tensor): Logarithm of the mixing weights for the components.
            means (torch.Tensor): Means of the Gaussian components.
            covs (torch.Tensor): the covariance matrices of the components.
            num_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Samples generated from the GMM.
        """
        if log_gates.shape[1] == 1:
            # print("target has only one component")
            samples = MultivariateNormal(means.squeeze(1), covs.squeeze(1)).sample((num_samples,))
            return samples.transpose(0, 1)
        else:
            samples = []
            for i in range(log_gates.shape[0]):
                cat = Categorical(ch.exp(log_gates[i]))
                indices = cat.sample((num_samples,))
                chosen_means = means[i, indices]
                chosen_covs = covs[i, indices]
                normal = MultivariateNormal(chosen_means, chosen_covs)
                samples.append(normal.sample())
            return ch.stack(samples)  # [n_contexts, n_samples, n_features]


    def log_prob_w2(self, mean: ch.Tensor, cov: ch.Tensor, samples: ch.Tensor) -> ch.Tensor:
        """
        Calculate the log probabilities of given samples under the Gaussian distributions.

        Args:
            mean (torch.Tensor): Means of the Gaussian components.
            cov (torch.Tensor): the covariance matrices.
            samples (torch.Tensor): Samples for which to compute the log probabilities.

        Returns:
            torch.Tensor: Log probabilities of the samples.
        """
        if samples.dim() == 3:
            batch_size, n_samples, _ = samples.shape
            mean_expanded = mean.unsqueeze(1).expand(-1, n_samples, -1)  # [batch_size, n_samples, n_features]
            cov_expanded = cov.unsqueeze(1).expand(-1, n_samples, -1, -1)

            mvn = MultivariateNormal(loc=mean_expanded, covariance_matrix=cov_expanded)
            log_probs = mvn.log_prob(samples)  # [batch_size, n_samples]
            # return log_probs.mean(dim=1)  # [batch_size]
            return log_probs
        else:
            raise ValueError("Shape of samples should be [batch_size, n_samples, n_features]")


    def log_prob_gmm_w2(self, means: ch.Tensor, covs: ch.Tensor, log_gates: ch.Tensor, samples: ch.Tensor) -> ch.Tensor:
        """
        Calculate the log probabilities of given samples under the Gaussian Mixture Model.

        Args:
            means (torch.Tensor): Means of the Gaussian components.
            covs (torch.Tensor): the covariance matrices of the components.
            log_gates (torch.Tensor): Logarithm of the mixing weights.
            samples (torch.Tensor): Samples for which to compute the log probabilities.

        Returns:
            torch.Tensor: Log probabilities of the samples under the GMM.
        """
        n_samples = samples.shape[1]
        n_contexts, n_components, _ = means.shape

        means_expanded = means.unsqueeze(1).expand(-1, n_samples, -1, -1)
        covs_expanded = covs.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
        samples_expanded = samples.unsqueeze(2).expand(-1, -1, n_components, -1)

        # since I only plot 2D, I need to modify here into right shape. This if only happens in plotting
        if means_expanded.shape[-1] != samples_expanded.shape[-1]:
            mvn = MultivariateNormal(means_expanded[..., :2], covs_expanded[..., :2, :2])
            log_probs = mvn.log_prob(samples_expanded)
        else:
            mvn = MultivariateNormal(means_expanded, covs_expanded)
            log_probs = mvn.log_prob(samples_expanded)  # [batch_size, n_samples, n_components]

        gate_expanded = log_gates.unsqueeze(1).expand(-1, n_samples, -1)
        log_probs += gate_expanded

        log_probs = ch.logsumexp(log_probs, dim=2)  # [batch_size, n_samples]
        # return ch.sum(log_probs, dim=1)
        return log_probs


    def auxiliary_reward_w2(self, j: int, gate_old: ch.Tensor, mean_old: ch.Tensor, cov_old: ch.Tensor,
                         samples: ch.Tensor) -> ch.Tensor:
        """
        Compute the auxiliary reward for reinforcement learning applications.

        Args:
            j (int): Index of the component.
            gate_old (torch.Tensor): Previous mixing weights.
            mean_old (torch.Tensor): Previous means of the Gaussian components.
            cov_old (torch.Tensor): Previous covariance matrices.
            samples (torch.Tensor): Samples for which to compute the auxiliary reward.

        Returns:
            torch.Tensor: Computed auxiliary rewards.
        """
        gate_old_expanded = gate_old.unsqueeze(1).expand(-1, samples.shape[1], -1)
        numerator = gate_old_expanded[:, :, j] + self.log_prob_w2(mean_old[:, j], cov_old[:, j], samples)
        denominator = self.log_prob_gmm_w2(mean_old, cov_old, gate_old, samples)
        aux_reward = numerator - denominator
        return aux_reward
