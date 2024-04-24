from abc import ABC, abstractmethod


class AbstractGMM(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def covariance(self, chol):
        pass

    @abstractmethod
    def get_rsamples(self, mean, chol, n_samples):
        pass

    @abstractmethod
    def log_prob(self, mean, chol, samples):
        pass

    @abstractmethod
    def log_prob_gmm(self, means, chols, log_gates, samples):
        pass

    @abstractmethod
    def auxiliary_reward(self, j, gate_old, mean_old, chol_old, samples):
        pass

