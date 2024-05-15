from abc import ABC, abstractmethod
import torch as ch


class AbstractTarget(ABC):
    """
    Abstract class representing a target distribution for various probabilistic models.
    """

    @abstractmethod
    def get_contexts(self, n_context: int) -> ch.Tensor:
        """
        Generates  a set of contexts used to parameterize the target distribution.

        Parameters:
        - n_context (int): The number of context vectors to generate.

        Returns:
        - torch.Tensor: A tensor containing the generated context vectors.
        """
        pass

    @abstractmethod
    def sample(self, contexts: ch.Tensor, n_samples: int) -> ch.Tensor:
        """
        Samples from the target distribution given the specified contexts.

        Parameters:
        - contexts (torch.Tensor): The context vectors parameterizing the target distribution.
        - n_samples (int): The number of samples to draw from the target distribution.

        Returns:
        - torch.Tensor: A tensor containing the samples drawn from the target distribution.
        """
        pass


    @abstractmethod
    def log_prob_tgt(self, contexts: ch.Tensor, samples: ch.Tensor) -> ch.Tensor:
        """
        Calculates the logarithm of the probability density/mass function of the target distribution
        for the given samples under the specified contexts.

        Parameters:
        - contexts (torch.Tensor): The context vectors parameterizing the target distribution.
        - samples (torch.Tensor): The samples for which to evaluate the log probability.

        Returns:
        - torch.Tensor: A tensor containing the log probabilities of the given samples.
        """
        pass

    @abstractmethod
    def visualize(self, contexts: ch.Tensor, n_samples: int = None):
        """
        Visualizes the target distribution given the contexts, optionally with a specified number of samples.

        Parameters:
        - contexts (torch.Tensor): The context vectors parameterizing the target distribution.
        - n_samples (int, optional): Number of samples to draw from the target distribution for visualization.

        This method does not return a value; it should create a visualization such as a plot.
        """
        pass
