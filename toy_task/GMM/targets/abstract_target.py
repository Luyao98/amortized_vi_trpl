from typing import Callable
from abc import ABC, abstractmethod
import torch as ch


from toy_task.GMM.utils.network_utils import eval_fn_grad


class AbstractTarget(ABC):
    """
    Abstract class representing a target distribution for various probabilistic targets.
    """

    @abstractmethod
    def get_contexts(self,
                     n_context: int
                     ) -> ch.Tensor:
        """
        Generates a set of contexts used to parameterize the target distribution.

        Parameters:
        - n_context (int): The number of context vectors to generate.

        Returns:
        - ch.Tensor: A tensor containing the generated context vectors with shape (n_contexts, context_dim).
        """
        pass

    @abstractmethod
    def sample(self,
               contexts: ch.Tensor,
               n_samples: int
               ) -> ch.Tensor:
        """
        Samples from the target distribution given the specified contexts.

        Parameters:
        - contexts (ch.Tensor): The context vectors parameterizing the target distribution.
        - n_samples (int): The number of samples to draw from the target distribution.

        Returns:
        - ch.Tensor: drawn samples from the target distribution with shape (n_samples, n_contexts, d_z).
        """
        pass


    @abstractmethod
    def log_prob_tgt(self,
                     contexts: ch.Tensor,
                     samples: ch.Tensor
                     ) -> ch.Tensor:
        """
        Calculates the logarithm of the probability density function of the target distribution
        for the given samples under the specified contexts.

        Parameters:
        - contexts (ch.Tensor): The context vectors parameterizing the target distribution.
        - samples (ch.Tensor): The samples for which to evaluate the log probability.

        Returns:
        - ch.Tensor: log densities of the target with shape (n_samples, n_contexts)
        """
        pass

    @abstractmethod
    def visualize(self,
                  contexts: ch.Tensor,
                  n_samples: int = None
                  ):
        """
        Visualizes the target distribution given the contexts, optionally with a specified number of samples.

        Parameters:
        - contexts (ch.Tensor): The context vectors parameterizing the target distribution.
        - n_samples (int, optional): Number of samples to draw from the target distribution for visualization.

        This method does not return a value; it should create a visualization such as a plot.
        """
        pass


    def update_samples(self,
                       samples: ch.Tensor,
                       fn: Callable,
                       lr: float,
                       n: int
                       ) -> ch.Tensor:
        """
        Updates the samples towards promising area by gradient ascend given the specified contexts.

        Parameters:
        - samples (ch.Tensor): The samples to update.
        - fn (Callable): The target function (e.g. GMM's log_prob).
        - lr (float): The learning rate.
        - n (int): The number of gradient updates.

        Returns:
        - ch.Tensor: The updated samples.

        """
        updated_samples = samples.clone().detach()
        updated_samples.requires_grad = True

        for i in range(n):
            _, grad = eval_fn_grad(fn, updated_samples, compute_grad=True)

            with ch.no_grad():
                updated_samples += lr * grad

        return updated_samples.detach()

