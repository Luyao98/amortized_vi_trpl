from typing import Callable, Tuple
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
                       input_samples: Tuple[ch.Tensor, ch.Tensor],
                       fn: Callable,
                       lr: float,
                       n: int
                       ) -> ch.Tensor:
        """
        Updates the samples towards promising area by gradient ascend given the specified contexts.

        Parameters:
        - input_samples (Tuple[ch.Tensor, ch.Tensor]): contexts and samples.
        - fn (Callable): The target function (e.g. GMM's log_prob).
        - lr (float): The learning rate.
        - n (int): The number of gradient updates.

        Returns:
        - ch.Tensor: The updated samples.

        """

        contexts, samples = input_samples
        updated_samples = samples.clone().detach()
        updated_samples.requires_grad = True
        if samples.shape[-1] == 2:
            for i in range(n):
                _, grad = eval_fn_grad(fn, contexts, updated_samples, compute_grad=True)

                with ch.no_grad():
                    updated_samples = updated_samples + lr * grad

        else:
            # need a better way for high dim samples
            optimizer = ch.optim.Adam([updated_samples], lr=lr)
            for i in range(n):
                optimizer.zero_grad()
                # _, grad = eval_fn_grad(fn, contexts, updated_samples, compute_grad=True)
                f_z = fn(contexts, updated_samples)
                if not ch.isfinite(f_z).all():
                    import wandb
                    wandb.finish()
                    raise ValueError("The target function returned inf or NaN")

                f_z.sum().backward()
                if updated_samples.grad is not None:
                    if not ch.isfinite(updated_samples.grad).all():
                        import wandb
                        wandb.finish()
                        raise ValueError("Gradient contains NaN or Inf after backward pass")
                ch.nn.utils.clip_grad_norm_([updated_samples], max_norm=0.1)

                optimizer.step()

        return updated_samples.detach()
