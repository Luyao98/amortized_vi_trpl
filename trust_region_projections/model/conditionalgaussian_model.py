from typing import Sequence, Tuple

import numpy as np
import torch as ch
import torch.nn as nn

from trust_region_projections.utils.network_utils import get_activation, get_mlp, initialize_weights
from trust_region_projections.utils.torch_utils import inverse_softplus, diag_bijector, fill_triangular, fill_triangular_inverse


class ConditionalGaussianModel(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, init: str = "orthogonal", hidden_sizes: Sequence[int] = (64, 64),
                 activation: str = "tanh", init_std: float = 1., minimal_std: float = 1e-5):
        """
        Conditional Gaussian Model P(x|c) with a fully connected neural network.
        The parameterizing tensor is a mean vector and a cholesky matrix.
        Args:
            obs_dim: dimensionality of c aka input dimensionality
            action_dim: dimensionality of x aka output dimensionality
            init: Initialization type for the layers
            hidden_sizes: Sequence of hidden layer sizes for each hidden layer in the neural network.
            activation: Type of ctivation for hidden layers
            init_std: initial value of the standard deviation matrix
            minimal_std: minimal standard deviation

        Returns:

        """
        super().__init__()

        self.activation = get_activation(activation)
        self.action_dim = action_dim
        self.minimal_std = minimal_std
        self.init_std = ch.tensor(init_std)

        self._affine_layers = get_mlp(obs_dim, hidden_sizes, init, True)

        prev_size = hidden_sizes[-1]

        self.diag_activation = nn.Softplus()
        self.diag_activation_inv = inverse_softplus

        # This shift is applied to the Parameter/cov NN output before applying the transformation
        # and gives hence the wanted initial cov
        self._pre_activation_shift = self._get_preactivation_shift(self.init_std, minimal_std)
        self._mean = self._get_mean(action_dim, prev_size, init)
        self._pre_std =self._get_std_layer(prev_size, action_dim, init)

    def forward(self, x: ch.Tensor, train: bool = True):
        self.train(train)

        for affine in self._affine_layers:
            x = self.activation(affine(x))

        flat_chol = self._pre_std(x)
        chol = fill_triangular(flat_chol).expand(x.shape[0], -1, -1)
        chol = diag_bijector(lambda z: self.diag_activation(z + self._pre_activation_shift) + self.minimal_std, chol)

        return self._mean(x), chol


    def _get_mean(self, action_dim, prev_size=None, init=None, scale=0.01):
        """
        Constructor method for mean prediction.
        Args:
            action_dim: action dimension for output shape
            prev_size: previous layer's output size
            init: initialization type of layer.
            scale

        Returns:
            Mean parametrization.
        """
        mean = nn.Linear(prev_size, action_dim)
        initialize_weights(mean, init, scale=scale)
        return mean

    def _get_std_layer(self, prev_size: int, action_dim: int, init: str):
        """
        Constructor method for std prediction.
        Args:
            prev_size: previous layer's output size
            action_dim: action dimension for output shape
            init: initialization type of layer.

        Returns:
            Torch layer for covariance prediction.
        """
        chol_shape = action_dim * (action_dim + 1) // 2
        flat_chol = nn.Linear(prev_size, chol_shape)
        initialize_weights(flat_chol, init, scale=0.01)
        return flat_chol

    def _get_preactivation_shift(self, init_std, minimal_std):
        """
        Compute the prediction shift to enforce an initial covariance value.
        Args:
            init_std: value to initialize the covariance output with.
            minimal_std: lower bound on the covariance.

        Returns:
            Preactivation shift to enforce minimal and initial covariance.
        """
        return self.diag_activation_inv(init_std - minimal_std)


    def sample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        """
        Given prob dist p=(mean, var), generate samples WITHOUT reparametrization trick
         Args:
            p: Tuple (means, var). means (batch_size, action_space), var (action_space,).
                p are batched probability distributions you're sampling from
            n: Number of samples

        Returns:
            Actions sampled from p_i (batch_size, action_dim)
        """
        return self.rsample(p, n).detach()

    def rsample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        """
        Given prob dist p=(mean, var), generate samples WITH reparametrization trick.
        This version applies the reparametrization trick to allow for backpropagate through it.
         Args:
            p: Tuple (means, var). means (batch_size, action_space), var (action_space,).
                p are batched probability distributions you're sampling from
            n: Number of samples
        Returns:
            Actions sampled from p_i (batch_size, action_dim)
        """
        means, chol = p
        eps = ch.randn((n,) + means.shape).to(dtype=chol.dtype, device=chol.device)[..., None]
        samples = (chol @ eps).squeeze(-1) + means
        return samples.squeeze(0)

    def log_probability(self, p: Tuple[ch.Tensor, ch.Tensor], x: ch.Tensor, **kwargs) -> ch.Tensor:
        """
        Computes the log probability of x given a batched distributions p (mean, std)
        Args:
            p: Tuple (means, var). means (batch_size, action_space), var (action_space,).
            x: Values to compute logpacs for
            **kwargs:

        Returns:
            Log probabilities of x.
        """
        mean, std = p
        k = mean.shape[-1]

        logdet = self.log_determinant(std)
        mean_diff = self.maha(x, mean, std)
        nll = 0.5 * (k * np.log(2 * np.pi) + logdet + mean_diff)
        return -nll

    def log_determinant(self, std: ch.Tensor) -> ch.Tensor:
        """
        Returns the log determinant of the cholesky matrix
        Args:
            std: cholesky matrix
        Returns:
            The log determinant of std, aka log sum the diagonal
        """
        return 2 * std.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    def maha(self, mean: ch.Tensor, mean_other: ch.Tensor, std: ch.Tensor):
        """
        Compute the mahalanbis distance between two means. std is the scaling matrix.
        Args:
            mean: left mean
            mean_other: right mean
            std: scaling matrix

        Returns:
            Mahalanobis distance between mean and mean_other
        """
        diff = (mean - mean_other)[..., None]
        return ch.triangular_solve(diff, std, upper=False)[0].pow(2).sum([-2, -1])

    def precision(self, std: ch.Tensor) -> ch.Tensor:
        """
        Compute precision matrix given the std.
        Args:
            std: std matrix

        Returns:
          inverse of std matrix. aka precision matrix
        """
        return ch.cholesky_solve(ch.eye(std.shape[-1], dtype=std.dtype, device=std.device), std, upper=False)

    def covariance(self, std) -> ch.Tensor:
        """
        Compute the full covariance matrix given the std.
        Args:
            std: std matrix

        Returns:

        """
        std = std.view((-1,) + std.shape[-2:])
        return (std @ std.permute(0, 2, 1)).squeeze(0)