import copy
import math
import torch as ch
from typing import Tuple, Union

from trust_region_projections.model.conditionalgaussian_model import ConditionalGaussianModel
from trust_region_projections.utils.network_utils import get_optimizer
from trust_region_projections.utils.projection_utils import gaussian_kl
from trust_region_projections.utils.torch_utils import generate_minibatches, select_batch, tensorize

def mean_projection(mean: ch.Tensor, old_mean: ch.Tensor, maha: ch.Tensor, eps: ch.Tensor):
    """
    Projects the mean based on the Mahalanobis objective and trust region.
    Args:
        mean: current mean vectors
        old_mean: old mean vectors
        maha: Mahalanobis distance between the two mean vectors
        eps: trust region bound

    Returns:
        projected mean that satisfies the trust region
    """
    batch_shape = mean.shape[:-1]
    mask = maha > eps

    # if nothing has to be projected skip computation
    if mask.any():
        omega = ch.ones(batch_shape, dtype=mean.dtype, device=mean.device)
        omega[mask] = ch.sqrt(maha[mask] / eps) - 1.
        omega = ch.max(-omega, omega)[..., None]

        m = (mean + omega * old_mean) / (1 + omega + 1e-16)
        proj_mean = ch.where(mask[..., None], m, mean)
    else:
        proj_mean = mean

    return proj_mean

class BaseProjectionLayer(object):

    def __init__(self,
                 proj_type: str = "",
                 mean_bound: float = 0.03,
                 cov_bound: float = 1e-3,
                 trust_region_coeff: float = 0.0,
                 scale_prec: bool = True,

                 action_dim: Union[None, int] = None,
                 total_train_steps: Union[None, int] = None,

                 do_regression: bool = False,
                 regression_iters: int = 1000,
                 regression_lr: int = 3e-4,
                 optimizer_type_reg: str = "adam",

                 cpu: bool = True,
                 dtype: ch.dtype = ch.float32,
                 ):

        """
        Base projection layer, which can be used to compute metrics for non-projection approaches.
        Args:
           proj_type: Which type of projection to use. None specifies no projection and uses the TRPO objective.
           mean_bound: projection bound for the step size w.r.t. mean
           cov_bound: projection bound for the step size w.r.t. covariance matrix
           trust_region_coeff: Coefficient for projection regularization loss term.
           scale_prec: If true used mahalanobis distance for projections instead of euclidean with Sigma_old^-1.
           action_dim: number of action dimensions to scale exp decay correctly.
           total_train_steps: total number of training steps to compute appropriate decay over time.
           do_regression: Conduct additional regression steps after the the policy steps to match projection and policy.
           regression_iters: Number of regression steps.
           regression_lr: Regression learning rate.
           optimizer_type_reg: Optimizer for regression.
           cpu: Compute on CPU only.
           dtype: Data type to use, either of float32 or float64. The later might be necessary for higher
                   dimensions in order to learn the full covariance.
        """

        # projection and bounds
        self.proj_type = proj_type
        self.mean_bound = tensorize(mean_bound, cpu=cpu, dtype=dtype)
        self.cov_bound = tensorize(cov_bound, cpu=cpu, dtype=dtype)
        self.trust_region_coeff = trust_region_coeff
        self.scale_prec = scale_prec

        # regression
        self.do_regression = do_regression
        self.regression_iters = regression_iters
        self.lr_reg = regression_lr
        self.optimizer_type_reg = optimizer_type_reg

    def __call__(self, policy, p: Tuple[ch.Tensor, ch.Tensor], q, step, *args, **kwargs):
        """
        handle method for projection (and other functionality later)
        """
        return self._trust_region__projection(policy, p, q, self.mean_bound, self.cov_bound, **kwargs)

    def _trust_region_projection(self, policy: ConditionalGaussianModel, p: Tuple[ch.Tensor, ch.Tensor],
                                 q: Tuple[ch.Tensor, ch.Tensor], eps: ch.Tensor, eps_cov: ch.Tensor, **kwargs):
        """
        Hook for implementing the specific trust region projection
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: mean trust region bound
            eps_cov: covariance trust region bound
            **kwargs:

        Returns:
            projected
        """
        return p

    def trust_region_value(self, policy, p, q):
        """
        Computes the KL divergence between two Gaussian distributions p and q.
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
        Returns:
            Mean and covariance part of the trust region metric.
        """
        return gaussian_kl(policy, p, q)

    def get_trust_region_loss(self, policy: ConditionalGaussianModel, p: Tuple[ch.Tensor, ch.Tensor],
                              proj_p: Tuple[ch.Tensor, ch.Tensor]):
        """
        Compute the trust region loss to ensure policy output and projection stay close.
        Args:
            policy: policy instance
            proj_p: projected distribution
            p: predicted distribution from network output

        Returns:
            trust region loss
        """
        p_target = (proj_p[0].detach(), proj_p[1].detach())
        mean_diff, cov_diff = self.trust_region_value(policy, p, p_target)

        delta_loss = (mean_diff + cov_diff).mean()

        return delta_loss * self.trust_region_coeff

    def compute_metrics(self, policy, p, q) -> dict:
        """
        Returns dict with constraint metrics.
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution

        Returns:
            dict with constraint metrics
        """
        with ch.no_grad():
            mean_kl, cov_kl = gaussian_kl(policy, p, q)
            kl = mean_kl + cov_kl

            mean_diff, cov_diff = self.trust_region_value(policy, p, q)

            combined_constraint = mean_diff + cov_diff

        return {'kl': kl.detach().mean(),
                'constraint': combined_constraint.mean(),
                'mean_constraint': mean_diff.mean(),
                'cov_constraint': cov_diff.mean(),
                'kl_max': kl.max(),
                'constraint_max': combined_constraint.max(),
                'mean_constraint_max': mean_diff.max(),
                'cov_constraint_max': cov_diff.max(),
                }

    def trust_region_regression(self, policy: ConditionalGaussianModel, obs: ch.Tensor, q: Tuple[ch.Tensor, ch.Tensor],
                                n_minibatches: int, global_steps: int):
        """
        Take additional regression steps to match projection output and policy output.
        The policy parameters are updated in-place.
        Args:
            policy: policy instance
            obs: collected observations from trajectories
            q: old distributions
            n_minibatches: split the rollouts into n_minibatches.
            global_steps: current number of steps, required for projection
        Returns:
            dict with mean of regession loss
        """

        if not self.do_regression:
            return {}

        policy_unprojected = copy.deepcopy(policy)
        optim_reg = get_optimizer(self.optimizer_type_reg, policy_unprojected.parameters(), learning_rate=self.lr_reg)
        optim_reg.reset()

        reg_losses = obs.new_tensor(0.)

        # get current projected values --> targets for regression
        p_flat = policy(obs)
        p_target = self(policy, p_flat, q, global_steps)

        for _ in range(self.regression_iters):
            batch_indices = generate_minibatches(obs.shape[0], n_minibatches)

            # Minibatches SGD
            for indices in batch_indices:
                batch = select_batch(indices, obs, p_target[0], p_target[1])
                b_obs, b_target_mean, b_target_std = batch
                proj_p = (b_target_mean.detach(), b_target_std.detach())

                p = policy_unprojected(b_obs)

                # invert scaling with coeff here as we do not have to balance with other losses
                loss = self.get_trust_region_loss(policy, p, proj_p) / self.trust_region_coeff

                optim_reg.zero_grad()
                loss.backward()
                optim_reg.step()
                reg_losses += loss.detach()

        policy.load_state_dict(policy_unprojected.state_dict())

        steps = self.regression_iters * (math.ceil(obs.shape[0] / n_minibatches))
        return {"regression_loss": (reg_losses / steps).detach()}