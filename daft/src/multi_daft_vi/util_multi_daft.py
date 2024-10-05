import einops
import torch
import numpy
import random

from daft.src.gmm_util.gmm import GMM
from daft.src.multi_daft_vi.lnpdf import LNPDF


# @tf.function(
#     input_signature=[
#         tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
#         tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
#         tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
#         tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
#     ]
# )
def model_fitting(
    mean: torch.Tensor, prec: torch.Tensor, samples: torch.Tensor, rewards_grad: torch.Tensor
):
    # model is 0.5 x^T R x + x^T r + r_0
    # It uses Stein for the quadratic matrix R
    diff = samples - mean
    prec_times_diff = einops.einsum(prec, diff, "t c n m, s t c m -> s t c n")
    exp_hessian_per_sample = einops.einsum(
        prec_times_diff,
        rewards_grad,
        "s t c n, s t c m -> s t c n m",
    )
    exp_hessian = einops.reduce(exp_hessian_per_sample, "s t c n m -> t c n m", "mean")
    exp_hessian = 0.5 * (exp_hessian + einops.rearrange(exp_hessian, "t c n m -> t c m n"))
    exp_gradient = einops.reduce(rewards_grad, "s t c m -> t c m", "mean")
    quad_term = exp_hessian
    lin_term = exp_gradient - einops.einsum(quad_term, mean, "t c n m, t c m -> t c n")

    return quad_term, lin_term


# @tf.function(
#     input_signature=[
#         tf.TensorSpec(shape=[None, None], dtype=tf.float32),
#         tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
#     ]
# )
def weight_update(model_log_w, rewards):
    # TODO one component: no update
    avg_rewards = einops.reduce(rewards, "s t c -> t c", "mean")
    unnormalized_log_weights = model_log_w + avg_rewards
    new_log_weights = unnormalized_log_weights - torch.logsumexp(
        unnormalized_log_weights, dim=-1, keepdim=True
    )
    return new_log_weights


def create_initial_gmm_parameters(
    d_z: int,
    n_tasks: int,
    n_components: int,
    prior_scale: float,
    initial_var: float,
):
    # TODO: check more sophisticated initializations
    prior = torch.distributions.Normal(loc=torch.zeros(d_z), scale=prior_scale * torch.ones(d_z))
    initial_cov = initial_var * torch.eye(d_z)  # same as prior covariance
    initial_prec = torch.linalg.inv(initial_cov)

    weights = torch.ones((n_tasks, n_components)) / n_components
    log_weights = torch.log(weights)
    means = prior.sample((n_tasks, n_components))
    precs = torch.stack([initial_prec] * n_components, dim=0)
    precs = torch.stack([precs] * n_tasks, dim=0)

    # check output
    assert log_weights.shape == (n_tasks, n_components)
    assert means.shape == (n_tasks, n_components, d_z)
    assert precs.shape == (n_tasks, n_components, d_z, d_z)
    return log_weights, means, precs


def compute_elbo(model: GMM, target_dist: LNPDF, num_samples_per_component, mini_batch_size=None):
    """
    Formula for elbo:
    ELBO = 1/N \sum_n \sum_o q_\theta(o) [\log p(z_{n,o}) - \log q_\theta(z_{n,o})],  z_{n,o} ~ q_\theta(z_{n,o} | o)
    N = num_samples_per_component
    return shape: [num_tasks]
    """
    samples_per_comp = model.sample_all_components(num_samples_per_component)
    num_samples, num_tasks, num_components, dim_z = tuple(samples_per_comp.shape)
    samples_flattened = einops.rearrange(samples_per_comp, "s t c d -> (s c) t d")
    # should now have shape [num_samples_per_comp * num_comp, task, dz]
    # get the tgt and model log densities
    if mini_batch_size is None:
        target_densities, _ = target_dist.log_density(samples_flattened, compute_grad=False)
    else:
        target_densities, _ = target_dist.mini_batch_log_density(
            samples_flattened,
            mini_batch_size=mini_batch_size,
            compute_grad=False,
        )
    model_densities, _ = model.log_density(samples_flattened, compute_grad=False)
    # unflatten results
    # unflatten
    target_densities = einops.rearrange(
        target_densities,
        "(s c) t -> s t c",
        s=num_samples,
        c=num_components,
    )
    model_densities = einops.rearrange(
        model_densities,
        "(s c) t -> s t c",
        s=num_samples,
        c=num_components,
    )
    # compute the elbo
    densities_diff = target_densities - model_densities
    # take the mean over the samples
    densities_mean = einops.reduce(densities_diff, "s t c -> t c", reduction="mean")
    # take the weighted sum over the components (weighted by the model weights exp(model.log_w))
    elbo = einops.einsum(
        densities_mean,
        torch.exp(model.log_w),
        "t c, t c -> t",
    )
    return elbo


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


