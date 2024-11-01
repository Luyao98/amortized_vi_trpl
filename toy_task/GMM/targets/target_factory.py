from toy_task.GMM.targets.banana_mixture_target import BananaMixtureTarget, get_curvature_fn
from toy_task.GMM.targets.gaussian_mixture_target import get_gmm_target
from toy_task.GMM.targets.gmm_star_target import get_star_target
from toy_task.GMM.targets.funnel_target import FunnelTarget, get_sig_fn


def get_target(target_name, target_components, context_dim=None):
    """
    Returns an instance of a target distribution class based on the provided target name.

    Parameters:
    - target_name (str): The name of the target distribution. Options include "bmm", "gmm", "funnel", "star".
    - target_components (int): The number of components used in the target distribution.
    - context_dim (int, optional): The dimensionality of the context vectors for the target distribution.

    Returns:
    - An instance of a target distribution class corresponding to the specified target name.

    Raises:
    - ValueError: If an unknown target name is provided.
    """

    # Return the Banana Mixture Model target distribution with the specified curvature function and components.
    if target_name == "bmm":
        return BananaMixtureTarget(get_curvature_fn, target_components)

    # Return a Gaussian Mixture Model (GMM) target distribution with the given components and context dimension.
    elif target_name == "gmm":
        return get_gmm_target(target_components, context_dim)

    # Return a Funnel distribution target with the sigmoid function and context dimension.
    elif target_name == "funnel":
        return FunnelTarget(get_sig_fn, context_dim)

    # Return a star-shaped Gaussian Mixture Model target with the given components and context dimension.
    elif target_name == "star":
        return get_star_target(target_components, context_dim)

    else:
        raise ValueError(f"Unknown target name: {target_name}")
