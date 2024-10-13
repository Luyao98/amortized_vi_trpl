from toy_task.GMM.targets.banana_mixture_target import BananaMixtureTarget, get_curvature_fn
from toy_task.GMM.targets.gaussian_mixture_target import get_gmm_target
from toy_task.GMM.targets.gmm_star_target import get_star_target
from toy_task.GMM.targets.funnel_target import FunnelTarget, get_sig_fn


def get_target(target_name, target_components, context_dim=None):
    if target_name == "bmm":
        return BananaMixtureTarget(get_curvature_fn, target_components)
    elif target_name == "gmm":
        return get_gmm_target(target_components, context_dim)
    elif target_name == "funnel":
        return FunnelTarget(get_sig_fn, context_dim)
    elif target_name == "star":
        return get_star_target(target_components, context_dim)
    else:
        raise ValueError(f"Unknown target name: {target_name}")
