from toy_task.GMM.targets.banana_mixture_target import BananaMixtureTarget, get_curvature_fn
from toy_task.GMM.targets.gaussian_mixture_target import get_gmm_target
from toy_task.GMM.targets.funnel_target import FunnelTarget, get_sig_fn


def get_target(target_name, n_components):
    if target_name == "bmm":
        return BananaMixtureTarget(get_curvature_fn)
    elif target_name == "gmm":
        return get_gmm_target(n_components)
    elif target_name == "funnel":
        return FunnelTarget(get_sig_fn)
    else:
        raise ValueError(f"Unknown target name: {target_name}")