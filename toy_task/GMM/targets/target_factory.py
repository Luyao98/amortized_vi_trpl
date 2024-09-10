from toy_task.GMM.targets.banana_mixture_target import BananaMixtureTarget, get_curvature_fn
from toy_task.GMM.targets.gaussian_mixture_target import get_gmm_target
from toy_task.GMM.targets.funnel_target import FunnelTarget, get_sig_fn


def get_target(target_name, target_components, context_dim=1):
    if target_name == "bmm":
        return BananaMixtureTarget(get_curvature_fn, target_components)
    elif target_name == "gmm":
        return get_gmm_target(target_components, context_dim)
    elif target_name == "funnel":
        # funnel function is a bit different, target_components is in fact the dimension of the target
        return FunnelTarget(get_sig_fn, target_components)
    else:
        raise ValueError(f"Unknown target name: {target_name}")
