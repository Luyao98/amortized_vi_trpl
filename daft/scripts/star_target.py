from daft.src.multi_daft_vi.multi_daft import MultiDaft
from daft.src.multi_daft_vi.util_multi_daft import create_initial_gmm_parameters

from daft.src.multi_daft_vi.lnpdf import make_star_target, LNPDF
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot(algorithm):
    from daft.src.multi_daft_vi.recording.util import plot2d_matplotlib

    fig, axes = plt.subplots(
        3, task_config.get("n_tasks") + 1, squeeze=False, figsize=(5 * task_config.get("n_tasks"), 10)
    )
    plt.ion()
    plot2d_matplotlib(
        target_dist=algorithm.target_dist,
        model=algorithm.model,
        fig=fig,
        axes=axes,
        min_x=-5,
        max_x=5,
        min_y=-5,
        max_y=5,
    )
    # fig.savefig("../reports/plots/star_target.png")
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    task_config = {"name": "star_target", "n_tasks": 3, "star_target": {"n_components": 7}}
    target_dist: LNPDF = make_star_target(
        task_config.get("n_tasks"), task_config.get("star_target").get("n_components")
    )

    log_weights, means, precs = create_initial_gmm_parameters(
        d_z=2,
        n_tasks=task_config.get("n_tasks"),
        n_components=10,
        prior_scale=1.0,
        initial_var=1.0,
    )

    algorithm_config = {
        "model": {
            "prior_scale": 1.0,
            "initial_var": 1.0,
            "n_components": 10,
            "n_dimensions": 2,
        },
        "n_samples_per_comp": 100,
        "mini_batch_size_for_target_density": 1000,
        "prec_regularization": 1e-6,
        "more": {
            "component_kl_bound": 0.01,
            "global_upper_bound": 1000,
            "global_lower_bound": 0.0,
            "dual_conv_tol": 0.1,
            "use_warm_starts": False,
            "warm_start_interval_size": 100,
            "max_prec_element_value": 1.0e+8,
            "max_dual_steps": 50,
        },
    }

    algorithm = MultiDaft(
        algorithm_config=algorithm_config,
        target_dist=target_dist,
        log_w_init=log_weights,
        mean_init=means,
        prec_init=precs,
    )

    for _ in range(1000):
        algorithm.step()
plot(algorithm)
