from daft.src.multi_daft_vi.multi_daft import MultiDaft
from daft.src.multi_daft_vi.util_multi_daft import create_initial_gmm_parameters

from daft.src.multi_daft_vi.lnpdf import make_contextual_star_target, make_contextual_gmm_target, LNPDF
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot(algorithm):
    from daft.src.multi_daft_vi.recording.util import plot2d_matplotlib

    fig, axes = plt.subplots(
        4, 3, squeeze=False, figsize=(4 * 3, 10)
    )

    plot2d_matplotlib(
        target_dist=algorithm.target_dist,
        model=algorithm.model,
        fig=fig,
        axes=axes,
        min_x=-15,
        max_x=15,
        min_y=-15,
        max_y=15,
    )
    # fig.savefig("../reports/plots/star_target.png")
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # task_config = {"name": "star_target", "n_tasks": 50, "star_target": {"n_components": 7}}
    # target_dist: LNPDF = make_contextual_star_target(
    #     task_config.get("n_tasks"), task_config.get("star_target").get("n_components")
    # )
    task_config = {"name": "gmm_target", "n_tasks": 3, "gmm_target": {"n_components": 10}}
    target_dist: LNPDF = make_contextual_gmm_target(
        task_config.get("n_tasks"), task_config.get("gmm_target").get("n_components")
    )

    log_weights, means, precs = create_initial_gmm_parameters(
        d_z=2,
        n_tasks=task_config.get("n_tasks"),
        n_components=10,
        prior_scale=50,
        initial_var=1.0,
        target_dist=target_dist
    )

    algorithm_config = {
        "model": {
            "prior_scale": 400,
            "initial_var": 1.0,
            "n_components": 10,
            "n_dimensions": 2,
        },
        "n_samples_per_comp": 5,
        "mini_batch_size_for_target_density": 1000,
        "prec_regularization": 1e-6,
        "more": {
            "component_kl_bound": 0.0001,
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

    for i in range(500):
        if i == 0 or i == 499:
            plot(algorithm)
        elbo = algorithm.step()

