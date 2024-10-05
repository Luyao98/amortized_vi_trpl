from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib import pyplot as plt


from daft.src.multi_daft_vi.lnpdf import LNPDF, make_contextual_star_target
from daft.src.multi_daft_vi.multi_daft import MultiDaft
from daft.src.multi_daft_vi.recording.util import plot2d_matplotlib
from daft.src.multi_daft_vi.util_multi_daft import create_initial_gmm_parameters, set_seed


def daft_star(config):
    # task parameters
    task_config = config["task"]
    n_tasks = task_config["n_tasks"]
    task_components = task_config["n_components"]

    # algorithm parameters
    algorithm_config = config["algorithm"]

    # model parameters
    model_config = config["algorithm"]["model"]
    n_components = model_config["n_components"]
    prior_scale = model_config["prior_scale"]
    initial_var = model_config["initial_var"]
    n_dimensions = model_config["n_dimensions"]


    target_dist: LNPDF = make_contextual_star_target(
        n_tasks=n_tasks,
        n_components=task_components,
    )

    log_weights, means, precs = create_initial_gmm_parameters(
        d_z=n_dimensions,
        n_tasks=n_tasks,
        n_components=n_components,
        prior_scale=prior_scale,
        initial_var=initial_var
    )
    algorithm = MultiDaft(
        algorithm_config=algorithm_config,
        target_dist=target_dist,
        log_w_init=log_weights,
        mean_init=means,
        prec_init=precs,
    )

    fig, axes = plt.subplots(3, 4, squeeze=False, figsize=(5 * 3, 10))

    itr = config["iterations"]
    n_plots = 10

    for i in range(itr):
        elbo = algorithm.step()
        if (i + 1) % (itr // n_plots) == 0:
            plot2d_matplotlib(
                algorithm.target_dist,
                algorithm.model,
                fig,
                axes,
                min_x=-5,
                max_x=5,
                min_y=-5,
                max_y=5,
            )
        wandb.log({
            "train_loss": elbo,
        })

    wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="config_star")
def toy(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    config_dict = OmegaConf.to_container(cfg.tgt, resolve=True)

    set_seed(cfg.tgt.seed)

    group_name = f"{cfg['exp_name']}"
    run_name = f"seed_{cfg.tgt.seed}_itr_{cfg.tgt.iterations}"
    wandb.init(project="spiral_gmm_target", group=group_name, config=config_dict, name=run_name)

    daft_star(config_dict)
    wandb.finish()


if __name__ == "__main__":
    toy()
