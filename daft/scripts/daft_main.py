from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch as ch
from matplotlib import pyplot as plt


from daft.src.multi_daft_vi.lnpdf import LNPDF, make_contextual_star_target, make_contextual_gmm_target
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
        initial_var=initial_var,
        target_dist=target_dist
    )
    algorithm = MultiDaft(
        algorithm_config=algorithm_config,
        target_dist=target_dist,
        log_w_init=log_weights,
        mean_init=means,
        prec_init=precs,
    )

    fig, axes = plt.subplots(1, 4, figsize=(15, 10))
    # intial plot
    plot2d_matplotlib(
        algorithm.target_dist,
        algorithm.model,
        fig,
        axes,
        min_x=-5,
        max_x=5,
        min_y=-5,
        max_y=5,
        logging=True
    )

    itr = config["iterations"]
    n_plots = 10
    infer_time = 0
    for i in range(itr):
        # Start timing
        start_event = ch.cuda.Event(enable_timing=True)
        end_event = ch.cuda.Event(enable_timing=True)
        start_event.record()

        elbo = algorithm.step()

        # End timing
        end_event.record()
        ch.cuda.synchronize()
        infer_time += start_event.elapsed_time(end_event)

        if (i + 1) % (itr // n_plots) == 0:
            fig, axes = plt.subplots(1, 4, figsize=(15, 10))
            plot2d_matplotlib(
                algorithm.target_dist,
                algorithm.model,
                fig,
                axes,
                min_x=-5,
                max_x=5,
                min_y=-5,
                max_y=5,
                logging=True
            )
            js_div = algorithm.evaluation(1000)
            wandb.log({
                "Jensen Shannon Divergence": js_div
            })
        wandb.log({
            "train_loss": elbo
        })
    wandb.log({
        "inference_time/ms": float(infer_time/itr)
    })
    wandb.finish()


def daft_gmm(config):
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


    target_dist: LNPDF = make_contextual_gmm_target(
        n_tasks=n_tasks,
        n_components=task_components,
    )

    log_weights, means, precs = create_initial_gmm_parameters(
        d_z=n_dimensions,
        n_tasks=n_tasks,
        n_components=n_components,
        prior_scale=prior_scale,
        initial_var=initial_var,
        target_dist=target_dist
    )
    algorithm = MultiDaft(
        algorithm_config=algorithm_config,
        target_dist=target_dist,
        log_w_init=log_weights,
        mean_init=means,
        prec_init=precs,
    )

    fig, axes = plt.subplots(1, 4, figsize=(15, 10))
    # intial plot
    plot2d_matplotlib(
        algorithm.target_dist,
        algorithm.model,
        fig,
        axes,
        min_x=-15,
        max_x=10,
        min_y=-10,
        max_y=15,
        logging=True
    )

    itr = config["iterations"]
    n_plots = 10
    n_eval = 50
    infer_time = 0

    for i in range(itr):
        # Start timing
        start_event = ch.cuda.Event(enable_timing=True)
        end_event = ch.cuda.Event(enable_timing=True)
        start_event.record()

        elbo = algorithm.step()

        # End timing
        end_event.record()
        ch.cuda.synchronize()
        infer_time += start_event.elapsed_time(end_event)

        if (i + 1) % (itr // n_plots) == 0:
            fig, axes = plt.subplots(1, 4, figsize=(15, 10))
            plot2d_matplotlib(
                algorithm.target_dist,
                algorithm.model,
                fig,
                axes,
                min_x=-15,
                max_x=10,
                min_y=-10,
                max_y=15,
                logging=True
            )
        if (i + 1) % (itr // n_eval) == 0:
            js_div = algorithm.evaluation(1000)
            wandb.log({
                "Jensen Shannon Divergence": js_div
            })
        wandb.log({
            "train_loss": elbo
        })
    wandb.log({
        "inference_time/ms": float(infer_time/itr)
    })
    wandb.finish()

@hydra.main(version_base=None, config_path="../daft_conf", config_name="config_gmm")
def toy(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    config_dict = OmegaConf.to_container(cfg.daft_target, resolve=True)

    set_seed(cfg.daft_target.seed)

    group_name = f"test_{cfg['exp_name']}"
    run_name = f"seed_{cfg.daft_target.seed}_kl_bound_{cfg.daft_target.algorithm.more.component_kl_bound}"
    wandb.init(project="gmm_target", group=group_name, config=config_dict, name=run_name)

    daft_gmm(config_dict)
    wandb.finish()


if __name__ == "__main__":
    toy()
