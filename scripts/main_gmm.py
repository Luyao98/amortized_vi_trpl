from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from toy_task.GMM.algorithms.algorithm import toy_task
from toy_task.GMM.utils.network_utils import set_seed


@hydra.main(version_base=None, config_path="../toy_task/GMM/conf", config_name="config_gmm")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.gmm_target.seed)
    config_dict = {
        **OmegaConf.to_container(cfg.gmm_target, resolve=True)
    }
    # group_name = f"sweeper_{cfg['exp_name']}_mean_bound_{cfg.gmm_target.projection.eps_mean}_cov_bound_{cfg.gmm_target.projection.eps_cov}_alpha_{cfg.gmm_target.projection.alpha}_gate_lr_{cfg.gmm_target.training_config.gate_lr}"
    group_name = f"sweeper_{cfg['exp_name']}_gate_lr_{cfg.gmm_target.training_config.gate_lr}"
    # run_name = f"seed_{cfg.star_target.seed}"
    run_name = f"seed_{cfg.gmm_target.seed}"
    wandb.init(project="gmm_target", group=group_name, config=config_dict, name=run_name)

    toy_task(config_dict)

    wandb.finish()


if __name__ == "__main__":
    my_app()
