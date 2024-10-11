from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from toy_task.GMM.algorithms.algorithm import toy_task
from toy_task.GMM.utils.network_utils import set_seed


@hydra.main(version_base=None, config_path="../toy_task/GMM/conf", config_name="config_star")
def star(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    set_seed(cfg.star_target.seed)
    config_dict = {
        **OmegaConf.to_container(cfg.star_target, resolve=True)
    }

    group_name = f"sweeper_{cfg['exp_name']}_alpha_{cfg.star_target.projection.alpha}"
    # run_name = f"seed_{cfg.star_target.seed}"
    run_name = f"seed_{cfg.star_target.seed}_mean_bound_{cfg.star_target.projection.eps_mean}_cov_bound_{cfg.star_target.projection.eps_cov}"
    wandb.init(project="star_target", group=group_name, config=config_dict, name=run_name)

    toy_task(config_dict)

    wandb.finish()


if __name__ == "__main__":
    star()
