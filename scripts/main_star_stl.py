from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from toy_task.GMM.algorithms.algorithm_stl import toy_task_stl
from toy_task.GMM.utils.network_utils import set_seed


@hydra.main(version_base=None, config_path="../toy_task/GMM/conf", config_name="config_star_stl")
def star(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.star_target.seed)
    config_dict = {
        **OmegaConf.to_container(cfg.star_target, resolve=True)
    }

    project = "dalmatian_final_star_target"
    group_name = f"{cfg['exp_name']}"
    run_name = f"seed_{cfg.star_target.seed}"
    wandb.init(project=project, group=group_name, config=config_dict, name=run_name)

    toy_task_stl(config_dict)

    wandb.finish()


if __name__ == "__main__":
    star()
