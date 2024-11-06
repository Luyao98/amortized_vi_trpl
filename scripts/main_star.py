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

    project = "bernhard_final_star_target"
    # group_name = f"{cfg['exp_name']}_mean_bound_{cfg.star_target.projection.eps_mean}_cov_bound_{cfg.star_target.projection.eps_cov}"
    # group_name = f"sweeper_{cfg['exp_name']}_gate_strategy_{cfg.star_target.training_config.adaption.gate_strategy}_history_size_{cfg.star_target.training_config.adaption.history_size}"
    group_name = f"{cfg['exp_name']}_projection_{cfg.star_target.projection.component_project}_adaption_{cfg.star_target.training_config.adaption.adapt}_itr_{cfg.star_target.training_config.adaption.itr}"
    # run_name = f"seed_{cfg.star_target.seed}_mean_bound_{cfg.star_target.projection.eps_mean}_cov_bound_{cfg.star_target.projection.eps_cov}_alpha_{cfg.star_target.projection.alpha}"
    run_name = f"seed_{cfg.star_target.seed}"
    wandb.init(project=project, group=group_name, config=config_dict, name=run_name)

    toy_task(config_dict)

    wandb.finish()


if __name__ == "__main__":
    star()
