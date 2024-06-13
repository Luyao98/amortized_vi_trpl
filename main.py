from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from toy_task.GMM.algorithms.algorithm import toy_task
from toy_task.GMM.algorithms.algorithm_direct import toy_task_2
from toy_task.GMM.algorithms.algorithm_stl import toy_task_3
from toy_task.GMM.utils.network_utils import set_seed


@hydra.main(version_base=None, config_path="conf", config_name="config_funnel")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed.seed)
    config_dict = {
        **OmegaConf.to_container(cfg.target, resolve=True),
        **OmegaConf.to_container(cfg.schema, resolve=True)
    }

    group_name = f"{cfg['exp_name']}_{cfg.target.model_name}"
    # run name for algorithm 1
    # run_name = f"num_gaussian_lr{cfg.target.gaussian_lr}_gate_lr{cfg.target.gate_lr}"
    run_name = f"seed_{cfg.seed.seed}"

    wandb.init(project="toy_task", group=group_name, config=config_dict, name=run_name)

    toy_task(config_dict)
    # toy_task_2(config_dict)
    # toy_task_3(config_dict)

    wandb.finish()


if __name__ == "__main__":
    my_app()
