from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from toy_task.GMM.algorithms.algorithm import toy_task
# from toy_task.GMM.algorithms.algorithm_direct import toy_task_2
# from toy_task.GMM.algorithms.algorithm_stl import toy_task_3
# from toy_task.GMM.algorithms.algorithm_p import toy_task_p
from toy_task.GMM.utils.network_utils import set_seed


@hydra.main(version_base=None, config_path="toy_task/GMM/conf", config_name="config_funnel")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed.seed)
    config_dict = {
        **OmegaConf.to_container(cfg.target, resolve=True),
        **OmegaConf.to_container(cfg.schema, resolve=True)
    }

    group_name = f"{cfg['exp_name']}_{cfg.target.model_name}_sweep_projection_parameters"
    # run_name = f"seed_{cfg.seed.seed}"
    run_name = f"seed_{cfg.seed.seed}_mean_{cfg.schema.eps_mean}_cov_{cfg.schema.eps_cov}_alpha_{cfg.schema.alpha}"
    wandb.init(project="toy_task", group=group_name, config=config_dict, name=run_name)

    #1: decomposition; 2: direct; 3: stl; p: decom + gate projection
    toy_task(config_dict)
    # toy_task_2(config_dict)
    # toy_task_3(config_dict)
    # toy_task_p(config_dict)

    wandb.finish()


if __name__ == "__main__":
    my_app()
