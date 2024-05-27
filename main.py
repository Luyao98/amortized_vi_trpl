from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from toy_task.GMM.algorithms.algorithm import toy_task
from toy_task.GMM.utils.network_utils import set_seed


@hydra.main(version_base=None, config_path="conf", config_name="config_bmm")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed.seed)
    config_dict = {
        **OmegaConf.to_container(cfg.target, resolve=True),
        **OmegaConf.to_container(cfg.schema, resolve=True),
        # 'seed': cfg.seed,
        # 'job_type': cfg.job_type,
        # 'exp_name': cfg.exp_name
    }

    group_name = cfg['exp_name']
    job_type = cfg.target.model_name
    run_name = f"seed_{cfg.seed.seed}_{cfg.target.model_name}_projection_{cfg.schema.project}"
    # run_name = f"mean{cfg.schema.eps_mean}_cov{cfg.schema.eps_cov}_alpha{cfg.schema.alpha}"
    # run_name = f"batch_{cfg.target.batch_size}_gate_lr_{cfg.target.gate_lr}_gauss_lr_{cfg.target.gaussian_lr}"
    wandb.init(project="ELBO", group=group_name, job_type=job_type, config=config_dict, name=run_name)

    toy_task(config_dict)

    wandb.finish()


if __name__ == "__main__":
    my_app()
