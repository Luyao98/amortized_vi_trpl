from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from toy_task.GMM.algorithms.algorithm import toy_task


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    toy_task(n_epochs=cfg.db.n_epochs,
             batch_size=cfg.db.batch_size,
             n_context=cfg.db.n_context,
             n_components=cfg.db.n_components,
             n_samples=cfg.db.n_samples,
             fc_layer_size=cfg.db.fc_layer_size,
             init_lr=cfg.db.init_lr,
             model_name=cfg.db.model_name,
             initialization_type=cfg.db.initialization_type,
             project=cfg.schema.project,
             eps_mean=cfg.schema.eps_mean,
             eps_cov=cfg.schema.eps_cov,
             alpha=cfg.schema.alpha)


if __name__ == "__main__":
    my_app()
