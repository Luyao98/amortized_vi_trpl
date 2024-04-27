from omegaconf import DictConfig, OmegaConf
import hydra
from toy_task.GMM.algorithms.algorithm import toy_task
from toy_task.GMM.utils.network_utils import set_seed


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed.seed)
    toy_task(n_epochs=cfg.model.n_epochs,
             batch_size=cfg.model.batch_size,
             n_context=cfg.model.n_context,
             n_components=cfg.model.n_components,
             n_samples=cfg.model.n_samples,
             fc_layer_size=cfg.model.fc_layer_size,
             init_lr=cfg.model.init_lr,
             model_name=cfg.model.model_name,
             initialization_type=cfg.model.initialization_type,
             project=cfg.schema.project,
             eps_mean=cfg.schema.eps_mean,
             eps_cov=cfg.schema.eps_cov,
             alpha=cfg.schema.alpha)


if __name__ == "__main__":
    my_app()
