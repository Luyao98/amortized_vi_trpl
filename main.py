from omegaconf import DictConfig, OmegaConf
import hydra
from toy_task.GMM.algorithms.train_bm import toy_task
from toy_task.GMM.utils.network_utils import set_seed


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed.seed)
    toy_task(n_epochs=cfg.target.n_epochs,
             batch_size=cfg.target.batch_size,
             n_context=cfg.target.n_context,
             n_components=cfg.target.n_components,
             n_samples=cfg.target.n_samples,
             fc_layer_size=cfg.target.fc_layer_size,
             init_lr=cfg.target.init_lr,
             model_name=cfg.target.model_name,
             dim=cfg.target.dim,
             initialization_type=cfg.target.initialization_type,
             project=cfg.schema.project,
             eps_mean=cfg.schema.eps_mean,
             eps_cov=cfg.schema.eps_cov,
             alpha=cfg.schema.alpha)


if __name__ == "__main__":
    my_app()
