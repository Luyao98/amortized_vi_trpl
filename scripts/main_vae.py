from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

from toy_task.GMM.utils.network_utils import set_seed
from toy_task.VAE.algorithms.simple_GMM_VAE import vae


@hydra.main(version_base=None, config_path="../toy_task/VAE/conf", config_name="config_vae")
def my_vae(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # set_seed(cfg.seed.seed)
    config_dict = {
        **OmegaConf.to_container(cfg.model, resolve=True)
    }

    # group = f"test_alpha_{cfg.model.alpha}"
    group = f"with_projection_gate_layer_{cfg.model.encoder_layer_1}_component_layer_{cfg.model.encoder_layer_2}_decoder_layer_{cfg.model.decoder_layer}"
    # run_name = f"eps_means{cfg.model.eps_means}_eps_chol{cfg.model.eps_chols}"
    run_name = f"beta{cfg.model.beta}_lr_{cfg.model.learning_rate}_samples_{cfg.model.n_samples}"
    wandb.init(project="VAE", group=group, config=config_dict, name=run_name)

    vae(config_dict)
    wandb.finish()


if __name__ == "__main__":
    my_vae()
