from cw2 import cluster_work, cw_error, experiment
from cw2.cw_data import cw_logging

import torch as ch
import wandb

from toy_task.GMM.models.model_factory import get_model
from toy_task.GMM.targets.GMM_target import get_gmm_target
from toy_task.GMM.algorithms.algorithm import train_model, plot


def toy_task(config: dict):
    # Device
    device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Training parameters
    configs = config["params"]
    n_epochs = configs["n_epochs"]
    batch_size = configs["batch_size"]
    n_context = configs["n_context"]
    n_components = configs["n_components"]
    n_samples = configs["n_samples"]
    fc_layer_size = configs["fc_layer_size"]
    init_lr = configs["init_lr"]
    eps_mean = configs["eps_mean"]       # mean projection bound
    eps_cov = configs["eps_cov"]       # cov projection bound
    alpha = configs["alpha"]            # regression penalty

    project = configs["project"]        # calling projection or not

    model_name = configs["model_name"]
    initialization_type = configs["initialization_type"]

    # Wandb
    wandb.init(project="ELBOopt_GMM", config={
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "n_context": n_context,
        "n_components": n_components,
        "fc_layer_size": fc_layer_size,
        "init_lr": init_lr,
        "eps_mean": eps_mean,
        "eps_cov": eps_cov,
        "alpha": alpha,
        "project": project,
        "model_name": model_name,
        "initialization_type": initialization_type})

    # Target
    target = get_gmm_target(n_components)

    # Model
    model = get_model(model_name,
                      device,
                      fc_layer_size,
                      n_components,
                      initialization_type)

    # Training
    train_model(model, target,
                n_epochs, batch_size, n_context, n_components, n_samples,  # training hyperparameter
                eps_mean, eps_cov, alpha,  # projection hyperparameter
                init_lr, device, project)

    # Plotting
    plot(model, target)


class MyExperiment(experiment.AbstractExperiment):
    def initialize(
        self, config: dict, rep: int, logger: cw_logging.LoggerArray
    ) -> None:
        cw_logging.getLogger().info(
            "Ready to start repetition {}. Resetting everything.".format(rep)
        )

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # # Do Something non-iteratively and logging the result.
        # cw_logging.getLogger().info("Doing Something.")
        # logger.process("Some Result")
        # cw_logging.getLogger().warning("Something went wrong")
        print(config)
        toy_task(config)

    def finalize(
        self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False
    ):
        if surrender is not None:
            cw_logging.getLogger().info("Run was surrendered early.")

        if crash:
            cw_logging.getLogger().warning("Run crashed with an exception.")
        cw_logging.getLogger().info("Finished. Closing Down.")


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(MyExperiment)
    cw.run()
