import torch as ch
import wandb

from toy_task.GMM.models.model_factory import get_model
from toy_task.GMM.targets.GMM_target import get_gmm_target
from toy_task.GMM.algorithms.algorithm import train_model, plot


if __name__ == "__main__":
    # Device
    device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Training parameters
    n_epochs = 150
    batch_size = 64
    n_context = 640
    n_components = 4
    n_samples = 15
    fc_layer_size = 256
    init_lr = 0.01
    eps_mean = 0.1       # mean projection bound
    eps_cov = 0.5       # cov projection bound
    alpha = 2            # regression penalty

    project = False        # calling projection or not

    model_name = "toy_task_model"
    initialization_type = "xavier"

    init_bias_mean_list = [
        [10.0, 10.0],
        [-10.0, -10.0],
        [10.0, -10.0],
        [-10.0, 10.0]
    ]
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
        "alpha": alpha
    })
    config = wandb.config

    # Target
    target = get_gmm_target(n_components)

    # Model
    model = get_model(model_name,
                      device,
                      fc_layer_size,
                      n_components,
                      init_bias_mean_list,
                      initialization_type)

    # Training
    train_model(model, target,
                n_epochs, batch_size, n_context, n_components, n_samples,  # training hyperparameter
                eps_mean, eps_cov, alpha,  # projection hyperparameter
                init_lr, device, project)

    # Plotting
    plot(model, target)
