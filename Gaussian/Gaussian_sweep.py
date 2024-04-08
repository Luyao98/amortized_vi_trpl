import torch.optim as optim
import wandb
from Gaussian.utils import initialize_weights
from Gaussian.Gaussian_targets import ConditionalGaussianTarget, get_cov_fn, get_mean_fn
from Gaussian.Gaussian_plot import gaussian_plot
from Gaussian.kl_projection import KLProjection
from Gaussian.split_kl_projection import split_projection
from Gaussian.Gaussian_model import GaussianNN, train_model


sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'training loss', 'goal': 'minimize'},
    'parameters': {
        'batch_size': {'values': [32, 64, 128]},
        'eps_mean': {'min': 0.001, 'max': 0.1, 'distribution': 'uniform'},
        'eps_cov': {'min': 0.0001, 'max': 0.01, 'distribution': 'uniform'},
        'alpha': {'min': 50, 'max': 100}
    }
}

sweep_id = wandb.sweep(sweep_config, project="ELBOopt_2D")


def train():
    wandb.init()

    config = wandb.config
    device = 'cpu'

    model = GaussianNN(64).to(device)
    initialize_weights(model, initialization_type="xavier")
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    mean_target = get_mean_fn('periodic')
    cov_target = get_cov_fn('periodic')
    target = ConditionalGaussianTarget(mean_target, cov_target)

    try:
        train_model(model, target, 50, config.batch_size, 1280, config.eps_mean,
                    config.eps_cov, config.alpha, optimizer, True, True)

        contexts = target.get_contexts_1g(5)
        gaussian_plot(model, target, contexts)
    except Exception as e:
        wandb.log({'error': str(e)})

    wandb.finish()


wandb.agent(sweep_id, function=train, count=30)

