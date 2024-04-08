import torch.optim as optim
import wandb
from Gaussian.utils import initialize_weights
from GMM.GMM_target import ConditionalGMMTarget, get_cov_fn, get_mean_fn
from GMM.GMM_simple_plot import gaussian_simple_plot
from GMM.GMM_model import ConditionalGMM, train_model


sweep_config = {
    'method': 'random',
    'metric': {'name': 'kl loss', 'goal': 'minimize'},
    'parameters': {
        'batch_size': {'values': [32, 64, 128]},
        'eps_mean': {'values': [0.001, 0.005, 0.01, 0.05, 0.01, 0.05, 0.1]},
        'eps_cov': {'values': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]},
        'alpha': {'values': [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="ELBOopt_GMM")


def train():
    wandb.init()

    config = wandb.config
    device = 'cuda'

    model = ConditionalGMM(64, 1).to(device)
    initialize_weights(model, initialization_type="xavier")
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    mean_target = get_mean_fn(1)
    cov_target = get_cov_fn(1)
    target = ConditionalGMMTarget(mean_target, cov_target)

    try:
        train_model(model, target, 100, config.batch_size, 1280, 1, config.eps_mean,
                    config.eps_cov, config.alpha, optimizer, device)

        contexts = target.get_contexts_gmm(5).to('cpu')
        gaussian_simple_plot(model.to('cpu'), target, contexts)
    except Exception as e:
        wandb.log({'error': str(e)})

    wandb.finish()


wandb.agent(sweep_id, function=train, count=100)
