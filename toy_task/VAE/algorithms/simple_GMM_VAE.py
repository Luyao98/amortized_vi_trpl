import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import random
import wandb

from toy_task.VAE.model.model import VAE
# from toy_task.VAE.model.model_original import VAE_original
from toy_task.VAE.utils.plot import plot_2d_encoder, show_samples, plot_generated_images, visualize_latent_space
from toy_task.GMM.utils.network_utils import initialize_weights


def load_amat(file_path):
    return np.loadtxt(file_path, dtype=np.float32)

class BinarizedMNISTDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def loss_function(log_gates, log_responsibility, kld, likelihood, beta):
    recon_loss = (log_responsibility + likelihood).mean(-1)
    loss = torch.exp(log_gates) * (beta * kld + log_gates - recon_loss)
    # loss = beta * kld + log_gates + recon_loss
    return loss.mean()


def train(epoch, model, optimizer, train_loader, device, train_loss_list, config):

    train_loss = 0

    # store init state
    old_model_state = model.state_dict()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        if model.projection:
            # load old model to get old distribution
            old_model = VAE(input_dim=784,
                            hidden_dim=config["hidden_dim"],
                            latent_dim=config["latent_dim"],
                            n_components=config["gmm_components"],
                            encoder_layer_1=config["encoder_layer_1"],
                            encoder_layer_2=config["encoder_layer_2"],
                            decoder_layer=config["decoder_layer"],
                            n_samples=config["n_samples"],
                            projection=config["projection"],
                            eps_means=config["eps_means"],
                            eps_chols=config["eps_chols"],
                            alpha=config["alpha"]).to(device)
            old_model.load_state_dict(old_model_state)
            old_model.eval()
            with torch.no_grad():
                _, means_old, chols_old= old_model.encoder(data)
                model.old_means = means_old
                model.old_chols = chols_old

        optimizer.zero_grad()
        log_gates, _, _, log_responsibility, kld, likelihood, _ = model(data)

        # beta scheduler
        if config["beta_scheduler"]:
            current_beta = beta_scheduler(epoch, config["beta"], config["epochs"])
        else:
            current_beta = config["beta"]

        loss = loss_function(log_gates, log_responsibility, kld, likelihood, current_beta)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        train_loss += loss.item()
        optimizer.step()
        wandb.log({
            "negative ELBO": loss.item(),
            "kld": kld.mean().item(),
            "likelihood": likelihood.mean().item(),
            "log_responsibility": log_responsibility.mean().item(),
            "log_gates": log_gates.mean().item(),
            "current_beta": current_beta
        })

    avg_loss = train_loss / len(train_loader.dataset)
    train_loss_list.append(avg_loss)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')

def test(model, test_loader, device, beta):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            log_gates, _, _, log_responsibility, kld, likelihood, _ = model(data)
            test_loss += loss_function(log_gates, log_responsibility, kld, likelihood, beta).item()

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

    random_idx = random.randint(0, len(test_loader) - 1)
    data = list(test_loader)[3].to(device)
    # log_px = marginal_likelihood(model, data[:8], device)
    # wandb.log({"log p(x)": log_px.item()})
    show_samples(model, data)
    # plot_2d_encoder(model, data[:3], min_x=-1.5, max_x=1.5, min_y=-1.5, max_y=1.5)


# def show_samples(model, data):
#     log_gates, _, _, _, _, _, recon = model(data)
#     avg_recon = torch.exp(log_gates).unsqueeze(-1) * recon
#     avg_recon = avg_recon.sum(1).view(-1, 28, 28).detach().cpu().numpy()
#     data = data.view(-1, 28, 28).detach().cpu().numpy()
#
#     n = min(len(data), 8)
#     fig = plt.figure(figsize=(15, 3))
#     for i in range(n):
#         ax = plt.subplot(2, n, i + 1)
#         plt.imshow(data[i], cmap='gray')
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
#         ax = plt.subplot(2, n, i + 1 + n)
#         plt.imshow(avg_recon[i], cmap='gray')
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.show()
#     wandb.log({"model output": wandb.Image(fig)})
#     plt.close(fig)

def sample_from_prior(model, device, n_samples=1000):
    z_samples = torch.randn(n_samples, model.latent_dim).to(device)
    return z_samples

def marginal_likelihood(model, x, device):
    """
    Compute p(x) >= E_q(z)[p(x|z)]
    :param model:
    :param x: input data with shape (batch_size, input_dim)
    :param device:
    return
    """
    batch_size = x.shape[0]
    n_samples = 1000
    latent_samples = torch.randn(batch_size, n_samples, model.latent_dim).to(device)
    recon_mean = model.decoder(latent_samples.view(-1, model.latent_dim)).view(batch_size, n_samples, -1)
    marginal_log_likelihood = torch.distributions.Normal(recon_mean, torch.ones_like(recon_mean)).log_prob(x.unsqueeze(1))
    return marginal_log_likelihood.mean()

def beta_scheduler(epoch, beta, max_epochs):
    return min(beta * (epoch / max_epochs), beta)

def vae(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = VAE(input_dim=784,
                hidden_dim=config["hidden_dim"],
                latent_dim=config["latent_dim"],
                n_components=config["gmm_components"],
                encoder_layer_1=config["encoder_layer_1"],
                encoder_layer_2=config["encoder_layer_2"],
                decoder_layer=config["decoder_layer"],
                n_samples=config["n_samples"],
                projection=config["projection"],
                eps_means=config["eps_means"],
                eps_chols=config["eps_chols"],
                alpha=config["alpha"]).to(device)
    # initialize_weights(model, initialization_type="xavier", preserve_bias_layers=['fc_mean', 'fc_gate'])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # load dataset
    current_dir = os.path.dirname(__file__)
    train_data = load_amat(os.path.join(current_dir, '../data/binarized_mnist_train.amat'))
    # valid_data = load_amat(os.path.join(current_dir, '../data/binarized_mnist_valid.amat'))
    test_data = load_amat(os.path.join(current_dir, '../data/binarized_mnist_test.amat'))

    train_dataset = BinarizedMNISTDataset(train_data)
    # valid_dataset = BinarizedMNISTDataset(valid_data)
    test_dataset = BinarizedMNISTDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    test_data = list(test_loader)[3].to(device)

    # train model
    train_loss_list = []
    for epoch in range(config["epochs"]):
        model.eval()
        with torch.no_grad():
            plot_2d_encoder(model, test_data[:3], min_x=-2.5, max_x=2.5, min_y=-2.5, max_y=2.5)

        model.train()
        train(epoch, model, optimizer, train_loader, device, train_loss_list, config)

    # plot training loss curve
    # plt.figure()
    # plt.plot(range(config["epochs"]), train_loss_list, label='Training Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Curve')
    # plt.legend()
    # plt.show()

    test(model, test_loader, device, config["beta"])
    plot_generated_images(model, device, n=20)
    # visualize_latent_space(model, latent_dim=config["latent_dim"],device=device)


    # generate_and_save_images(model, test_loader, device, config["output_dir"])
    #
    # # calculate FID score
    # real_dir = os.path.join(config["output_dir"], "real")
    # fake_dir = os.path.join(config["output_dir"], "fake")
    # fid_score_value = calculate_fid(real_dir, fake_dir, device)
    # print(f'FID score: {fid_score_value:.4f}')
    #
    # wandb.log({
    #     "fid_score_value": fid_score_value.item(),
    # })

if __name__ == "__main__":

    # configuration parameters
    config = {
        "batch_size": 100,
        "latent_dim": 2,
        "gmm_components": 3,
        "hidden_dim": 400,
        "encoder_layer_1": 1,
        "encoder_layer_2": 3,
        "decoder_layer": 1,
        "n_samples": 1,
        "beta": 1,
        "beta_scheduler": True,
        "epochs": 10,
        "learning_rate": 1e-3,
        "projection": False,
        "eps_means": 0.01,
        "eps_chols": 0.001,
        "alpha": 10,
        "output_dir": "output_images_1",
        "wandb_project": "VAE",
        "wandb_group": "simple_GMM_VAE",
        "wandb_run_name": "init_try"
    }
    wandb.init(project=config["wandb_project"], group=config["wandb_group"], config=config,
               name=config["wandb_run_name"])
    vae(config)
