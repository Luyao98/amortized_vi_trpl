import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA
import random
import wandb
from PIL import Image
from pytorch_fid import fid_score
from toy_task.VAE.model.model import VAE
from toy_task.VAE.model.model_original import VAE_original
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

def loss_function(log_gates, log_responsibility, kld, likelihood, beta=0.1):
    loss = torch.exp(log_gates) * (beta * kld + log_gates - log_responsibility - likelihood)
    # loss = likelihood + kld
    return loss.mean()


def train(epoch, model, optimizer, train_loader, device, train_loss_list, config):
    model.train()
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
                            n_samples=1,
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
        log_gates, z, log_responsibility, kld, likelihood, _ = model(data)
        loss = loss_function(log_gates, log_responsibility, kld, likelihood)
        loss.backward()
        wandb.log({
            "negative ELBO": loss.item(),
            "kld": kld.mean().item(),
            "likelihood": likelihood.mean().item(),
            # "log_responsibility": log_responsibility.mean().item()
        })
        train_loss += loss.item()
        optimizer.step()

    avg_loss = train_loss / len(train_loader.dataset)
    train_loss_list.append(avg_loss)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            log_gates, z, log_responsibility, kld, likelihood, _ = model(data)
            test_loss += loss_function(log_gates, log_responsibility, kld, likelihood).item()

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

    show_samples(model, test_loader, device)

def show_samples(model, test_loader, device):
    random_idx = random.randint(0, len(test_loader) - 1)
    data = list(test_loader)[random_idx].to(device)
    log_gates, _, _, _, _, recon = model(data)
    avg_recon = torch.exp(log_gates).unsqueeze(-1) * recon
    avg_recon = avg_recon.sum(1).view(-1, 28, 28).detach().cpu().numpy()
    data = data.view(-1, 28, 28).detach().cpu().numpy()

    n = min(len(data), 8)
    fig = plt.figure(figsize=(15, 3))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(data[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(avg_recon[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    wandb.log({"model output": wandb.Image(fig)})
    plt.close(fig)

def plot_generated_images(model, device, n):
    digit_size = 28
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    model.eval()
    figure = np.zeros((digit_size * n, digit_size * n))
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sampled = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
            with torch.no_grad():
                decode = model.decoder(z_sampled).cpu().numpy()
                digit = decode.reshape((digit_size, digit_size))
                figure[
                    i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size
                ] = digit
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="Greys_r")
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()
    wandb.log({"decoder output": wandb.Image(fig)})
    plt.close(fig)

def visualize_latent_space(model, latent_dim, device, grid_size=20, figsize=(10, 10)):
    z_samples = np.random.normal(size=(grid_size * grid_size, latent_dim))
    z_samples = torch.tensor(z_samples, dtype=torch.float32, device=device)
    x_decoded = model.decoder(z_samples).cpu().detach().numpy()

    z_samples_np = z_samples.cpu().numpy()
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_samples_np)

    figure = np.zeros((28 * grid_size, 28 * grid_size))

    z_pca_min = z_pca.min(0)
    z_pca_max = z_pca.max(0)
    z_pca = (z_pca - z_pca_min) / (z_pca_max - z_pca_min)
    z_pca = (z_pca * (grid_size - 1)).astype(int)

    for idx, (i, j) in enumerate(z_pca):
        digit = x_decoded[idx].reshape(28, 28)
        figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

    fig = plt.figure(figsize=figsize)
    plt.imshow(figure, cmap='Greys_r')
    plt.title('Latent Space Visualization')
    plt.show()
    wandb.log({"decoder output": wandb.Image(fig)})
    plt.close(fig)

# def plot_latent_space(model, test_loader, device):
#     model.eval()
#     z_means = []
#     with torch.no_grad():
#         for data in test_loader:
#             data = data.to(device)
#             _, mean, _ = model.encoder(data)
#             z_means.append(mean.squeeze(1).cpu().numpy())
#     z_means = np.concatenate(z_means, axis=0)
#     plt.figure(figsize=(10, 8))
#     plt.scatter(z_means[:, 0], z_means[:, 1], s=2)
#     plt.xlabel('z1')
#     plt.ylabel('z2')
#     plt.title('Latent Space')
#     plt.show()

def save_images(images, directory, size=(299, 299)):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, img in enumerate(images):
        img = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8).reshape(28, 28))
        img = img.resize(size).convert("RGB")  # Resize and convert to RGB
        img.save(os.path.join(directory, f"{i}.png"))

def generate_and_save_images(model, data_loader, device, output_dir, n_samples=1000):
    model.eval()
    real_images = []
    generated_images = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(device)
            real_images.append(data)
            log_gates, _, _, _, _, recon = model(data)
            avg_recon = torch.exp(log_gates).unsqueeze(-1) * recon
            avg_recon = avg_recon.sum(1)
            generated_images.append(avg_recon)

            if len(real_images) * data_loader.batch_size >= n_samples:
                break

    real_images = torch.cat(real_images)[:n_samples]
    generated_images = torch.cat(generated_images)[:n_samples]

    save_images(real_images, os.path.join(output_dir, "real"))
    save_images(generated_images, os.path.join(output_dir, "fake"))

def calculate_fid(real_dir, fake_dir, device):
    fid_value = fid_score.calculate_fid_given_paths([real_dir, fake_dir], batch_size=50, device=device, dims=2048)
    return fid_value

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
                n_samples=1,
                projection=config["projection"],
                eps_means=config["eps_means"],
                eps_chols=config["eps_chols"],
                alpha=config["alpha"]).to(device)
    initialize_weights(model, initialization_type="xavier", preserve_bias_layers="fc_gate")
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # load dataset
    current_dir = os.path.dirname(__file__)
    train_data = load_amat(os.path.join(current_dir, '../data/binarized_mnist_train.amat'))
    valid_data = load_amat(os.path.join(current_dir, '../data/binarized_mnist_valid.amat'))
    test_data = load_amat(os.path.join(current_dir, '../data/binarized_mnist_test.amat'))

    train_dataset = BinarizedMNISTDataset(train_data)
    valid_dataset = BinarizedMNISTDataset(valid_data)
    test_dataset = BinarizedMNISTDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # train model
    train_loss_list = []
    for epoch in range(1, config["epochs"] + 1):
        train(epoch, model, optimizer, train_loader, device, train_loss_list, config)

    # plot training loss curve
    plt.figure()
    plt.plot(range(1, config["epochs"] + 1), train_loss_list, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

    test(model, test_loader, device)
    # plot_generated_images(model, device, n=20)
    visualize_latent_space(model, latent_dim=config["latent_dim"],device=device)
    # plot_latent_space(model, test_loader, device)

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
        "gmm_components": 10,
        "hidden_dim": 400,
        "encoder_layer_1": 3,
        "encoder_layer_2": 4,
        "decoder_layer": 3,
        "epochs": 30,
        "learning_rate": 1e-3,
        "projection": False,
        "eps_means": 0.1,
        "eps_chols": 0.1,
        "alpha": 10,
        "output_dir": "output_images_1",
        "wandb_project": "VAE",
        "wandb_group": "simple_GMM_VAE",
        "wandb_run_name": "init_try"
    }
    wandb.init(project=config["wandb_project"], group=config["wandb_group"], config=config,
               name=config["wandb_run_name"])
    vae(config)

