import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random


def load_amat(file_path):
    return np.loadtxt(file_path, dtype=np.float32)


class BinarizedMNISTDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GMMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_components):
        super(GMMEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim * n_components)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim * n_components)
        self.fc_gate = nn.Linear(hidden_dim, n_components)
        self.n_components = n_components
        self.latent_dim = latent_dim

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        # h = torch.relu(self.fc2(h))
        # h = torch.relu(self.fc3(h))
        mu = self.fc_mu(h).view(-1, self.n_components, self.latent_dim)
        logvar = self.fc_logvar(h).view(-1, self.n_components, self.latent_dim)
        gate_logits = self.fc_gate(h)
        gates = torch.softmax(gate_logits, dim=1)
        return mu, logvar, gates


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        # h = torch.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_components):
        super(VAE, self).__init__()
        self.encoder = GMMEncoder(input_dim, hidden_dim, latent_dim, n_components)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.n_components = n_components
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar, gate = self.encoder(x)
        m = Categorical(gate)
        k = m.sample().unsqueeze(1).unsqueeze(2).expand(-1, 1, self.latent_dim)
        mu_selected = torch.gather(mu, 1, k).squeeze(1)
        logvar_selected = torch.gather(logvar, 1, k).squeeze(1)
        z = self.reparameterize(mu_selected, logvar_selected)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, gate, z


def loss_function(recon_x, x, mu, logvar, gates):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    KLD = torch.sum(KLD, dim=[1, 2]).mean()
    return BCE + KLD


def train(epoch, model, optimizer, train_loader, device, train_loss_list):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, gates, _ = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, gates)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # if batch_idx % 100 == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    avg_loss = train_loss / len(train_loader.dataset)
    train_loss_list.append(avg_loss)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')


def test(model, test_loader, device, num_samples=8):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon, mu, logvar, gates, _ = model(data)
            test_loss += loss_function(recon, data, mu, logvar, gates).item()

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

    # show samples
    random_idx = random.randint(0, len(test_loader) - 1)
    data = list(test_loader)[random_idx].to(device)
    recon, _, _, _, _ = model(data)
    recon = recon.detach().view(-1, 28, 28).cpu().numpy()
    data = data.detach().view(-1, 28, 28).cpu().numpy()

    n = min(len(data), num_samples)
    plt.figure(figsize=(15, 3))
    for i in range(n):
        # original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(data[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # reconstructed image
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recon[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# model parameters
batch_size = 128
latent_dim = 20
gmm_components = 10
hidden_dim = 400
epochs = 100
learning_rate = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(input_dim=784, hidden_dim=hidden_dim, latent_dim=latent_dim, n_components=gmm_components).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# load dataset
# current_dir = os.path.dirname(__file__)
# train_data = load_amat(os.path.join(current_dir, '../data/binarized_mnist_train.amat'))
# valid_data = load_amat(os.path.join(current_dir, '../data/binarized_mnist_valid.amat'))
# test_data = load_amat(os.path.join(current_dir, '../data/binarized_mnist_test.amat'))
train_data = load_amat('../data/binarized_mnist_train.amat')
valid_data = load_amat('../data/binarized_mnist_valid.amat')
test_data = load_amat('../data/binarized_mnist_test.amat')


train_dataset = BinarizedMNISTDataset(train_data)
valid_dataset = BinarizedMNISTDataset(valid_data)
test_dataset = BinarizedMNISTDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# train model
train_loss_list = []
for epoch in range(1, epochs + 1):
    train(epoch, model, optimizer, train_loader, device, train_loss_list)

# plot training loss curve
plt.figure()
plt.plot(range(1, epochs + 1), train_loss_list, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

test(model, test_loader, device)
