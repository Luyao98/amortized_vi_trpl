import torch
import numpy as np
from scipy.linalg import sqrtm

from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from PIL import Image  # Import PIL for image conversions

import os
from pytorch_fid import fid_score

# self-written method,bigger than pytorch-fid
def calculate_fid(model, real_loader, device, n_samples=10000, batch_size=128):
    model.eval()
    real_images = []
    generated_images = []

    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device)
    inception.eval()

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # MNIST is grayscale, repeat channels to match Inception input
    ])

    def get_activations(images, model):
        model.eval()
        with torch.no_grad():
            pred = model(images.to(device))
        return pred.detach().cpu().numpy()

    with torch.no_grad():
        for i, data in enumerate(real_loader):
            data = data.to(device)
            real_images.append(data)
            log_gates, _, _, _, _, recon = model(data)
            avg_recon = torch.exp(log_gates).unsqueeze(-1) * recon
            avg_recon = avg_recon.sum(1)
            generated_images.append(avg_recon)

            if len(real_images) * real_loader.batch_size >= n_samples:
                break

    real_images = torch.cat(real_images)[:n_samples]
    generated_images = torch.cat(generated_images)[:n_samples]

    real_activations = []
    generated_activations = []

    for i in range(0, len(real_images), batch_size):
        real_batch = real_images[i:i+batch_size]
        generated_batch = generated_images[i:i+batch_size]

        real_batch = torch.stack([preprocess(Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8).reshape(28, 28))) for img in real_batch])
        generated_batch = torch.stack([preprocess(Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8).reshape(28, 28))) for img in generated_batch])

        real_activations.append(get_activations(real_batch, inception))
        generated_activations.append(get_activations(generated_batch, inception))

    real_activations = np.concatenate(real_activations, axis=0)
    generated_activations = np.concatenate(generated_activations, axis=0)

    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    mu_gen = np.mean(generated_activations, axis=0)
    sigma_gen = np.cov(generated_activations, rowvar=False)

    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid


# pytorch-fid
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
