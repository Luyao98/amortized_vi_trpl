import numpy as np
import torch as ch
from torch.distributions import Normal, MultivariateNormal
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA

import wandb
import io
from PIL import Image


def show_samples(model, data):
    _, _, _, _, _, _, recon = model(data)
    batch_size, n_components, _, _ = recon.size()
    recon = recon[:,:,0].view(batch_size, n_components, 28, 28).detach().cpu().numpy()
    data = data.view(-1, 28, 28).detach().cpu().numpy()

    n = min(batch_size, 8)

    fig = plt.figure(figsize=(15, 2 * n_components))

    for i in range(n):
        ax = plt.subplot(n_components + 1, n, i + 1)
        plt.imshow(data[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for j in range(n_components):
            ax = plt.subplot(n_components + 1, n, (j + 1) * n + i + 1)
            component_image = recon[i, j]
            plt.imshow(component_image, cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.tight_layout()
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
            z_sampled = ch.tensor([[xi, yi]], dtype=ch.float32).to(device)
            with ch.no_grad():
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
    # plt.show()
    wandb.log({"decoder output": wandb.Image(fig)})
    plt.close(fig)


def visualize_latent_space(model, latent_dim, device, grid_size=20, figsize=(10, 10)):
    z_samples = np.random.normal(size=(grid_size * grid_size, latent_dim))
    z_samples = ch.tensor(z_samples, dtype=ch.float32, device=device)
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


def plot_2d_encoder(model,
                    x,
                    min_x: float,
                    max_x: float,
                    min_y: float,
                    max_y: float,
                    normalize_output=False,
                    device: str = "cpu"):
    # get data for plotting
    data = compute_data_for_plot(model,
                                 x,
                                 normalize_output=normalize_output,
                                 device=device,
                                 min_x=min_x,
                                 max_x=max_x,
                                 min_y=min_y,
                                 max_y=max_y)
    n_tasks = data["n_tasks"]
    n_components = data["n_components"]
    n_plt = data["n_plt"]
    xx = data["xx"]
    yy = data["yy"]
    p_tgt = data["p_tgt"]
    p_model = data["p_model"]
    locs = data["locs"]
    scale_trils = data["scale_trils"]
    weights = data["weights"]
    # plot
    fig, axes = plt.subplots(3, n_tasks, figsize=(15, 20))
    for l in range(n_tasks):
        # plot model distribution with background target distribution
        if n_tasks == 1:
            ax = axes[0]
        else:
            ax = axes[0, l]
        ax.clear()
        ax.contourf(xx, yy, p_tgt[l].reshape(n_plt, n_plt), levels=100)
        colors = []
        for k in range(n_components):
            color = next(ax._get_lines.prop_cycler)["color"]
            colors.append(color)
            cur_scale_tril = scale_trils[l, k]
            cur_loc = locs[l, k]
            ax.scatter(x=cur_loc[0], y=cur_loc[1])
            # ellipses = compute_gaussian_ellipse(cur_loc[:2], cur_scale_tril[:2, :2])  # modification for funnel
            ellipses = compute_gaussian_ellipse(cur_loc, cur_scale_tril)
            ax.plot(ellipses[0, :], ellipses[1, :], color=color)
        ax.axis("scaled")
        ax.set_title("encoder density with prior as background")
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")

        # plot model distribution with background model distribution
        if n_tasks == 1:
            ax = axes[1]
        else:
            ax = axes[1, l]
        ax.clear()
        ax.contourf(xx, yy, p_model[l].reshape(n_plt, n_plt), levels=100)
        ax.axis("scaled")
        ax.set_title("encoder density with itself as background")
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")

        # plot weights
        if n_tasks == 1:
            ax = axes[2]
        else:
            ax = axes[2, l]
        ax.clear()
        ax.pie(weights[l], labels=[f"{w * 100:.2f}%" for w in weights[l]], colors=colors)
        ax.axis("scaled")
        ax.set_title("encoder's weights")

    # ax = axes[0, -1]
    # # color bar of last target density
    # cbar = plt.colorbar(contour_plot, cax=ax)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    # Create an image from BytesIO object
    image = Image.open(img_buf)
    wandb.log({"Plot": [wandb.Image(image, caption="encoder distribution")]})
    fig.tight_layout()
    # plt.show()
    plt.close(fig)


def compute_data_for_plot(model,
                          contexts,
                          min_x: float,
                          max_x: float,
                          min_y: float,
                          max_y: float,
                          normalize_output=False,
                          device: str = "cpu") -> dict:
    # create meshgrid
    n_plt = 100

    log_gates, means, scale_trils = model.encoder(contexts)

    # determine n_task. i.e. n_contexts
    n_tasks, n_components, _ = means.shape

    x = np.linspace(min_x, max_x, n_plt)
    y = np.linspace(min_y, max_y, n_plt)
    xx, yy = np.meshgrid(x, y)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    xy = ch.tensor(xy, dtype=ch.float32).to(device)  # (n_plt**2, 2)

    with ch.no_grad():
        # log_p_tgt is log prior
        log_p_tgt = Normal(0,1).log_prob(xy).sum(-1).unsqueeze(0).expand(n_tasks, -1)
        # log_component with shape (b, o, s)
        log_component = MultivariateNormal(means.unsqueeze(2), scale_tril=scale_trils.unsqueeze(2)).log_prob(xy.unsqueeze(0).unsqueeze(0).to(means.device))  # (b, o, s=n_plt**2)
        log_p_model = ch.logsumexp(log_component + log_gates.unsqueeze(-1), dim=1)  # (b, s=n_plt**2)
        # log_p_model = model.encoder.log_prob_gmm(means, scale_trils, log_gates, xy.unsqueeze(0).expand(n_tasks, -1, -1).to(means.device))
    if normalize_output:
        # maximum is now 0, so exp(0) = 1
        log_p_tgt -= log_p_tgt.max()
    log_p_model = log_p_model.detach().to("cpu").numpy()
    log_p_tgt = log_p_tgt.detach().to("cpu").numpy()
    p_tgt = np.exp(log_p_tgt)
    p_model = np.exp(log_p_model)
    # extract gmm parameters
    locs = means.detach().to("cpu").numpy()
    scale_trils = scale_trils.detach().to("cpu").numpy()
    weights = np.exp(log_gates.detach().to("cpu").numpy())

    return {
        "n_tasks": n_tasks,
        "n_components": n_components,
        "n_plt": n_plt,
        "xx": xx,
        "yy": yy,
        "p_tgt": p_tgt,
        "log_p_tgt": log_p_tgt,
        "p_model": p_model,
        "log_p_model": log_p_model,
        "locs": locs,
        "scale_trils": scale_trils,
        "weights": weights,
    }


def compute_gaussian_ellipse(mean, scale_tril) -> np.ndarray:
    n_plot = 100
    evals, evecs = np.linalg.eig(scale_tril @ scale_tril.T)
    theta = np.linspace(0, 2 * np.pi, n_plot)
    ellipsis = (np.sqrt(evals[None, :]) * evecs) @ [np.sin(theta), np.cos(theta)]
    ellipsis = ellipsis + mean[:, None]
    return ellipsis