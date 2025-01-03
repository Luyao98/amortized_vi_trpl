import numpy as np
import torch as ch
import matplotlib.pyplot as plt


from toy_task.GMM.utils.torch_utils import get_numpy

import wandb
import io
from PIL import Image

"""
the current version for GMM case plotting, based on the code from Philipp Dahlinger.
"""


def plot2d_matplotlib(
        target_dist,
        model,
        contexts,
        plot_type,
        normalize_output=False,
        device: str = "cpu",
        min_x: int or None = None,
        max_x: int or None = None,
        min_y: int or None = None,
        max_y: int or None = None,
):
    model.eval()
    # get data for plotting
    data = compute_data_for_plot(
        target_dist,
        model,
        contexts,
        normalize_output=normalize_output,
        device=device,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
    )
    n_contexts = data["n_contexts"]
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
    if n_contexts == 1:
        fig, axes = plt.subplots(n_contexts, 4, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(4, n_contexts, figsize=(15, 20))

    for l in range(n_contexts):
        # plot target distribution
        ax = axes[0] if n_contexts == 1 else axes[0, l]
        ax.clear()
        contour_plot = ax.contourf(xx, yy, p_tgt[l].reshape(n_plt, n_plt), levels=100)
        ax.axis("scaled")
        ax.set_title(f"Target Density")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        # plot model distribution with background target distribution
        ax = axes[1] if n_contexts == 1 else axes[1, l]
        ax.clear()
        ax.contourf(xx, yy, p_tgt[l].reshape(n_plt, n_plt), levels=100)
        colors = []
        for k in range(n_components):
            color = next(ax._get_lines.prop_cycler)["color"]
            colors.append(color)
            cur_scale_tril = scale_trils[l, k]
            cur_loc = locs[l, k]
            ax.scatter(x=cur_loc[0], y=cur_loc[1])
            ellipses = compute_gaussian_ellipse(cur_loc[:2], cur_scale_tril[:2, :2])  # modification for funnel
            ax.plot(ellipses[0, :], ellipses[1, :], color=color)
        ax.axis("scaled")
        ax.set_title("Comparison")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        # plot model distribution with background model distribution
        ax = axes[2] if n_contexts == 1 else axes[2, l]
        ax.clear()
        ax.contourf(xx, yy, p_model[l].reshape(n_plt, n_plt), levels=100)
        ax.axis("scaled")
        ax.set_title("Model Density")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        # plot weights
        ax = axes[3] if n_contexts == 1 else axes[3, l]
        ax.clear()
        ax.pie(weights[l], labels=[f"{w * 100:.2f}%" for w in weights[l]], colors=colors)
        ax.axis("scaled")
        ax.set_title("Mixture weights")

    # ax = axes[0, -1]
    # # color bar of last target density
    # cbar = plt.colorbar(contour_plot, cax=ax)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    # Create an image from BytesIO object
    image = Image.open(img_buf)
    wandb.log({plot_type: [wandb.Image(image, caption="Plot of target and target distributions")]})
    fig.tight_layout()
    # plt.show()
    plt.close(fig)


def compute_data_for_plot(
        target_dist,
        model,
        contexts,
        normalize_output=False,
        device: str = "cpu",
        min_x: float or None = None,
        max_x: float or None = None,
        min_y: float or None = None,
        max_y: float or None = None,
) -> dict:
    # create meshgrid
    n_plt = 100

    weights, means, scale_trils = model(contexts)
    n_contexts, n_components, dim = means.shape

    mean_flatten = ch.reshape(means, (-1, dim))
    scale_trils_flatten = ch.reshape(scale_trils, (-1, dim, dim))
    if min_x is not None:
        assert max_x is not None
        assert min_y is not None
        assert max_y is not None
    else:
        assert dim == 2, "feature dimension must be 2 for visualization"
        min_x, max_x, min_y, max_y = (
            mean_flatten[:, 0].min(),
            mean_flatten[:, 0].max(),
            mean_flatten[:, 1].min(),
            mean_flatten[:, 1].max(),
        )
        min_x = min_x.detach().cpu().numpy()
        max_x = max_x.detach().cpu().numpy()
        min_y = min_y.detach().cpu().numpy()
        max_y = max_y.detach().cpu().numpy()
        for scale_tril, mean in zip(scale_trils_flatten, mean_flatten):
            ellipse = compute_gaussian_ellipse(
                mean.detach().cpu().numpy(), scale_tril.detach().cpu().numpy()
            )
            min_x = np.min([min_x, ellipse[0, :].min()])
            max_x = np.max([max_x, ellipse[0, :].max()])
            min_y = np.min([min_y, ellipse[1, :].min()])
            max_y = np.max([max_y, ellipse[1, :].max()])

    x = np.linspace(min_x, max_x, n_plt)
    y = np.linspace(min_y, max_y, n_plt)
    xx, yy = np.meshgrid(x, y)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    xy = ch.tensor(xy, dtype=ch.float32).unsqueeze(1).expand(-1, n_contexts, -1).to(device)  # (n_plt**2, n_contexts, 2)

    # evaluate distributions
    with ch.no_grad():
        log_p_tgt = target_dist.log_prob_tgt(contexts, xy) # (n_plt**2, n_contexts)
        log_p_model = model.log_prob_gmm(means, scale_trils, weights, xy)

    log_p_tgt = get_numpy(log_p_tgt.transpose(0, 1)) # (n_contexts, n_plt**2)
    if normalize_output:
        # maximum is now 0, so exp(0) = 1
        log_p_tgt -= log_p_tgt.max()
    log_p_model = get_numpy(log_p_model.transpose(0, 1))
    p_tgt = np.exp(log_p_tgt)
    p_model = np.exp(log_p_model)

    locs = get_numpy(means)
    scale_trils = get_numpy(scale_trils)
    weights = get_numpy(weights.exp())

    return {
        "n_contexts": n_contexts,
        "n_components": n_components,
        "n_plt": n_plt,
        "xx": xx,
        "yy": yy,
        "p_tgt": p_tgt,
        "p_model": p_model,
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