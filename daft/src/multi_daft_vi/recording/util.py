from typing import Callable

import numpy as np
import torch
from matplotlib import pyplot as plt

from daft.src.gmm_util.gmm import GMM
from daft.src.multi_daft_vi.lnpdf import LNPDF
from daft.src.multi_daft_vi.util_lnpdf import mini_batch_function_no_grad

import wandb
import io
from PIL import Image


def compute_data_for_plot2d(
        target_dist: LNPDF or Callable,
        model: GMM,
        mini_batch_size: int or None = None,
        normalize_output=False,
        device: str = "cpu",
        min_x: float or None = None,
        max_x: float or None = None,
        min_y: float or None = None,
        max_y: float or None = None,
) -> dict:
    """
    params:
        target_dist: Either LNPDF, then the log_density is called. Otherwise it is a function to compute the log density directly
    """
    # create meshgrid
    n_plt = 100

    # extract weights already here, to figure out which components are relevant
    weights = np.exp(model.log_w.detach().to("cpu").numpy())
    mask = (weights > 0.001).flatten()
    relevant_means = torch.reshape(model.mean, (-1, 2))[mask, :]
    scale_trils = model.cov_chol
    relevant_scale_trils = torch.reshape(scale_trils, (-1, 2, 2))[mask, :, :]
    if min_x is not None:
        assert max_x is not None
        assert min_y is not None
        assert max_y is not None
    else:
        min_x, max_x, min_y, max_y = (
            relevant_means[:, 0].min(),
            relevant_means[:, 0].max(),
            relevant_means[:, 1].min(),
            relevant_means[:, 1].max(),
        )
        min_x = min_x.detach().cpu().numpy()
        max_x = max_x.detach().cpu().numpy()
        min_y = min_y.detach().cpu().numpy()
        max_y = max_y.detach().cpu().numpy()
        for scale_tril, mean in zip(relevant_scale_trils, relevant_means):
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
    xy = torch.tensor(xy, dtype=torch.float32).to(device)

    # determine n_task
    n_tasks = model.mean.shape[0]
    xy = torch.broadcast_to(xy[:, None, :], (n_plt ** 2, n_tasks, 2))

    # evaluate distributions
    with torch.no_grad():
        try:
            if mini_batch_size is None:
                log_p_tgt, _ = target_dist.log_density(xy, compute_grad=False)
            else:
                log_p_tgt, _ = target_dist.mini_batch_log_density(
                    xy, mini_batch_size=mini_batch_size, compute_grad=False
                )
        except AttributeError:
            # callable
            if mini_batch_size is None:
                log_p_tgt = target_dist(xy)
            else:
                fun = lambda z: (target_dist(z), None)
                log_p_tgt, _ = mini_batch_function_no_grad(fun, xy, mini_batch_size=mini_batch_size)

        log_p_model, _ = model.log_density(xy.to(model.mean.device), compute_grad=False)
    log_p_tgt = log_p_tgt.to("cpu").numpy()
    if normalize_output:
        # maximum is now 0, so exp(0) = 1
        log_p_tgt -= log_p_tgt.max()
    log_p_model = log_p_model.to("cpu").numpy()
    p_tgt = np.exp(log_p_tgt)
    p_model = np.exp(log_p_model)
    # extract gmm parameters
    locs = model.mean.detach().to("cpu").numpy()
    scale_trils = model.cov_chol.detach().to("cpu").numpy()

    return {
        "n_tasks": n_tasks,
        "n_plt": n_plt,
        "x": x,
        "y": y,
        "xx": xx,
        "yy": yy,
        "xy": xy,
        "p_tgt": p_tgt,
        "log_p_tgt": log_p_tgt,
        "p_model": p_model,
        "log_p_model": log_p_model,
        "locs": locs,
        "scale_trils": scale_trils,
        "weights": weights,
    }


def compute_data_for_plot1d(target_dist,
                            model,
                            mini_batch_size,
                            device,
                            min_x: float or None = None,
                            max_x: float or None = None, ):
    # create linspace
    n_plt = 100
    # extract weights already here, to figure out which components are relevant
    weights = np.exp(model.log_w.detach().to("cpu").numpy())
    mask = (weights > 0.01).flatten()
    relevant_means = torch.reshape(model.mean, (-1, 1))[mask, :]
    relevant_stds = torch.reshape(torch.sqrt(model.cov), (-1, 1, 1))[mask, :, :]
    if min_x is not None:
        assert max_x is not None
    else:
        min_x, max_x = (
            relevant_means[:, 0].min(),
            relevant_means[:, 0].max(),
        )
        min_x -= relevant_stds[:, 0, 0].max()
        max_x += relevant_stds[:, 0, 0].max()
        min_x = min_x.detach().cpu().numpy()
        max_x = max_x.detach().cpu().numpy()

    x = torch.linspace(min_x, max_x, n_plt).to(device)
    # determine n_task
    n_tasks = model.mean.shape[0]
    x = torch.broadcast_to(x[:, None, None], (n_plt, n_tasks, 1))

    # evaluate distributions
    with torch.no_grad():
        try:
            if mini_batch_size is None:
                log_p_tgt, _ = target_dist.log_density(x, compute_grad=False)
            else:
                log_p_tgt, _ = target_dist.mini_batch_log_density(
                    x, mini_batch_size=mini_batch_size, compute_grad=False
                )
        except AttributeError:
            # callable
            if mini_batch_size is None:
                log_p_tgt = target_dist(x)
            else:
                fun = lambda z: (target_dist(z), None)
                log_p_tgt, _ = mini_batch_function_no_grad(fun, x, mini_batch_size=mini_batch_size)

        log_p_model, _ = model.log_density(x, compute_grad=False)
    log_p_tgt = log_p_tgt.to("cpu").numpy()
    log_p_model = log_p_model.to("cpu").numpy()
    p_tgt = np.exp(log_p_tgt)
    p_model = np.exp(log_p_model)
    x = x.to("cpu").numpy()
    # extract gmm parameters
    locs = model.mean.detach().to("cpu").numpy()
    stds = np.sqrt(model.cov.detach().to("cpu").numpy())

    return {
        "n_tasks": n_tasks,
        "n_plt": n_plt,
        "x": x[:, 0, 0],
        "p_tgt": p_tgt,
        "log_p_tgt": log_p_tgt,
        "p_model": p_model,
        "log_p_model": log_p_model,
        "locs": locs,
        "stds": stds,
        "weights": weights,
    }


def compute_gaussian_ellipse(mean, scale_tril) -> np.ndarray:
    n_plot = 100
    evals, evecs = np.linalg.eig(scale_tril @ scale_tril.T)
    theta = np.linspace(0, 2 * np.pi, n_plot)
    ellipsis = (np.sqrt(evals[None, :]) * evecs) @ [np.sin(theta), np.cos(theta)]
    ellipsis = ellipsis + mean[:, None]
    return ellipsis


def plot2d_matplotlib(
        target_dist: LNPDF or Callable,
        model: GMM,
        fig,
        axes,
        mini_batch_size: int or None = None,
        normalize_output=False,
        device: str = "cpu",
        min_x: int or None = None,
        max_x: int or None = None,
        min_y: int or None = None,
        max_y: int or None = None,
        logging = False
):

    assert model.d_z == 2, "Only 2D models are supported"

    # get data for plotting
    data = compute_data_for_plot2d(
        target_dist,
        model,
        mini_batch_size,
        normalize_output=normalize_output,
        device=device,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
    )
    n_tasks = 1 # only plot fisrt 3 tasks
    n_plt = data["n_plt"]
    xx = data["xx"]
    yy = data["yy"]
    p_tgt = data["p_tgt"]
    p_model = data["p_model"]
    locs = data["locs"]
    scale_trils = data["scale_trils"]
    weights = data["weights"]
    # plot
    for l in range(n_tasks):
        # plot tgt distribution
        ax = axes[0]
        ax.clear()
        contour_plot = ax.contourf(xx, yy, p_tgt[:, l].reshape(n_plt, n_plt), levels=100)
        ax.axis("scaled")
        ax.set_title("Target Density")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        # plot model distribution
        ax = axes[1]
        ax.clear()
        ax.contourf(xx, yy, p_tgt[:, l].reshape(n_plt, n_plt), levels=100)
        colors = []
        for k in range(model.n_components):
            color = next(ax._get_lines.prop_cycler)["color"]
            colors.append(color)
            cur_scale_tril = scale_trils[l, k]
            cur_loc = locs[l, k]
            ax.scatter(x=cur_loc[0], y=cur_loc[1])
            ellipses = compute_gaussian_ellipse(cur_loc, cur_scale_tril)
            ax.plot(ellipses[0, :], ellipses[1, :], color=color)
        ax.axis("scaled")
        ax.set_title("Comparision")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        # ax.set_xlim(min_x, max_x)
        # ax.set_ylim(min_y, max_y)

        # plot model distribution
        ax = axes[2]
        ax.clear()
        ax.contourf(xx, yy, p_model[:, l].reshape(n_plt, n_plt), levels=100)
        ax.axis("scaled")
        ax.set_title("Model Density")
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")

        # plot weights
        ax = axes[3]
        ax.clear()
        ax.pie(weights[l], labels=[f"{w * 100:.2f}%" for w in weights[l]], colors=colors)
        ax.axis("scaled")
        ax.set_title("Mixture weights")
    # ax = axes[0, -1]
    # color bar of last tgt density
    # cbar = plt.colorbar(contour_plot, cax=ax)

    if logging:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        image = Image.open(img_buf)
        wandb.log({"Evaluation": [wandb.Image(image, caption="Plot of target and target distributions")]})
        fig.tight_layout()
        plt.close(fig)
    else:
        fig.tight_layout()
