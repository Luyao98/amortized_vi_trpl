from typing import Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
import plotly.subplots as sp
import plotly.express as px
import plotly.graph_objects as go

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
    mask = (weights > 0.01).flatten()
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

        log_p_model, _ = model.log_density(xy, compute_grad=False)
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


def plot1d_plotly(target_dist: LNPDF or Callable,
                  model: GMM,
                  mini_batch_size: int or None = None,
                  use_log_space: bool = False,
                  device: str = "cpu",
                  min_x: int or None = None,
                  max_x: int or None = None, ):
    # get data for plotting
    data = compute_data_for_plot1d(
        target_dist,
        model,
        mini_batch_size,
        device,
        min_x=min_x,
        max_x=max_x,
    )
    n_tasks = data["n_tasks"]
    x = data["x"]
    p_tgt = data["p_tgt"]
    log_p_tgt = data["log_p_tgt"]
    locs = data["locs"]
    stds = data["stds"]
    weights = data["weights"]

    # create plotly figure object
    specs = []
    for l in range(n_tasks):
        specs.append([{"type": "xy"}, {"type": "pie"}])
    fig = sp.make_subplots(
        rows=2,
        cols=n_tasks,
        specs=list(zip(*specs)),
        column_titles=[f"Task {i}" for i in range(n_tasks)],
        row_titles=["Target", "Weights"],
        shared_xaxes=True,
        shared_yaxes=True,
    )
    # build new ones
    for l in range(n_tasks):
        # plot tgt distribution
        plot_traces = {}
        if use_log_space:
            density = log_p_tgt
        else:
            density = p_tgt
        plot_traces["density_plot"] = go.Scatter(
            x=x,
            y=density[:, l],
            mode="lines",
            marker={"color": "black"},
        )
        plot_traces["means"] = []
        colors = px.colors.qualitative.G10
        for k in range(model.n_components):
            cur_std = stds[l, k]
            cur_loc = locs[l, k]
            color = colors[k % len(colors)]
            fig.add_trace({})
            fig.add_vline(x=cur_loc[0], line_width=5, row=1, col=l + 1, line_color=color)
            fig.add_vrect(x0=cur_loc[0] - 2 * cur_std[0, 0], x1=cur_loc[0] + 2 * cur_std[0, 0], line_width=0,
                          fillcolor=color, opacity=0.2, row=1, col=l + 1)

        # plot weights
        plot_traces["pie"] = go.Pie(
            values=weights[l],
            labels=[f"Component {i}" for i, w in enumerate(weights[l])],
            marker=dict(colors=colors),
        )
        # assign to subplots
        fig.add_trace(plot_traces["density_plot"], row=1, col=l + 1)
        fig.add_trace(plot_traces["pie"], row=2, col=l + 1)

    # update layout
    fig.update_traces(showlegend=False)


    return fig


def plot2d_plotly(
        target_dist: LNPDF or Callable,
        model: GMM,
        mini_batch_size: int or None = None,
        use_log_space: bool = False,
        normalize_output: bool = False,
        device: str = "cpu",
        min_x: int or None = None,
        max_x: int or None = None,
        min_y: int or None = None,
        max_y: int or None = None,

):
    """
    params:
        target_dist: Either LNPDF, then the log_density is called. Otherwise it is a function to compute the log density directly
        model: GMM
        mini_batch_size: int | None = None
        use_log_space: bool = False -> if True, the log density is plotted
        normalize_output: bool = False -> if True, the max log output is set to 0 in the visualization.
    """
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
    n_tasks = data["n_tasks"]
    n_plt = data["n_plt"]
    x = data["x"]
    y = data["y"]
    xx = data["xx"]
    yy = data["yy"]
    xy = data["xy"]
    p_tgt = data["p_tgt"]
    log_p_tgt = data["log_p_tgt"]
    locs = data["locs"]
    scale_trils = data["scale_trils"]
    weights = data["weights"]

    # create plotly figure object
    specs = []
    for l in range(n_tasks):
        specs.append([{"type": "xy"}, {"type": "pie"}])
    fig = sp.make_subplots(
        rows=2,
        cols=n_tasks,
        specs=list(zip(*specs)),
        column_titles=[f"Task {i}" for i in range(n_tasks)],
        row_titles=["Target", "Weights"],
        shared_xaxes=True,
        shared_yaxes=True,
    )
    # build new ones
    for l in range(n_tasks):
        # plot tgt distribution
        plot_traces = {}
        if use_log_space:
            density = log_p_tgt
        else:
            density = p_tgt
        plot_traces["contour plot"] = go.Contour(
            x=x,
            y=y,
            z=density[:, l].reshape(n_plt, n_plt),
            ncontours=100,
            contours=dict(showlines=False),
            colorscale="Greys",
        )
        plot_traces["means"] = []
        plot_traces["ellipses"] = []
        colors = px.colors.qualitative.G10[:model.n_components]
        for k in range(model.n_components):
            cur_scale_tril = scale_trils[l, k]
            cur_loc = locs[l, k]
            color = colors[k % len(colors)]
            plot_traces["means"].append(
                go.Scatter(x=[cur_loc[0]], y=[cur_loc[1]], mode="markers", marker=dict(color=color, size=10))
            )
            ellipses = compute_gaussian_ellipse(cur_loc, cur_scale_tril)
            if color is None:
                plot_traces["ellipses"].append(
                    go.Scatter(x=ellipses[0, :], y=ellipses[1, :], mode="lines")
                )
            else:
                plot_traces["ellipses"].append(
                    go.Scatter(
                        x=ellipses[0, :], y=ellipses[1, :], mode="lines", line=dict(color=color, width=5)
                    )
                )
        # plot weights
        plot_traces["pie"] = go.Pie(
            values=weights[l],
            labels=[f"Component {i}" for i, w in enumerate(weights[l])],
            marker=dict(colors=colors),
        )
        # assign to subplots
        fig.add_trace(plot_traces["contour plot"], row=1, col=l + 1)
        for k in range(model.n_components):
            # if torch.exp(model.log_w)[0, k].item() < 0.2:
            #     continue
            fig.add_trace(plot_traces["means"][k], row=1, col=l + 1)
            fig.add_trace(plot_traces["ellipses"][k], row=1, col=l + 1)
        fig.add_trace(plot_traces["pie"], row=2, col=l + 1)
    # update layout
    fig.update_traces(showlegend=False)
    return fig


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
):
    def plot_gaussian_ellipse(ax, mean, scale_tril, color):
        n_plot = 100
        evals, evecs = np.linalg.eig(scale_tril @ scale_tril.T)
        theta = np.linspace(0, 2 * np.pi, n_plot)
        ellipsis = (np.sqrt(evals[None, :]) * evecs) @ [np.sin(theta), np.cos(theta)]
        ellipsis = ellipsis + mean[:, None]
        ax.plot(ellipsis[0, :], ellipsis[1, :], color=color)

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
    n_tasks = data["n_tasks"]
    n_plt = data["n_plt"]
    x = data["x"]
    y = data["y"]
    xx = data["xx"]
    yy = data["yy"]
    xy = data["xy"]
    p_tgt = data["p_tgt"]
    locs = data["locs"]
    scale_trils = data["scale_trils"]
    weights = data["weights"]
    # plot
    for l in range(n_tasks):
        # plot tgt distribution
        ax = axes[0, l]
        ax.clear()
        contour_plot = ax.contourf(xx, yy, p_tgt[:, l].reshape(n_plt, n_plt), levels=100)
        ax.axis("scaled")
        ax.set_title("Target density")
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")

        # plot model distribution
        ax = axes[1, l]
        ax.clear()
        ax.contourf(xx, yy, p_tgt[:, l].reshape(n_plt, n_plt), levels=100)
        colors = []
        tab_color_map = plt.get_cmap('tab10')
        for k in range(model.n_components):
            color = tab_color_map.colors[k]
            colors.append(color)
            cur_scale_tril = scale_trils[l, k]
            cur_loc = locs[l, k]
            ax.scatter(x=cur_loc[0], y=cur_loc[1])
            ellipses = compute_gaussian_ellipse(cur_loc, cur_scale_tril)
            ax.plot(ellipses[0, :], ellipses[1, :], color=color)
        ax.axis("scaled")
        ax.set_title("Model density")
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
        # ax.set_xlim(min_x, max_x)
        # ax.set_ylim(min_y, max_y)

        # plot weights
        ax = axes[2, l]
        ax.clear()
        ax.pie(weights[l], labels=[f"{w * 100:.2f}%" for w in weights[l]], colors=colors)
        ax.axis("scaled")
        ax.set_title("Mixture weights")
    ax = axes[0, -1]
    # color bar of last tgt density
    cbar = plt.colorbar(contour_plot, cax=ax)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    image = Image.open(img_buf)
    wandb.log({"Evaluation": [wandb.Image(image, caption="Plot of tgt and tgt distributions")]})
    fig.tight_layout()
    plt.close(fig)