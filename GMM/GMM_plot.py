import numpy as np
import torch
import matplotlib.pyplot as plt
from Gaussian.Gaussian_model import log_prob_gmm

"""
the current version for GMM case plotting, based on the code from Philipp Dahlinger.
"""


def plot2d_matplotlib(
        target_dist,
        model,
        contexts,
        # fig,
        # axes,
        normalize_output=False,
        device: str = "cpu",
        min_x: int or None = None,
        max_x: int or None = None,
        min_y: int or None = None,
        max_y: int or None = None,
):

    model.eval()

    # get data for plotting
    data = compute_data_for_plot2d(
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
    n_tasks = data["n_tasks"]
    n_components = data["n_components"]
    n_plt = data["n_plt"]
    xx = data["xx"]
    yy = data["yy"]
    xy = data["xy"]
    p_tgt = data["p_tgt"]
    p_model = data["p_model"]
    locs = data["locs"]
    scale_trils = data["scale_trils"]
    # weights = data["weights"]
    print("model mean:", locs)
    # plot
    fig, axes = plt.subplots(2, n_tasks, figsize=(15, 10))
    for l in range(n_tasks):
        # plot target distribution
        ax = axes[0, l]
        ax.clear()
        contour_plot = ax.contourf(xx, yy, p_tgt[l].reshape(n_plt, n_plt), levels=100)
        ax.axis("scaled")
        ax.set_title("Target density")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        # plot model distribution
        ax = axes[1, l]
        ax.clear()
        ax.contourf(xx, yy, p_model[l].reshape(n_plt, n_plt), levels=100)
        colors = []
        for k in range(n_components):
            color = next(ax._get_lines.prop_cycler)["color"]
            colors.append(color)
            cur_scale_tril = scale_trils[l, k]
            cur_loc = locs[l, k]
            ax.scatter(x=cur_loc[0], y=cur_loc[1])
            ellipses = compute_gaussian_ellipse(cur_loc, cur_scale_tril)
            ax.plot(ellipses[0, :], ellipses[1, :], color=color)
        ax.axis("scaled")
        ax.set_title("Model density")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        # ax.set_xlim(min_x, max_x)
        # ax.set_ylim(min_y, max_y)

        # # plot weights
        # ax = axes[2, l]
        # ax.clear()
        # ax.pie(weights[l], labels=[f"{w * 100:.2f}%" for w in weights[l], colors=colors)
        # ax.axis("scaled")
        # ax.set_title("Mixture weights")
    # ax = axes[0, -1]
    # # color bar of last target density
    # cbar = plt.colorbar(contour_plot, cax=ax)

    fig.tight_layout()
    plt.show()


def compute_data_for_plot2d(
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

    # extract weights already here, to figure out which components are relevant
    means, scale_trils = model(contexts)
    # determine n_task. i.e. n_contexts
    n_tasks, n_components, _ = means.shape
    weights = (1.0 / n_components) * torch.ones(n_tasks, n_components)  # for uniform weights
    # weights = np.exp(weights.detach().to("cpu").numpy())
    weights = weights.detach().to("cpu").numpy()
    # print("weight here", weights)
    mask = (weights > 0.01).flatten()
    relevant_means = torch.reshape(means, (-1, 2))[mask, :]
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

    # evaluate distributions
    with torch.no_grad():
        log_sum = []
        for c in contexts:
            c_expanded = c.unsqueeze(0).expand(xy.shape[0], -1)
            log_prob = target_dist.log_prob_tgt(c_expanded, xy).exp().view(xx.shape).detach() + 1e-6
            log_sum.append(log_prob)
        log_p_tgt = torch.stack(log_sum, dim=0)
        log_p_model = log_prob_gmm(means, scale_trils, torch.log(torch.from_numpy(weights)), xy)

    log_p_tgt = log_p_tgt.to("cpu").numpy()
    if normalize_output:
        # maximum is now 0, so exp(0) = 1
        log_p_tgt -= log_p_tgt.max()
    log_p_model = log_p_model.to("cpu").numpy()
    p_tgt = np.exp(log_p_tgt)
    p_model = np.exp(log_p_model)
    # extract gmm parameters
    locs = means.detach().to("cpu").numpy()
    scale_trils = scale_trils.detach().to("cpu").numpy()

    return {
        "n_tasks": n_tasks,
        "n_components": n_components,
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


def compute_gaussian_ellipse(mean, scale_tril) -> np.ndarray:
    n_plot = 100
    evals, evecs = np.linalg.eig(scale_tril @ scale_tril.T)
    theta = np.linspace(0, 2 * np.pi, n_plot)
    ellipsis = (np.sqrt(evals[None, :]) * evecs) @ [np.sin(theta), np.cos(theta)]
    ellipsis = ellipsis + mean[:, None]
    return ellipsis