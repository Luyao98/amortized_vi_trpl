import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
"""
the initial version for GMM case plotting, based on Gaussian_plot.py.
currently, it is replaced by the GMM_plot.py
"""


def gaussian_simple_plot(model, target, contexts):
    x1_test = np.linspace(-3, 3, 100)
    x2_test = np.linspace(-3, 3, 100)
    x1, x2 = np.meshgrid(x1_test, x2_test)
    x1_flat = torch.from_numpy(x1.reshape(-1, 1).astype(np.float32)).detach()
    x2_flat = torch.from_numpy(x2.reshape(-1, 1).astype(np.float32)).detach()
    fig, axes = plt.subplots(2, len(contexts), figsize=(20, 10))
    for i, c in enumerate(contexts):
        # plot target distribution
        ax = axes[0, i]
        ax.clear()
        c_expanded = c.unsqueeze(0).expand(x1_flat.shape[0], -1)
        x = torch.cat((x1_flat, x2_flat), dim=1)
        log_probs = target.log_prob_tgt(c_expanded, x.to('cpu')).exp().view(x1.shape).detach() + 1e-6
        ax.contourf(x1, x2, log_probs, levels=100)
        ax.set_title(f'Target density with Context: {c}')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        # plot model distribution
        ax = axes[1, i]
        ax.clear()
        mean, chol = model(c_expanded)
        mean = mean.squeeze(1)
        chol = chol.squeeze(1)
        mean = mean.to('cpu')
        chol = chol.to('cpu')
        log_probs_model = MultivariateNormal(mean, scale_tril=chol).log_prob(x).exp().view(x1.shape).detach() + 1e-6
        ax.contourf(x1, x2, log_probs_model, levels=100)
        ax.set_title(f'Model density with Context: {c}')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    plt.tight_layout()
    plt.show()
