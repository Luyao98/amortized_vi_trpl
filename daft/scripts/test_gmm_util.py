import math

import einops
import torch
from einops import rearrange

from daft.src.gmm_util.util import (
    prec_to_prec_chol,
    prec_to_cov_chol,
    cov_chol_to_cov,
    sample_gmm,
    gmm_log_density,
    gmm_log_component_densities,
    gmm_log_responsibilities,
    gmm_log_density_grad,
    # gmm_log_responsibilities,
    # gmm_log_density_grad_hess,
)

from daft.src.gmm_util.gmm import GMM
from daft.src.gmm_util.util_autograd import eval_fn_grad


def test_prec_to_prec_chol():
    # low D
    L_true = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 1.0]],
            [[3.0, 0.0], [-7.0, 3.5]],
            [[math.pi, 0.0], [math.e, 124]],
        ],
    )
    prec = L_true @ rearrange(L_true, "... d_z d_z2 -> ... d_z2 d_z")
    L_comp = prec_to_prec_chol(prec=prec)
    assert torch.allclose(L_comp, L_true)

    # low D, additional batch-dim
    L_true = torch.tensor(
        [
            [
                [[1.0, 0.0], [2.0, 1.0]],
                [[3.0, 0.0], [-7.0, 3.5]],
                [[math.pi, 0.0], [math.e, 124]],
            ],
            [
                [[3.0, 0.0], [-70.0, 3.5]],
                [[math.e, 0.0], [math.pi, 124]],
                [[1.0, 0.0], [2.0, 1.0]],
            ],
        ]
    )
    prec = L_true @ rearrange(L_true, "... d_z d_z2 -> ... d_z2 d_z")
    L_comp = prec_to_prec_chol(prec=prec)
    assert torch.allclose(L_comp, L_true)


def test_prec_to_cov_chol():
    # low D
    L = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 1.0]],
            [[3.0, 0.0], [-7.0, 3.5]],
            [[math.pi, 0.0], [math.e, 124]],
        ]
    )
    cov = L @ rearrange(L, "... dz dz2 -> ... dz2 dz")
    prec = torch.linalg.inv(cov)
    scale_tril = prec_to_cov_chol(prec=prec)
    assert torch.allclose(scale_tril, L)

    # low D, additional batch-dim
    L_true = torch.tensor(
        [
            [
                [[1.0, 0.0], [2.0, 1.0]],
                [[3.0, 0.0], [-7.0, 3.5]],
                [[math.pi, 0.0], [math.e, 124]],
            ],
            [
                [[3.0, 0.0], [-70.0, 3.5]],
                [[math.e, 0.0], [math.pi, 124]],
                [[1.0, 0.0], [2.0, 1.0]],
            ],
        ]
    )
    prec = L_true @ rearrange(L_true, "... dz dz2 -> ... dz2 dz")
    L_comp = prec_to_prec_chol(prec=prec)
    assert torch.allclose(L_comp, L_true)


def test_cov_chol_to_cov():
    # check 1
    scale_tril = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
        ]
    )
    true_cov = torch.tensor(
        [
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
        ]
    )
    cov = cov_chol_to_cov(cov_chol=scale_tril)
    assert torch.allclose(cov, true_cov)

    # check 2 with additional batch dim
    scale_tril = torch.tensor(
        [
            [
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            ],
            [
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            ],
        ]
    )
    true_cov = torch.tensor(
        [
            [
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            ],
            [
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            ],
        ]
    )
    cov = cov_chol_to_cov(cov_chol=scale_tril)
    assert torch.allclose(cov, true_cov)


def test_sample_gmm():
    # check 0: d_z == 1, n_components == 1
    n_samples = 10
    d_z = 1
    log_w = torch.log(torch.tensor([1.0]))
    mean = torch.tensor([[-1.0]])
    cov_chol = torch.tensor([[[0.1]]])
    samples = sample_gmm(n_samples=10, log_w=log_w, mean=mean, cov_chol=cov_chol)
    assert samples.shape == (n_samples, d_z)

    # check 1: d_z == 1, n_components == 2
    n_samples = 10
    d_z = 1
    log_w = torch.log(torch.tensor([0.8, 0.2]))
    mean = torch.tensor([[-1.0], [1.0]])
    cov_chol = torch.tensor([[[0.2]], [[0.2]]])
    samples = sample_gmm(n_samples=10, log_w=log_w, mean=mean, cov_chol=cov_chol)
    assert samples.shape == (n_samples, d_z)

    # check 2: d_z == 2, n_components == 1
    n_samples = 10
    d_z = 2
    log_w = torch.log(torch.tensor([1.0]))
    mean = torch.tensor([[-1.0, 1.0]])
    cov_chol = torch.tensor([[[0.1, 0.0], [-0.2, 1.0]]])
    samples = sample_gmm(n_samples=10, log_w=log_w, mean=mean, cov_chol=cov_chol)
    assert samples.shape == (n_samples, d_z)

    # check 3: d_z == 2, n_components == 2
    n_samples = 10
    d_z = 2
    n_batch = 3
    log_w = torch.log(torch.tensor([0.8, 0.2]))
    mean = torch.tensor(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    cov_chol = torch.tensor(
        [
            [[0.1, 0.0], [-0.2, 1.0]],
            [[0.2, 0.0], [0.2, 2.0]],
        ],
    )
    samples = sample_gmm(n_samples=10, log_w=log_w, mean=mean, cov_chol=cov_chol)
    assert samples.shape == (n_samples, d_z)

    # # check 4: d_z == 2, n_components == 2, batch_dim
    # d_z = 2
    # n_samples = 10
    # n_batch = 3
    # log_w = torch.log(
    #     [
    #         [0.8, 0.2],
    #         [0.3, 0.7],
    #         [0.1, 0.9],
    #     ]
    # )
    # loc = torch.tensor(
    #     [
    #         [
    #             [1.0, 1.0],
    #             [-1.0, 1.0],
    #         ],
    #         [
    #             [2.0, 2.0],
    #             [-1.0, 3.0],
    #         ],
    #         [
    #             [2.0, 2.0],
    #             [-1.0, 3.0],
    #         ],
    #     ]
    # )
    # scale_tril = torch.tensor(
    #     [
    #         [
    #             [[0.1, 0.0], [-0.2, 1.0]],
    #             [[0.2, 0.0], [0.2, 2.0]],
    #         ],
    #         [
    #             [[0.2, 0.0], [-0.3, 2.0]],
    #             [[0.1, 0.0], [0.5, 3.0]],
    #         ],
    #         [
    #             [[0.2, 0.0], [-0.3, 2.0]],
    #             [[0.1, 0.0], [0.5, 3.0]],
    #         ],
    #     ]
    # )
    # samples = sample_gmm(n_samples=10, log_w=log_w, loc=loc, scale_tril=scale_tril)
    # assert samples.shape == (n_samples, n_batch, d_z)


def test_gmm_log_density():
    # check 0: d_z == 1, n_components == 1
    n_samples = 10
    torch.manual_seed(42)
    z = torch.randn((n_samples, 1))
    log_w = torch.log(torch.tensor([1.0]))
    loc = torch.tensor([[1.0]])
    cov_chol = torch.tensor([[[0.1]]])
    prec = torch.linalg.inv(
        einops.einsum(cov_chol, cov_chol, "... dz dz2, ... dz3 dz2 -> ... dz dz3")
    )
    assert torch.allclose(prec_to_cov_chol(prec), cov_chol)
    prec_chol = prec_to_prec_chol(prec)
    log_densities = gmm_log_density(
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(z=z, mean=loc, prec_chol=prec_chol),
    )
    true_log_densities = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        component_distribution=torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=cov_chol, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples,)
    assert log_densities.shape == (n_samples,)
    assert torch.allclose(log_densities, true_log_densities)

    # check 1: d_z == 1, n_components == 2
    n_samples = 10
    z = torch.randn((n_samples, 1))
    log_w = torch.log(torch.tensor([0.8, 0.2]))
    loc = torch.tensor(
        [
            [1.0],
            [-1.0],
        ]
    )
    cov_chol = torch.tensor(
        [
            [[0.1]],
            [[0.2]],
        ]
    )
    prec = torch.linalg.inv(
        einops.einsum(cov_chol, cov_chol, "... dz dz2, ... dz3 dz2 -> ... dz dz3")
    )
    assert torch.allclose(prec_to_cov_chol(prec), cov_chol)
    prec_chol = prec_to_prec_chol(prec)
    log_densities = gmm_log_density(
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(z=z, mean=loc, prec_chol=prec_chol),
    )
    true_log_densities = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(
            logits=log_w,
            validate_args=True,
        ),
        component_distribution=torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=cov_chol, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples,)
    assert log_densities.shape == (n_samples,)
    assert torch.allclose(log_densities, true_log_densities)

    # check 2: d_z == 2, n_components == 1
    n_samples = 10
    z = torch.randn((n_samples, 2))
    log_w = torch.log(torch.tensor([1.0]))
    loc = torch.tensor([[1.0, 1.0]])
    cov_chol = torch.tensor([[[0.1, 0.0], [-0.2, 1.0]]])
    prec = torch.linalg.inv(
        einops.einsum(cov_chol, cov_chol, "... dz dz2, ... dz3 dz2 -> ... dz dz3")
    )
    assert torch.allclose(prec_to_cov_chol(prec), cov_chol)
    prec_chol = prec_to_prec_chol(prec)
    log_densities = gmm_log_density(
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(z=z, mean=loc, prec_chol=prec_chol),
    )
    true_log_densities = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        component_distribution=torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=cov_chol, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples,)
    assert log_densities.shape == (n_samples,)
    assert torch.allclose(log_densities, true_log_densities)

    # check 3: d_z == 2, n_components == 2
    n_samples = 10
    z = torch.randn((n_samples, 2))
    log_w = torch.log(torch.tensor([0.8, 0.2]))
    loc = torch.tensor(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    cov_chol = torch.tensor(
        [
            [[0.1, 0.0], [-0.2, 1.0]],
            [[0.2, 0.0], [0.2, 2.0]],
        ]
    )
    prec = torch.linalg.inv(
        einops.einsum(cov_chol, cov_chol, "... dz dz2, ... dz3 dz2 -> ... dz dz3")
    )
    assert torch.allclose(prec_to_cov_chol(prec), cov_chol)
    prec_chol = prec_to_prec_chol(prec)
    log_densities = gmm_log_density(
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(z=z, mean=loc, prec_chol=prec_chol),
    )
    true_log_densities = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(
            logits=log_w,
            validate_args=True,
        ),
        component_distribution=torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=cov_chol, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples,)
    assert log_densities.shape == (n_samples,)
    assert torch.allclose(log_densities, true_log_densities)

    # check 4: d_z == 2, n_components == 2, batch_dim
    n_samples = 10
    n_batch = 3
    z = torch.randn((n_samples, n_batch, 2))
    log_w = torch.log(
        torch.tensor(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.1, 0.9],
            ]
        )
    )
    loc = torch.tensor(
        [
            [
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
        ]
    )
    cov_chol = torch.tensor(
        [
            [
                [[0.1, 0.0], [-0.2, 1.0]],
                [[0.2, 0.0], [0.2, 2.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
        ]
    )
    prec = torch.linalg.inv(
        einops.einsum(cov_chol, cov_chol, "... dz dz2, ... dz3 dz2 -> ... dz dz3")
    )
    assert torch.allclose(prec_to_cov_chol(prec), cov_chol)
    prec_chol = prec_to_prec_chol(prec)
    log_densities = gmm_log_density(
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(z=z, mean=loc, prec_chol=prec_chol),
    )
    true_log_densities = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(
            logits=log_w,
            validate_args=True,
        ),
        component_distribution=torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=cov_chol, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples, n_batch)
    assert log_densities.shape == (n_samples, n_batch)
    assert torch.allclose(log_densities, true_log_densities)


def test_gmm_log_component_densities():
    # check 1: d_z == 1, n_components == 1
    n_samples = 10
    n_components = 1
    torch.manual_seed(42)
    z = torch.randn((n_samples, 1))
    mean = torch.tensor([[1.0]])
    cov_chol = torch.tensor([[[0.1]]])
    prec_chol = torch.linalg.inv(cov_chol)
    log_component_densities = gmm_log_component_densities(
        z=z,
        mean=mean,
        prec_chol=prec_chol,
    )
    true_log_component_densities = torch.distributions.MultivariateNormal(
        loc=mean, scale_tril=cov_chol, validate_args=True
    ).log_prob(z[:, None, :])
    assert true_log_component_densities.shape == (n_samples, n_components)
    assert log_component_densities.shape == (n_samples, n_components)
    assert torch.allclose(true_log_component_densities, log_component_densities)

    # check 2: d_z == 2, n_components == 1
    n_samples = 10
    n_components = 1
    z = torch.randn((n_samples, 2))
    mean = torch.tensor([[1.0, -1.0]])
    cov_chol = torch.tensor([[[0.1, 0.0], [-2.0, 1.0]]])
    prec = torch.linalg.inv(
        einops.einsum(cov_chol, cov_chol, "... dz dz2, ... dz3 dz2 -> ... dz dz3")
    )
    assert torch.allclose(prec_to_cov_chol(prec), cov_chol)
    prec_chol = prec_to_prec_chol(prec)
    log_component_densities = gmm_log_component_densities(
        z=z,
        mean=mean,
        prec_chol=prec_chol,
    )
    true_log_component_densities = torch.distributions.MultivariateNormal(
        loc=mean, scale_tril=cov_chol, validate_args=True
    ).log_prob(z[:, None, :])
    assert true_log_component_densities.shape == (n_samples, n_components)
    assert log_component_densities.shape == (n_samples, n_components)
    assert torch.allclose(true_log_component_densities, log_component_densities)

    # check 3: d_z == 2, n_components == 2
    n_samples = 10
    n_components = 2
    z = torch.randn((n_samples, 2))
    mean = torch.tensor([[1.0, -1.0], [1.0, 1.0]])
    cov_chol = torch.tensor([[[0.1, 0.0], [-2.0, 1.0]], [[0.2, 0.0], [-2.0, 1.0]]])
    prec = torch.linalg.inv(
        einops.einsum(cov_chol, cov_chol, "... dz dz2, ... dz3 dz2 -> ... dz dz3")
    )
    assert torch.allclose(prec_to_cov_chol(prec), cov_chol)
    prec_chol = prec_to_prec_chol(prec)
    log_component_densities = gmm_log_component_densities(
        z=z,
        mean=mean,
        prec_chol=prec_chol,
    )
    true_log_component_densities = torch.distributions.MultivariateNormal(
        loc=mean, scale_tril=cov_chol, validate_args=True
    ).log_prob(z[:, None, :])
    assert true_log_component_densities.shape == (n_samples, n_components)
    assert log_component_densities.shape == (n_samples, n_components)
    assert torch.allclose(true_log_component_densities, log_component_densities)

    # check 4: d_z == 2, n_components == 2, batch_dim
    n_samples = 10
    n_components = 2
    n_batch = 2
    z = torch.randn((n_samples, 2, 2))
    mean = torch.tensor(
        [
            [[1.0, -1.0], [1.0, 1.0]],
            [[2.0, -2.0], [2.0, 2.0]],
        ]
    )
    cov_chol = torch.tensor(
        [
            [[[0.1, 0.0], [-2.0, 1.0]], [[0.2, 0.0], [-2.0, 1.0]]],
            [[[0.3, 0.0], [-3.0, 2.0]], [[0.4, 0.0], [-1.0, 2.0]]],
        ]
    )
    prec = torch.linalg.inv(
        einops.einsum(cov_chol, cov_chol, "... dz dz2, ... dz3 dz2 -> ... dz dz3")
    )
    assert torch.allclose(prec_to_cov_chol(prec), cov_chol)
    prec_chol = prec_to_prec_chol(prec)
    log_component_densities = gmm_log_component_densities(
        z=z,
        mean=mean,
        prec_chol=prec_chol,
    )
    true_log_component_densities = torch.distributions.MultivariateNormal(
        loc=mean, scale_tril=cov_chol, validate_args=True
    ).log_prob(z[:, :, None, :])
    assert true_log_component_densities.shape == (n_samples, n_batch, n_components)
    assert log_component_densities.shape == (n_samples, n_batch, n_components)
    assert torch.allclose(true_log_component_densities, log_component_densities)


def test_gmm_log_density_grad_hess():
    # check 0: d_z == 1, n_components == 1
    n_samples = 10
    d_z = 1
    torch.manual_seed(42)
    z = torch.randn((n_samples, d_z))
    log_w = torch.log(torch.tensor([1.0]))
    loc = torch.tensor([[1.0]])
    cov_chol = torch.tensor([[[0.1]]])
    prec = torch.linalg.inv(
        einops.einsum(cov_chol, cov_chol, "... dz dz2, ... dz3 dz2 -> ... dz dz3")
    )
    assert torch.allclose(prec_to_cov_chol(prec), cov_chol)
    prec_chol = prec_to_prec_chol(prec)
    log_density, log_density_grad = gmm_log_density_grad(
        z=z,
        log_w=log_w,
        mean=loc,
        prec=prec,
        prec_chol=prec_chol,
        compute_grad=True,
    )
    gmm = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        component_distribution=torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=cov_chol, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad = eval_fn_grad(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
    )
    assert log_density.shape == (n_samples,)
    assert log_density_grad.shape == (n_samples, d_z)
    assert true_log_density.shape == (n_samples,)
    assert true_log_density_grad.shape == (n_samples, d_z)
    assert torch.allclose(true_log_density, log_density)
    assert torch.allclose(true_log_density_grad, log_density_grad)

    # check 1: d_z == 1, n_components == 2
    n_samples = 10
    d_z = 1
    z = torch.randn((n_samples, d_z))
    log_w = torch.log(torch.tensor([0.8, 0.2]))
    loc = torch.tensor(
        [
            [1.0],
            [-1.0],
        ]
    )
    cov_chol = torch.tensor(
        [
            [[0.1]],
            [[0.2]],
        ]
    )
    prec = torch.linalg.inv(
        einops.einsum(cov_chol, cov_chol, "... dz dz2, ... dz3 dz2 -> ... dz dz3")
    )
    assert torch.allclose(prec_to_cov_chol(prec), cov_chol)
    prec_chol = prec_to_prec_chol(prec)
    log_density, log_density_grad = gmm_log_density_grad(
        z=z,
        log_w=log_w,
        mean=loc,
        prec=prec,
        prec_chol=prec_chol,
        compute_grad=True,
    )
    gmm = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        component_distribution=torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=cov_chol, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad = eval_fn_grad(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
    )
    assert log_density.shape == (n_samples,)
    assert log_density_grad.shape == (n_samples, d_z)
    assert true_log_density.shape == (n_samples,)
    assert true_log_density_grad.shape == (n_samples, d_z)
    assert torch.allclose(true_log_density, log_density)
    assert torch.allclose(true_log_density_grad, log_density_grad)

    # check 2: d_z == 2, n_components == 1
    n_samples = 10
    d_z = 2
    z = torch.randn((n_samples, d_z))
    log_w = torch.log(torch.tensor([1.0]))
    loc = torch.tensor([[1.0, 1.0]])
    cov_chol = torch.tensor([[[0.1, 0.0], [-0.2, 1.0]]])
    prec = torch.linalg.inv(
        einops.einsum(cov_chol, cov_chol, "... dz dz2, ... dz3 dz2 -> ... dz dz3")
    )
    assert torch.allclose(prec_to_cov_chol(prec), cov_chol)
    prec_chol = prec_to_prec_chol(prec)
    log_density, log_density_grad = gmm_log_density_grad(
        z=z,
        log_w=log_w,
        mean=loc,
        prec=prec,
        prec_chol=prec_chol,
        compute_grad=True,
    )
    gmm = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        component_distribution=torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=cov_chol, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad = eval_fn_grad(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
    )
    assert log_density.shape == (n_samples,)
    assert log_density_grad.shape == (n_samples, d_z)
    assert true_log_density.shape == (n_samples,)
    assert true_log_density_grad.shape == (n_samples, d_z)
    assert torch.allclose(true_log_density, log_density)
    assert torch.allclose(true_log_density_grad, log_density_grad)

    # check 3: d_z == 2, n_components == 2
    n_samples = 10
    d_z = 2
    n_components = 2
    z = torch.randn((n_samples, d_z))
    log_w = torch.log(torch.tensor([0.8, 0.2]))
    loc = torch.tensor(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    cov_chol = torch.tensor(
        [
            [[0.1, 0.0], [-0.2, 1.0]],
            [[0.2, 0.0], [0.2, 2.0]],
        ]
    )
    prec = torch.linalg.inv(
        einops.einsum(cov_chol, cov_chol, "... dz dz2, ... dz3 dz2 -> ... dz dz3")
    )
    assert torch.allclose(prec_to_cov_chol(prec), cov_chol)
    prec_chol = prec_to_prec_chol(prec)
    log_density, log_density_grad = gmm_log_density_grad(
        z=z,
        log_w=log_w,
        mean=loc,
        prec=torch.linalg.inv(cov_chol_to_cov(cov_chol)),
        prec_chol=prec_chol,
        compute_grad=True,
    )
    gmm = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        component_distribution=torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=cov_chol, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad = eval_fn_grad(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
    )
    assert log_density.shape == (n_samples,)
    assert log_density_grad.shape == (n_samples, d_z)
    assert true_log_density.shape == (n_samples,)
    assert true_log_density_grad.shape == (n_samples, d_z)
    assert torch.allclose(true_log_density, log_density)
    assert torch.allclose(true_log_density_grad, log_density_grad)

    # check 4: d_z == 2, n_components == 2, batch_dim
    n_samples = 10
    n_components = 2
    d_z = 2
    n_batch = 3
    z = torch.randn((n_samples, n_batch, d_z))
    log_w = torch.log(
        torch.tensor(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.1, 0.9],
            ]
        )
    )
    loc = torch.tensor(
        [
            [
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
        ]
    )
    cov_chol = torch.tensor(
        [
            [
                [[0.1, 0.0], [-0.2, 1.0]],
                [[0.2, 0.0], [0.2, 2.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
        ]
    )
    prec = torch.linalg.inv(
        einops.einsum(cov_chol, cov_chol, "... dz dz2, ... dz3 dz2 -> ... dz dz3")
    )
    assert torch.allclose(prec_to_cov_chol(prec), cov_chol)
    prec_chol = prec_to_prec_chol(prec)
    log_density, log_density_grad = gmm_log_density_grad(
        z=z,
        log_w=log_w,
        mean=loc,
        prec=prec,
        prec_chol=prec_chol,
        compute_grad=True,
    )
    gmm = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        component_distribution=torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=cov_chol, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad = eval_fn_grad(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
    )
    assert log_density.shape == (n_samples, n_batch)
    assert log_density_grad.shape == (n_samples, n_batch, d_z)
    assert true_log_density.shape == (n_samples, n_batch)
    assert true_log_density_grad.shape == (n_samples, n_batch, d_z)
    assert torch.allclose(true_log_density, log_density)
    assert torch.allclose(true_log_density_grad, log_density_grad)


def test_gmm():
    ## (1) n_batch_dims = 0
    # generate valid parameters
    # set 1
    log_w = torch.log(torch.tensor([0.1, 0.3, 0.6]))
    loc = torch.tensor(
        [
            [3.0, 4.0],
            [-1.0, 2.0],
            [-7.0, -1.0],
        ]
    )
    cov_chol = torch.tensor(
        [
            [
                [1.0, 0.0],
                [-0.5, 0.1],
            ],
            [
                [7.0, 0.0],
                [1.0, 7.0],
            ],
            [
                [0.2, 0.0],
                [-9.0, 9.0],
            ],
        ]
    )
    cov = cov_chol_to_cov(cov_chol)
    prec = torch.linalg.inv(cov)
    prec_chol = prec_to_prec_chol(prec)
    # set 2
    log_w2 = torch.log(torch.tensor([0.5, 0.1, 0.4]))
    loc2 = torch.tensor(
        [
            [1.0, 4.0],
            [1.0, 2.0],
            [-5.0, -1.0],
        ]
    )
    cov_chol2 = torch.tensor(
        [
            [
                [2.0, 0.0],
                [-0.5, 1.1],
            ],
            [
                [1.0, 0.0],
                [2.0, 9.0],
            ],
            [
                [0.3, 0.0],
                [-9.0, 9.0],
            ],
        ]
    )
    cov2 = cov_chol_to_cov(cov_chol2)
    prec2 = torch.linalg.inv(cov2)
    prec_chol2 = prec_to_prec_chol(prec2)
    assert torch.allclose(torch.linalg.cholesky(cov2), cov_chol2)
    assert torch.allclose(torch.linalg.inv(cov2), prec2)
    assert torch.allclose(torch.linalg.cholesky(prec2), prec_chol2)
    # (i) Initialize with precision
    gmm = GMM(log_w=log_w, mean=loc, prec=prec)
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert torch.allclose(gmm.log_w, log_w)
    assert torch.allclose(gmm.prec, prec)
    assert torch.allclose(gmm.cov, cov)
    assert torch.allclose(gmm.prec_chol, prec_chol)
    assert torch.allclose(gmm.cov_chol, cov_chol)
    # (ii) Set new parameters (prec)
    gmm.log_w = log_w2
    gmm.loc = loc2
    gmm.prec = prec2
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert torch.allclose(gmm.log_w, log_w2)
    assert torch.allclose(gmm.prec, prec2)
    assert torch.allclose(gmm.cov, cov2)
    assert torch.allclose(gmm.prec_chol, prec_chol2)
    assert torch.allclose(gmm.cov_chol, cov_chol2)
    # (iii) call all methods (validity of results is confirmed by the other tests)
    z = torch.randn((10, 2))
    s = gmm.sample(n_samples=10)
    assert s.shape == (10, 2)
    ld, ldg = gmm.log_density(z=z, compute_grad=True)
    assert ld.shape == (10,)
    assert ldg.shape == (10, 2)
    lcd = gmm.log_component_densities(z=z)
    assert lcd.shape == (10, 3)
    lr = gmm.log_responsibilities(z=z)
    assert lr.shape == (10, 3)
    s = gmm.sample_all_components(num_samples_per_component=10)
    assert s.shape == (10, 3, 2)

    ## (2) n_batch_dims = 1
    # generate valid parameters
    # set 1
    log_w = torch.log(torch.tensor([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6]]))
    loc = torch.tensor(
        [
            [
                [3.0, 4.0],
                [-1.0, 2.0],
                [-7.0, -1.0],
            ],
            [
                [3.0, 4.0],
                [-1.0, 2.0],
                [-7.0, -1.0],
            ],
        ]
    )
    cov_chol = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [-0.5, 0.1],
                ],
                [
                    [7.0, 0.0],
                    [1.0, 7.0],
                ],
                [
                    [0.2, 0.0],
                    [-9.0, 9.0],
                ],
            ],
            [
                [
                    [1.0, 0.0],
                    [-0.5, 0.1],
                ],
                [
                    [7.0, 0.0],
                    [1.0, 7.0],
                ],
                [
                    [0.2, 0.0],
                    [-9.0, 9.0],
                ],
            ],
        ]
    )
    cov = cov_chol_to_cov(cov_chol)
    prec = torch.linalg.inv(cov)
    prec_chol = prec_to_prec_chol(prec)
    # set 2
    log_w2 = torch.log(torch.tensor([[0.5, 0.1, 0.4], [0.5, 0.1, 0.4]]))
    loc2 = torch.tensor(
        [
            [
                [1.0, 4.0],
                [1.0, 2.0],
                [-5.0, -1.0],
            ],
            [
                [1.0, 4.0],
                [1.0, 2.0],
                [-5.0, -1.0],
            ],
        ]
    )
    cov_chol2 = torch.tensor(
        [
            [
                [
                    [2.0, 0.0],
                    [-0.5, 1.1],
                ],
                [
                    [1.0, 0.0],
                    [2.0, 9.0],
                ],
                [
                    [0.3, 0.0],
                    [-9.0, 9.0],
                ],
            ],
            [
                [
                    [2.0, 0.0],
                    [-0.5, 1.1],
                ],
                [
                    [1.0, 0.0],
                    [2.0, 9.0],
                ],
                [
                    [0.3, 0.0],
                    [-9.0, 9.0],
                ],
            ],
        ]
    )
    cov2 = cov_chol_to_cov(cov_chol2)
    prec2 = torch.linalg.inv(cov2)
    prec_chol2 = prec_to_prec_chol(prec2)
    assert torch.allclose(torch.linalg.cholesky(cov2), cov_chol2)
    assert torch.allclose(torch.linalg.inv(cov2), prec2)
    assert torch.allclose(torch.linalg.cholesky(prec2), prec_chol2)
    # (i) Initialize with precision
    gmm = GMM(log_w=log_w, mean=loc, prec=prec)
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert torch.allclose(gmm.log_w, log_w)
    assert torch.allclose(gmm.prec, prec)
    assert torch.allclose(gmm.cov, cov)
    assert torch.allclose(gmm.prec_chol, prec_chol)
    assert torch.allclose(gmm.cov_chol, cov_chol)
    # (ii) Set new parameters (prec)
    gmm.log_w = log_w2
    gmm.loc = loc2
    gmm.prec = prec2
    assert gmm.n_components == 3
    assert gmm.d_z == 2
    assert torch.allclose(gmm.log_w, log_w2)
    assert torch.allclose(gmm.prec, prec2)
    assert torch.allclose(gmm.cov, cov2)
    assert torch.allclose(gmm.prec_chol, prec_chol2)
    assert torch.allclose(gmm.cov_chol, cov_chol2)
    # (iii) call all methods (validity of results is confirmed by the other tests)
    z = torch.randn((10, 2, 2))
    s = gmm.sample(n_samples=10)
    assert s.shape == (10, 2, 2)
    ld, ldg = gmm.log_density(z=z, compute_grad=True)
    assert ld.shape == (10, 2)
    assert ldg.shape == (10, 2, 2)
    lcd = gmm.log_component_densities(z=z)
    assert lcd.shape == (10, 2, 3)
    lr = gmm.log_responsibilities(z=z)
    assert lr.shape == (10, 2, 3)
    s = gmm.sample_all_components(num_samples_per_component=10)
    assert s.shape == (10, 2, 3, 2)


if __name__ == "__main__":
    test_prec_to_prec_chol()
    test_prec_to_cov_chol()
    test_cov_chol_to_cov()
    test_sample_gmm()
    test_gmm_log_component_densities()
    test_gmm_log_density()
    test_gmm_log_density_grad_hess()
    test_gmm()

    print("All tests passed!")
