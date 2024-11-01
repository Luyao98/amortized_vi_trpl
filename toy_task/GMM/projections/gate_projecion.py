import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import torch as ch
from torch.distributions import Categorical, kl_divergence


def compute_kl_div(pred, old):
    pred_dist = Categorical(probs=pred)
    old_dist = Categorical(probs=old)
    kl_div = kl_divergence(pred_dist, old_dist)
    return kl_div


def create_kl_projection_problem(n_components):
    """
    Create the CVXPY problem and corresponding CvxpyLayer for KL projection.
    :param n_components: Number of components for the gate.
    :return: A CvxpyLayer object for the KL projection problem.
    """
    # Create variables and parameters for the optimization
    p = cp.Variable(n_components, nonneg=True)
    q = cp.Parameter(n_components)
    r = cp.Parameter(n_components)
    eps = cp.Parameter(nonneg=True)

    # Define the objective function and constraints
    objective = cp.Minimize(cp.sum(cp.kl_div(p, q)))
    constraints = [
        cp.sum(cp.kl_div(p, r)) <= eps,
        cp.sum(p) == 1
    ]

    # Create the optimization problem and layer
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()  # Ensure the problem is DPP compliant
    cvxpylayer = CvxpyLayer(problem, parameters=[q, r, eps], variables=[p])

    return cvxpylayer


def kl_projection_gate(gate_old, gate_pred, epsilon):
    """
    Projects the predicted gates onto the feasible set of gates that are within epsilon KL divergence from the old gates.
    :param gate_old: Tensor of old gates.
    :param gate_pred: Tensor of predicted gates.
    :param epsilon: KL divergence threshold.
    :param cvxpylayer: Pre-created CvxpyLayer for KL projection.
    :return: Projected gates tensor.
    """

    batch_size, n_components = gate_pred.shape

    eps_batch = ch.full((batch_size,), epsilon, dtype=gate_pred.dtype, device=gate_pred.device)

    # Create variables and parameters for the optimization
    p = cp.Variable(n_components, nonneg=True)
    q = cp.Parameter(n_components)
    r = cp.Parameter(n_components)
    eps = cp.Parameter(nonneg=True)

    # Define the objective function and constraints
    objective = cp.Minimize(cp.sum(cp.kl_div(p, q)))
    constraints = [
        cp.sum(cp.kl_div(p, r)) <= eps,
        cp.sum(p) == 1
    ]

    # Create the optimization problem and layer
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()  # Ensure the problem is DPP compliant
    cvxpylayer = CvxpyLayer(problem, parameters=[q, r, eps], variables=[p])

    # quiet slow, just project all gates
    # # Compute KL divergence and apply the optimization layer
    # kl_divs = compute_kl_div(gate_pred, gate_old)
    # mask = kl_divs > epsilon
    # gate_proj = gate_pred.clone()
    #
    # # Only optimizing the gates that need to be projected
    # if mask.any():
    #     optimized_p = cvxpylayer(gate_pred[mask], gate_old[mask], eps_batch[mask])[0]
    #     gate_proj[mask] = optimized_p

    optimized_p = cvxpylayer(gate_pred, gate_old, eps_batch)[0]
    gate_proj = optimized_p

    return gate_proj


def kl_projection_gate_non_batch(gate_old, gate_pred, epsilon):
    """
    according to my tests, this version is not working, it requires much more memory
    :param gate_old:
    :param gate_pred:
    :param epsilon:
    :return:
    """
    batch_size, n_components = gate_pred.shape

    # Create variables and parameters for the optimization
    p = cp.Variable((batch_size, n_components), nonneg=True)
    q = cp.Parameter((batch_size, n_components))
    r = cp.Parameter((batch_size, n_components))
    eps = cp.Parameter(nonneg=True)

    # Define the objective function and constraints
    objective = cp.Minimize(cp.sum(cp.kl_div(p, q)))
    constraints = [
        cp.sum(cp.kl_div(p, r), axis=1) <= eps,
        cp.sum(p, axis=1) == 1
    ]

    # Create the optimization problem and layer
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()  # Ensure the problem is DPP compliant
    cvxpylayer = CvxpyLayer(problem, parameters=[q, r, eps], variables=[p])

    # # Compute KL divergence and apply the optimization layer
    # kl_divs = compute_kl_div(gate_pred, gate_old)
    # mask = kl_divs > epsilon
    # gate_proj = gate_pred.clone()
    #
    # # Only optimizing the gates that need to be projected
    # if mask.any():
    #     optimized_p = cvxpylayer(gate_pred[mask], gate_old[mask], ch.tensor([epsilon]))
    #     gate_proj[mask] = optimized_p[0]

    optimized_p = cvxpylayer(gate_pred, gate_old, ch.tensor([epsilon]))[0]
    gate_proj = optimized_p[0]

    return gate_proj


