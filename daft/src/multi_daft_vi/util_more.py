import torch
import einops


# @tf.function(
#     input_signature=[
#         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # eta
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # old_lin_term
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # old_prec
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # reward_lin_term
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # reward_quad_term
#     ]
# )
def get_natural_params_simple_reward(
    eta, old_lin_term, old_precision, reward_lin_term, reward_quad_term
):
    """
    Uses the reformulation of the reward with log(target_densitiy) - log(model_density). This leads to the parameterization
    new_prec = old_prec - 1/eta * reward_quad_term
    new_lin = old_lin + 1/eta * reward_lin_term
    """
    expanded_eta = einops.rearrange(eta, "... -> ... 1")
    new_lin = old_lin_term + reward_lin_term / expanded_eta
    twice_expanded_eta = einops.rearrange(expanded_eta, "... first_exp -> ... first_exp 1")
    new_precision = old_precision - reward_quad_term / twice_expanded_eta
    return new_lin, new_precision


# def get_natural_params_max_entropy(eta, old_lin_term, old_precision, reward_lin_term, reward_quad_term):
#     expanded_eta = tf.expand_dims(eta, -1)
#     new_lin = (expanded_eta * old_lin_term + reward_lin_term) / (expanded_eta + 1)
#     twice_expanded_eta = tf.expand_dims(expanded_eta, -1)
#     new_precision = (twice_expanded_eta * old_precision - reward_quad_term) / (twice_expanded_eta + 1)
#     return new_lin, new_precision


# @tf.function(
#     input_signature=[
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # chol_old_cov
#         tf.TensorSpec(shape=[], dtype=torch.float32),  # dim_z
#     ]
# )
def get_kl_const_part(chol_old_cov, dim_z):
    old_logdet = 2 * torch.sum(torch.log(torch.linalg.diagonal(chol_old_cov)), dim=-1)
    return old_logdet - dim_z


# @tf.function(
#     input_signature=[
#         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # eta
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # old_lin_term
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # old_mean
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # old_prec
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # inv_chol_old_cov
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # reward_lin_term
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # reward_quad_term
#         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # kl_const_part
#         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # eye_matrix
#     ]
# )
def get_kl(
    eta,
    old_lin_term,
    old_mean,
    old_prec,
    transposed_inv_chol_old_cov,
    reward_lin_term,
    reward_quad_term,
    kl_const_part,
    eye_matrix,
):
    new_lin, new_prec = get_natural_params_simple_reward(
        eta, old_lin_term, old_prec, reward_lin_term, reward_quad_term
    )

    new_prec_chol, info = torch.linalg.cholesky_ex(new_prec)
    # check the info dict for values > 0 indicating that here the corresponding matrix was not positive def.
    error_indices = info > 0
    error_indices = einops.rearrange(error_indices, "t c -> t c 1 1")
    # replace nans with identities such that the program does not throw errors.
    # ATTENTION: need to re-replace these with nans before outputting
    safe_new_prec_chol = torch.where(error_indices, eye_matrix, new_prec_chol)
    safe_new_prec = torch.where(error_indices, eye_matrix, new_prec)
    # compute parts for the kl:
    new_logdet = -2 * torch.sum(torch.log(torch.linalg.diagonal(safe_new_prec_chol)), dim=-1)

    # uses that trace(M@M.T) = ||M||^2_2, and that trace has its cyclic property and cholesky identities
    # we can use triangular solve since safe_chol_new_prec is lower triangular
    trace_matrix = torch.linalg.solve_triangular(
        safe_new_prec_chol, transposed_inv_chol_old_cov, upper=False, left=True
    )
    trace_term = torch.sum(trace_matrix * trace_matrix, dim=(-2, -1))
    # compute the new_mean with the safe_new_prec to make sure that there is no matrix inversion error.
    new_mean = torch.linalg.solve(safe_new_prec, einops.rearrange(new_lin, "... -> ... 1"))[..., 0]
    diff = old_mean - new_mean  # shape batch x dim_z
    # TODO make sure transposition is correct, but looks fine imo
    chol_prec_times_diff = einops.einsum(
        transposed_inv_chol_old_cov, diff, "... dz2 dz, ... dz2 -> ... dz"
    )
    mahalanobis_dist = einops.reduce(
        chol_prec_times_diff * chol_prec_times_diff, "... dz -> ...", "sum"
    )
    kl = 0.5 * (kl_const_part - new_logdet + trace_term + mahalanobis_dist)
    # replace the components flagged as nan with real nans
    kl = torch.where(error_indices[..., 0, 0], torch.tensor(float("NaN")).to(kl.device), kl)
    return kl


# @tf.function(
#     input_signature=[
#         tf.TensorSpec(shape=[], dtype=torch.float32),  # eps
#         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # lower_bound
#         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # upper_bound
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # old_mean
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # old_lin_term
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # old_prec
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # inv_chol_old_cov
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # reward_lin_term
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # reward_quad_term
#         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # kl_const_part
#         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # eye_matrix
#     ]
# )
# def log_space_bracketing_search(eps, lower_bound, upper_bound, old_mean, old_lin_term, old_prec,
#                                 transposed_inv_chol_old_cov,
#                                 reward_lin_term,
#                                 reward_quad_term,
#                                 kl_const_part, eye_matrix):
#     for iters in tf.range(1000):
#         eta = 0.5 * (upper_bound + lower_bound)
#         # test current eta
#         kl = get_kl(tf.math.exp(eta), old_lin_term, old_mean, old_prec, transposed_inv_chol_old_cov, reward_lin_term,
#                     reward_quad_term,
#                     kl_const_part,
#                     eye_matrix)
#         converged = tf.math.exp(upper_bound) - tf.exp(lower_bound) < 1e-4
#         if tf.math.reduce_all(converged):
#             break
#         # if kl is nan, condition is also false, which is what we want
#         f_eval = eps - kl
#         condition = f_eval > 0
#         upper_bound = tf.where(condition, eta, upper_bound)
#         lower_bound = tf.where(condition, lower_bound, eta)
#     # use upper_bound as final value (less greedy)
#     return tf.math.exp(upper_bound)


# @tf.function(
#     input_signature=[
#         tf.TensorSpec(shape=[], dtype=torch.float32),  # eps
#         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # lower_bound
#         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # upper_bound
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # old_mean
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # old_lin_term
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # old_prec
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # transposed_inv_chol_old_cov
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # reward_lin_term
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # reward_quad_term
#         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # kl_const_part
#         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # eye_matrix
#         tf.TensorSpec(shape=[], dtype=torch.float32),  # conv_tol
#         tf.TensorSpec(shape=[], dtype=tf.int32),  # max_iter
#     ]
# )
def bracketing_search(
    eps,
    lower_bound,
    upper_bound,
    old_mean,
    old_lin_term,
    old_prec,
    transposed_inv_chol_old_cov,
    reward_lin_term,
    reward_quad_term,
    kl_const_part,
    eye_matrix,
    conv_tol,
    max_iter,
):
    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # lower_bound
    #         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # upper_bound
    #         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # eta
    #         tf.TensorSpec(shape=[], dtype=tf.int32),  # max_iter
    #     ]
    # )
    def converged_condition(
        lower_bound_var, upper_bound_var: torch.Tensor, eta: torch.Tensor, num_iter
    ):
        converged = torch.all(
            torch.any(
                torch.stack((upper_bound_var - eta < conv_tol, eta - lower_bound_var < conv_tol)),
                dim=0,
            )
        )
        # repeat until it is converged
        return torch.any(torch.tensor([converged, num_iter >= max_iter]))
        # return num_iter < max_iter

    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # lower_bound
    #         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # upper_bound
    #         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # eta
    #         tf.TensorSpec(shape=[], dtype=tf.int32),  # num_iter
    #     ]
    # )
    def loop_body(lower_bound_var, upper_bound_var, eta, num_iter):
        kl: torch.Tensor = get_kl(
            eta,
            old_lin_term,
            old_mean,
            old_prec,
            transposed_inv_chol_old_cov,
            reward_lin_term,
            reward_quad_term,
            kl_const_part,
            eye_matrix,
        )
        condition: torch.Tensor = kl < eps
        upper_bound_var = torch.where(condition, eta, upper_bound_var)
        lower_bound_var = torch.where(condition, lower_bound_var, eta)
        eta = 0.5 * (upper_bound_var + lower_bound_var)
        return lower_bound_var, upper_bound_var, eta, num_iter + 1

    num_iter = torch.tensor(0, dtype=torch.int32)
    eta = 0.5 * (upper_bound + lower_bound)
    while not converged_condition(lower_bound, upper_bound, eta, num_iter):
        lower_bound, upper_bound, eta, num_iter = loop_body(lower_bound, upper_bound, eta, num_iter)

    return upper_bound, int(num_iter)


# @tf.function(
#     input_signature=[
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # new_lin_term
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # new_prec
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # old_lin_term
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # old_prec
#     ]
# )
def satisfy_pd_constraint_and_upper_limit(new_lin_term, new_prec, old_lin_term, old_prec, max_prec_element_value):
    # test for pd by chol decomp
    # new_prec = torch.clip(new_prec, max=max_prec_element_value)
    new_prec_chol, info = torch.linalg.cholesky_ex(new_prec)
    # check the info dict for values > 0 indicating that here the corresponding matrix was not positive def.
    error_indices = info > 0
    # error_indices has shape ([batch_dims], num_components)
    error_indices = einops.rearrange(error_indices, "t c -> t c 1 1")
    # where the update fails it takes the old parameters
    safe_new_prec = torch.where(error_indices, old_prec, new_prec)
    safe_new_lin_term = torch.where(error_indices[..., 0], old_lin_term, new_lin_term)

    return safe_new_lin_term, safe_new_prec, torch.logical_not(error_indices[..., 0, 0])


# @tf.function(
#     input_signature=[
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # chol_old_cov
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # old_prec
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # old_mean_term
#         tf.TensorSpec(shape=[], dtype=torch.float32),  # dim_z
#         tf.TensorSpec(shape=[], dtype=torch.float32),  # eta_lower_bound
#         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # old_eta
#         tf.TensorSpec(shape=[None, None], dtype=tf.bool),  # old_success
#         tf.TensorSpec(shape=[], dtype=torch.float32),  # global_lower_bound
#         tf.TensorSpec(shape=[], dtype=torch.float32),  # global_upper_bound
#         tf.TensorSpec(shape=[], dtype=tf.bool),  # log_space
#         tf.TensorSpec(shape=[], dtype=tf.bool),  # use_warm_starts
#         tf.TensorSpec(shape=[], dtype=torch.float32),  # warm_start_interval_size
#     ]
# )
def init_bracketing_search(
    chol_old_cov,
    old_prec,
    old_mean,
    dim_z,
    eta_lower_bound,
    old_eta: torch.Tensor,
    old_success,
    global_lower_bound,
    global_upper_bound,
    log_space,
    use_warm_starts,
    warm_start_interval_size,
):
    kl_const_part = get_kl_const_part(chol_old_cov, dim_z)
    old_lin_term = einops.einsum(old_prec, old_mean, "... i j, ... j -> ... i")
    transposed_inv_chol_old_cov = einops.rearrange(
        torch.linalg.inv(chol_old_cov), "... i j -> ... j i"
    )

    if use_warm_starts:
        # warm start
        if log_space:
            lower_bound = torch.maximum(
                torch.maximum(torch.tensor(0.0), torch.log(eta_lower_bound)),
                torch.log(old_eta) - 0.1,
            )
            upper_bound = torch.log(old_eta) + 0.1
        else:
            lower_bound = torch.maximum(
                torch.maximum(torch.tensor(1.0), eta_lower_bound),
                old_eta - warm_start_interval_size / 2.0,
            )
            upper_bound = old_eta + warm_start_interval_size / 2.0

        # select warm start on previous successful updates
        lower_bound = torch.where(old_success, lower_bound, global_lower_bound)
        upper_bound = torch.where(old_success, upper_bound, global_upper_bound)
    else:
        identity = torch.ones_like(old_eta)
        lower_bound = identity * global_lower_bound
        upper_bound = identity * global_upper_bound

    return kl_const_part, old_lin_term, transposed_inv_chol_old_cov, lower_bound, upper_bound


# @tf.function(
#     input_signature=[
#         tf.TensorSpec(shape=[None, None], dtype=torch.float32),  # eta
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # old_lin_term
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # old_prec
#         tf.TensorSpec(shape=[None, None, None], dtype=torch.float32),  # reward_lin
#         tf.TensorSpec(shape=[None, None, None, None], dtype=torch.float32),  # reward_quad
#     ]
# )
def get_distribution(eta, old_lin_term, old_prec, reward_lin_term, reward_quad_term, max_prec_element_value):
    new_lin_term, new_prec = get_natural_params_simple_reward(
        eta, old_lin_term, old_prec, reward_lin_term, reward_quad_term
    )
    # test the output to make sure that precision is positive definite
    new_lin_term, new_prec, success = satisfy_pd_constraint_and_upper_limit(
        new_lin_term, new_prec, old_lin_term, old_prec, max_prec_element_value
    )
    new_mean = torch.linalg.solve(new_prec, einops.rearrange(new_lin_term, "... -> ... 1"))[..., 0]
    # do the checking of the prec after the calculation of the mean
    # this way, we update the mean, but still make sure that our prec does not get too large
    new_prec = satisfy_max_prec_element_value(max_prec_element_value, new_prec, old_prec)

    return new_mean, new_prec, success


def satisfy_max_prec_element_value(max_prec_element_value, new_prec, old_prec):
    max_elements = torch.amax(torch.abs(new_prec), dim=(-1, -2))
    max_violation = max_elements > max_prec_element_value
    max_violation = einops.rearrange(max_violation, "t c -> t c 1 1")
    new_prec = torch.where(max_violation, old_prec, new_prec)
    return new_prec
