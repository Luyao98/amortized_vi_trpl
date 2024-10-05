import einops
import torch

from daft.src.multi_daft_vi.util_more import bracketing_search, init_bracketing_search, get_distribution


class MORE:
    def __init__(
        self,
        batch_dim_and_components: torch.Size,
        dim_z: int,
        log_space: bool,
        conv_tol: float,
        global_upper_bound: float,
        global_lower_bound: float,
        use_warm_starts: bool,
        warm_start_interval_size: float,
        max_prec_element_value: float,
        max_dual_steps: int,
        device: str = "cpu",
    ):
        self._device = device
        self.log_space = torch.tensor(log_space, dtype=torch.bool).to(self._device)
        self.eta_lower_bound = torch.tensor(global_lower_bound, dtype=torch.float32).to(
            self._device
        )
        self.use_warm_starts = torch.tensor(use_warm_starts, dtype=torch.bool).to(self._device)
        self.warm_start_interval_size = torch.tensor(
            warm_start_interval_size, dtype=torch.float32
        ).to(self._device)
        self.max_dual_steps = max_dual_steps
        self.max_prec_element_value = max_prec_element_value

        # full range
        if self.log_space:
            self.global_lower_bound = torch.maximum(
                torch.tensor(0.0), torch.log(torch.tensor(global_lower_bound))
            ).to(self._device)
            self.global_upper_bound = torch.log(torch.tensor(global_upper_bound)).to(self._device)
        else:
            self.global_lower_bound = torch.maximum(
                torch.tensor(1.0), torch.tensor(global_lower_bound)
            ).to(self._device)
            self.global_upper_bound = torch.tensor(global_upper_bound).to(self._device)

        self.eye_matrix = einops.repeat(
            torch.eye(dim_z),
            "m n -> t c m n",
            t=batch_dim_and_components[0],
            c=batch_dim_and_components[1],
        ).to(self._device)
        # this selects if warm start or big range for dual optimization. In the beginning: no warm start
        self.old_success = torch.zeros(batch_dim_and_components, dtype=torch.bool).to(self._device)
        self.old_eta = (
            torch.ones(batch_dim_and_components, dtype=torch.float32).to(self._device)
            * global_upper_bound
        )
        self.conv_tol = torch.tensor(conv_tol).to(self._device)

    @property
    def eta(self):
        return self.old_eta

    @eta.setter
    def eta(self, value):
        self.old_eta = value

    @property
    def success(self):
        return self.old_success

    @success.setter
    def success(self, value):
        self.old_success = value

    def step(
        self,
        eps: torch.Tensor,
        old_mean: torch.Tensor,
        chol_old_cov: torch.Tensor,
        old_prec: torch.Tensor,
        reward_quad_term: torch.Tensor,
        reward_lin_term: torch.Tensor,
    ):
        dim_z = torch.tensor(reward_lin_term.shape[-1], dtype=torch.float32).to(self._device)
        (
            kl_const_part,
            old_lin_term,
            transposed_inv_chol_old_cov,
            lower_bound,
            upper_bound,
        ) = init_bracketing_search(
            chol_old_cov,
            old_prec,
            old_mean,
            dim_z,
            self.global_lower_bound,
            self.old_eta,
            self.old_success,
            self.global_lower_bound,
            self.global_upper_bound,
            self.log_space,
            self.use_warm_starts,
            self.warm_start_interval_size,
        )
        # solve dual
        if self.log_space:
            raise NotImplementedError("Log Space search not implemented yet")
            # self.old_eta = log_space_bracketing_search(eps, lower_bound, upper_bound, old_mean, old_lin_term, old_prec,
            #                                            transposed_inv_chol_old_cov,
            #                                            reward_lin_term, reward_quad_term,
            #                                            kl_const_part, self.eye_matrix)
        else:
            self.old_eta, num_iter = bracketing_search(
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
                self.eye_matrix,
                self.conv_tol,
                max_iter=self.max_dual_steps,
            )
            # print(f"{num_iter=}")
        new_mean, new_prec, self.old_success = get_distribution(
            self.old_eta, old_lin_term, old_prec, reward_lin_term, reward_quad_term, self.max_prec_element_value
        )
        return new_mean, new_prec
