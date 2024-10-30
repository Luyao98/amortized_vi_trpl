from typing import Tuple

import einops
import torch

from daft.src.gmm_util.gmm import GMM
from daft.src.multi_daft_vi.lnpdf import LNPDF
from daft.src.multi_daft_vi.more import MORE
from daft.src.multi_daft_vi.util_multi_daft import model_fitting, weight_update, compute_elbo


class MultiDaft:
    def __init__(
        self,
        algorithm_config: dict,
        target_dist: LNPDF,
        log_w_init: torch.Tensor,
        mean_init: torch.Tensor,
        prec_init: torch.Tensor,
        device: str = "cuda",
    ):
        self._device = device
        self.batch_shape = log_w_init.shape[:-1]
        self.n_components = log_w_init.shape[-1]
        self.d_z = mean_init.shape[-1]
        self.num_tasks = mean_init.shape[0]
        assert log_w_init.shape == self.batch_shape + (self.n_components,)
        assert mean_init.shape == self.batch_shape + (self.n_components, self.d_z)
        assert prec_init.shape == self.batch_shape + (self.n_components, self.d_z, self.d_z)

        self.config = algorithm_config
        self.more_config = algorithm_config.get("more")

        # check compatibility of model and tgt dist
        assert target_dist.get_num_dimensions() == self.d_z
        self.target_dist = target_dist

        # instantiate model, this is the GMM from which we draw samples and update the parameters to fit the tgt distribution
        self.model = GMM(
            log_w=log_w_init.to(device),
            mean=mean_init.to(device),
            prec=prec_init.to(device),
        )

        self.more_optimizer = MORE(
            batch_dim_and_components=torch.Size(self.batch_shape + (self.n_components,)),
            dim_z=self.d_z,
            log_space=False,  # log space not yet implemented
            conv_tol=self.more_config["dual_conv_tol"],
            global_upper_bound=self.more_config["global_upper_bound"],
            global_lower_bound=self.more_config["global_lower_bound"],
            # is bounded by 1 for non log space and 0 for log space
            use_warm_starts=self.more_config["use_warm_starts"],
            warm_start_interval_size=self.more_config["warm_start_interval_size"],
            max_prec_element_value=self.more_config["max_prec_element_value"],
            max_dual_steps=self.more_config["max_dual_steps"],
            device=device,
        )

    def step(self, logging: bool = False) -> float:
        # get samples and rewards from the model and tgt distribution
        samples, rewards, rewards_grad = self.get_samples_and_rewards()
        num_samples_per_component, num_task, num_components, dim_z = samples.shape

        # quadratic model fitting using Stein
        quad_term, lin_term = model_fitting(self.model.mean, self.model.prec, samples, rewards_grad)

        # Solve dual and get new parameters
        new_mean, new_prec = self.more_optimizer.step(
            torch.tensor(self.more_config["component_kl_bound"], dtype=torch.float32),
            self.model.mean,
            self.model.cov_chol,
            self.model.prec,
            quad_term,
            lin_term,
        )
        try:
            # update parameters
            self.model.mean = new_mean
            self.model.prec = new_prec
        except RuntimeError as e:
            print("Error in update: ", e)
            print("New mean: ", new_mean)
            print("New prec: ", new_prec)
            print("Old mean: ", self.model.mean)
            print("Old prec: ", self.model.prec)
            print("Old cov: ", self.model.cov)
            print("Old cov_chol: ", self.model.cov_chol)
            raise ValueError("Pos Def. Error.")

        # update of weights
        new_weights = weight_update(self.model.log_w, rewards)
        self.model.log_w = new_weights

        # if logging:
            # elbo
        with torch.no_grad():
            elbo = compute_elbo(
                model=self.model,
                target_dist=self.target_dist,
                num_samples_per_component=self.config["n_samples_per_comp"],
                mini_batch_size=self.config["mini_batch_size_for_target_density"],
            )

        return  - elbo.mean().detach().cpu().numpy()
        #     if len(elbo) == 1:
        #         # only one task, save the scalar
        #         elbo = elbo[0]
        #     results = {
        #         # "total_samples": num_samples_per_component * num_components * num_task,
        #         "elbo": elbo,
        #     }
        #     #                       "success": self.more_optimizer.old_success.detach().cpu().numpy(),
        #     #                        "eta": self.more_optimizer.old_eta.detach().cpu().numpy(),
        # else:
        #     results = {}
        #
        # return results

    def get_samples_and_rewards(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        samples_per_comp = self.model.sample_all_components(self.config["n_samples_per_comp"])
        rewards, rewards_grad = self._get_rewards(samples_per_comp)
        return samples_per_comp, rewards, rewards_grad

    def _get_rewards(self, samples_per_comp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_samples, num_tasks, num_components, dim_z = tuple(samples_per_comp.shape)
        samples_flattened = einops.rearrange(samples_per_comp, "s t c d -> (s c) t d")
        # should now have shape [num_samples_per_comp * num_comp, task, dz]
        # get the tgt and model log densities + gradients
        # use mini batch for tgt density, to avoid memory issues
        target_densities, target_grad = self.target_dist.mini_batch_log_density(
            samples_flattened.cpu(),
            mini_batch_size=self.config["mini_batch_size_for_target_density"],
            compute_grad=True,
        )
        model_densities, model_grad = self.model.log_density(samples_flattened, compute_grad=True)
        # combine to get rewards
        rewards_flatten = target_densities.to(self._device) - model_densities
        rewards_grad_flatten = target_grad.to(self._device)- model_grad
        # unflatten
        rewards = einops.rearrange(
            rewards_flatten,
            "(s c) t -> s t c",
            s=num_samples,
            c=num_components,
        )
        rewards_grad = einops.rearrange(
            rewards_grad_flatten,
            "(s c) t d -> s t c d",
            s=num_samples,
            c=num_components,
        )
        return rewards, rewards_grad

    def evaluation(self, num_samples: int) -> Tuple[float, float]:
        with torch.no_grad():
            model_samples = self.model.sample(num_samples)
            target_samples = self.target_dist.sample(num_samples)

            model_log_model, _ = self.model.log_density(model_samples, compute_grad=False)
            model_log_target, _ = self.model.log_density(target_samples.to(self._device), compute_grad=False)
            target_log_target, _ = self.target_dist.log_density(target_samples, compute_grad=False)
            target_log_model, _ = self.target_dist.log_density(model_samples.cpu(), compute_grad=False)

            mid_t = torch.logsumexp(torch.stack([target_log_target.to(self._device), model_log_target]), dim=0) - torch.log(torch.tensor(2.0))
            mid_m = torch.logsumexp(torch.stack([target_log_model.to(self._device), model_log_model]), dim=0) - torch.log(torch.tensor(2.0))

            kl_target_midpoint = target_log_target.to(self._device) - mid_t
            kl_model_midpoint = model_log_model - mid_m
            js_div = 0.5 * (kl_target_midpoint + kl_model_midpoint)

            kl_target_model = target_log_target.to(self._device) - model_log_target
            kl_model_target = model_log_model - target_log_model.to(self._device)
            jeffreys_div = 0.5 * (kl_target_model + kl_model_target)

        return js_div.mean().detach().cpu().numpy(), jeffreys_div.mean().detach().cpu().numpy()