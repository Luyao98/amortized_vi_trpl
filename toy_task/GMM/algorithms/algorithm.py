import  numpy as np
import torch as ch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, kl_divergence

from toy_task.GMM.models.GMM_model_3 import EmbeddedConditionalGMM
from toy_task.GMM.targets.abstract_target import AbstractTarget
from toy_task.GMM.models.model_factory import get_model
from toy_task.GMM.targets.target_factory import get_target
from toy_task.GMM.algorithms.visualization.GMM_plot import plot2d_matplotlib
from toy_task.GMM.algorithms.evaluation.JensenShannon_Div import js_divergence, js_divergence_gate, kl_divergence_gate
from toy_task.GMM.algorithms.evaluation.Jeffreys_Div import jeffreys_divergence
from toy_task.GMM.projections.split_kl_projection import split_kl_projection
# from toy_task.GMM.projections.gate_projection import kl_projection_gate
from toy_task.GMM.utils.torch_utils import get_numpy

import wandb


def init_some_components(model: EmbeddedConditionalGMM,
                         target: AbstractTarget,
                         contexts,
                         plot_contexts,
                         device,
                         scale
                         ):
    model.eval()
    with ch.no_grad():
        required_components = len(model.active_component_indices)
        current_gate, current_mean, current_chol = model(contexts)

        # draw samples from a basic Gaussian distribution for better exploration
        basic_mean = ch.zeros((contexts.shape[0], model.dim))
        basic_cov = scale * ch.eye(model.dim).unsqueeze(0).expand(contexts.shape[0], -1, -1)
        basic_samples = MultivariateNormal(loc=basic_mean, covariance_matrix=basic_cov).sample(ch.Size([100]))

        model_samples = model.get_samples_gmm(current_gate, current_mean, current_chol, 50)  # (s,c,f)
        samples = ch.cat([basic_samples.to(device), model_samples], dim=0)  # (s=s1+s2,c,f)
        with ch.enable_grad():
            samples = target.update_samples((contexts, samples), target.log_prob_tgt, lr=0.01, n=10)
        log_target = target.log_prob_tgt(contexts, samples)  # (s,c)

        max_value, max_idx = ch.max(log_target, dim=0)
        chosen_ctx = ch.argmax(max_value)

        sorted_values, sorted_indices = ch.sort(log_target[:, chosen_ctx], descending=True)
        chosen_sample = sorted_indices[:required_components]
        chosen_mean = samples[chosen_sample, chosen_ctx]

        # update the mean bias of the new component
        model.embedded_mean_bias[:required_components] += chosen_mean - current_mean[chosen_ctx]

        plot(model, target, device=device, contexts=plot_contexts)
    model.train()


def delete_components(model,
                      contexts,
                      threshold
                      ):
    current_log_gate, _, _ = model(contexts)
    avg_gate = ch.mean(ch.exp(current_log_gate), dim=0)

    deleted_indices = [idx for i, idx in enumerate(model.active_component_indices) if avg_gate[i] < threshold]
    model.active_component_indices = [idx for i, idx in enumerate(model.active_component_indices) if avg_gate[i] >= threshold]

    if deleted_indices == model.previous_deleted_indices:
        model.delete_counter += 1
    else:
        model.delete_counter = 1
    model.previous_deleted_indices = deleted_indices

    if deleted_indices:
        print(f"Deleting Step: remaining active components: {model.active_component_indices}.")
        print(f"Deleted components: {deleted_indices}. Consecutive same deletions: {model.delete_counter}")
    else:
        print(f"Deleting Step: no components deleted, currently {len(model.active_component_indices)} active components.")
        print(f"Gate values after deleting: {avg_gate}")


def add_components(model: EmbeddedConditionalGMM,
                   target: AbstractTarget,
                   contexts,
                   gate_strategy,
                   chol_scale,
                   scale,
                   update_sample,
                   lr,
                   itr
                   ):
    all_indices = set(range(model.max_components))
    active_indices = set(model.active_component_indices)
    available_indices = sorted(all_indices - active_indices)

    assert available_indices, "Adding failed. All components are active."

    idx = available_indices[0]
    model.add_component(idx)
    mask = ch.ones(len(model.active_component_indices), dtype=ch.bool)
    mask[idx] = False

    device = contexts.device
    current_gate, current_mean, current_chol = model(contexts)

    # init new gate
    unnormalized_current_gate = model.gate(contexts)
    avg_gate = ch.mean(ch.exp(unnormalized_current_gate[:, list(active_indices)]), dim=0)
    if gate_strategy == 1:
        # idea 1: from Hongyi, set the new gate w.r.t. the current gate
        set_gate = 0.001 * ch.max(avg_gate)
    elif gate_strategy == 2:
        # idea 2: treat the result from GateNN as unnormalized log of gate
        set_gate = ch.tensor(1e-4).to(device)
    elif gate_strategy == 3.1:
        # idea 3: based on idea 2, but dynamically set the gate
        set_gate = 1e-3 * ch.tensor(1 / len(model.active_component_indices)).to(device)
    elif gate_strategy == 3.2:
        # idea 3: based on idea 2, but dynamically set the gate
        set_gate = 1e-4 * ch.tensor(1 / len(model.active_component_indices)).to(device)
    elif gate_strategy == 3.3:
        # idea 3: based on idea 2, but dynamically set the gate
        set_gate = 1e-5 * ch.tensor(1 / len(model.active_component_indices)).to(device)
    elif gate_strategy == 3.4:
        # idea 3: based on idea 2, but dynamically set the gate
        set_gate = 1e-4 * ch.tensor(model.delete_counter / len(model.active_component_indices)).to(device)
    elif gate_strategy == 3.5:
        # idea 3: based on idea 2, but dynamically set the gate
        set_gate = 1e-4 * ch.tensor(1 / (len(model.active_component_indices)-1)).to(device)
    elif gate_strategy == 4:
        # basic idea from VIPS
        init_gates = ch.tensor([1000, 500, 250, 100])
        random_index = ch.randint(0, len(init_gates), (1,)).item()
        set_gate = ch.exp(-init_gates[random_index]).to(device)
    else:
        raise ValueError("Invalid gate strategy.")
    new_component_gate = ch.log(set_gate)
    # new_gate_bias = ch.log(set_gate * ch.sum(avg_gate) / (1 - set_gate))
    print("Adding Step: new component gate:", get_numpy(set_gate))

    # draw samples from a basic Gaussian distribution for better exploration
    basic_mean = ch.zeros((contexts.shape[0], model.dim))
    basic_cov = scale * ch.eye(model.dim).unsqueeze(0).expand(contexts.shape[0], -1, -1)
    basic_samples = MultivariateNormal(loc=basic_mean, covariance_matrix=basic_cov).sample(ch.Size([10])).to(device)

    model_samples = model.get_samples_gmm(current_gate[:, mask], current_mean[:, mask], current_chol[:, mask], 10)
    samples = ch.cat([basic_samples, model_samples], dim=0) # (s=s1+s2,c,f)

    # improve the sample quality depending on the gradient of the log density
    if update_sample:
        # test time
        start_event = ch.cuda.Event(enable_timing=True)
        end_event = ch.cuda.Event(enable_timing=True)
        start_event.record()

        with ch.enable_grad():
            samples = target.update_samples((contexts, samples), target.log_prob_tgt, lr, itr)

        end_event.record()
        ch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        wandb.log({"elapsed_time/ms": float(elapsed_time)})
        # test end

    log_model = model.log_prob_gmm(current_mean[:, mask], current_chol[:, mask], current_gate[:, mask], samples)
    log_target = target.log_prob_tgt(contexts, samples)

    # log density of new components = \log q(o_n|c) + \log q_{x_s}(x_s|o_n,c)
    log_new_o = MultivariateNormal(loc=samples, scale_tril=current_chol[:, idx].unsqueeze(0)).log_prob(samples)

    rewards = log_target - ch.max(log_model, new_component_gate + log_new_o)  # (s, c)

    flat_idx = ch.argmax(rewards)
    row_idx, col_idx = ch.div(flat_idx, rewards.size(1), rounding_mode='floor'), flat_idx % rewards.size(1)
    chosen_mean = samples[row_idx, col_idx]
    chosen_context = contexts[col_idx]
    print("Adding Step: new component mean:", chosen_mean)

    # update the mean bias of the new component
    model.embedded_mean_bias[idx] += chosen_mean - current_mean[col_idx, idx]

    # update the cholesky bias of the new component
    new_component_chol = chol_scale * ch.eye(model.dim, dtype=ch.float32).to(device)
    chol_bias = new_component_chol - current_chol[col_idx, idx] + model.embedded_chol_bias[idx]
    # remove negative bias to avoid negative chol
    for i in range(chol_bias.size(0)):
        if chol_bias[i, i] < 0:
            chol_bias[i, i] = 0
            print("Adding Step: desired chol too small, using origin chol.")
    model.embedded_chol_bias[idx] = chol_bias

    # update the gate of the new component
    chosen_gate = unnormalized_current_gate[col_idx, list(active_indices)]
    new_gate_bias = ch.log(set_gate / (1 - set_gate)) + ch.logsumexp(chosen_gate, dim=0)
    model.gate.fc_gate.bias.data[idx] = new_gate_bias
    model.gate.fc_gate.weight.data[idx] = ch.tensor(0, dtype=ch.float32).to(device)

    return chosen_context


def adaptive_components(model: EmbeddedConditionalGMM,
                        target: AbstractTarget,
                        adaption_contexts,
                        plot_contexts,
                        device,
                        adaption_config
                        ):
    threshold = adaption_config["threshold"]

    gate_strategy = adaption_config["gate_strategy"]
    chol_scale = adaption_config["chol_scale"]
    scale = adaption_config["scale"]
    update_sample = adaption_config["update_sample"]
    lr = adaption_config["lr"]
    itr = adaption_config["itr"]

    model.eval()
    with ch.no_grad():
        delete_components(model, adaption_contexts, threshold)
        if len(model.active_component_indices) < model.max_components:
            chosen_context = add_components(model, target, adaption_contexts, gate_strategy, chol_scale, scale,
                                            update_sample, lr, itr)
            new_plot_contexts = ch.cat([plot_contexts, chosen_context.unsqueeze(0).to('cpu')])
        else:
            new_plot_contexts = plot_contexts
        plot(model, target, device=device, contexts=new_plot_contexts, plot_type="Adaptive Step")
    model.train()


def get_all_contexts(target,
                     n_context,
                     device
                     ):
    train_contexts = target.get_contexts(n_context).to(device)
    eval_contexts = target.get_contexts(200).to(device)
    # bmm and gmm
    # plot_contexts = ch.tensor([[-0.3],
    #                            [0.7],
    #                            [-1.8]])

    # funnel
    # plot_contexts = ch.tensor([[-0.3],
    #                            [0.1],
    #                            [-0.8]])

    # random gmm
    plot_contexts = target.get_contexts(3)
    return train_contexts, eval_contexts, plot_contexts


def get_optimizer(model,
                  gate_lr,
                  gaussian_lr
                  ):
    return optim.Adam([
        {'params': model.gate.parameters(), 'lr': gate_lr,},
        {'params': model.gaussian_list.parameters(), 'lr': gaussian_lr}
    ])


def evaluate_model(model: EmbeddedConditionalGMM,
                   target: AbstractTarget,
                   eval_contexts,
                   plot_contexts,
                   epoch,
                   n_epochs,
                   adapt,
                   adaption,
                   loss_history,
                   history_size,
                   device
                   ):
    if adapt:
        if len(loss_history) > history_size:
            loss_history.pop(0)
            assert len(loss_history) == history_size

        if len(loss_history) == history_size:
            loss_history_array = np.array(loss_history)
            first_half = loss_history_array[:history_size // 2].sum()
            second_half = loss_history_array[history_size // 2:].sum()
            if np.abs(first_half - second_half) / second_half < 0.1:
                adaption = True
                loss_history = []  # reset loss history, to avoid immediate adaption
                if len(model.active_component_indices) < model.max_components:
                    print(f"\nStability reached at epoch {epoch}. Start adaption.")
                else:
                    print(f"\nStability reached at epoch {epoch} with max active components.")

    n_eval = 10
    n_plot = 10
    if (epoch + 1) % (n_epochs // n_eval)== 0:
        model.eval()
        with ch.no_grad():
            # approx_reward = ch.cat(batched_approx_reward, dim=1)
            # js_divergence_gates = js_divergence_gate(approx_reward, model, eval_contexts, device)
            # kl_gate = kl_divergence_gate(approx_reward, model, eval_contexts, device)
            js_div = js_divergence(model, target, eval_contexts, device)
            # j_div = jeffreys_divergence(model, target, eval_contexts, device)

            if (epoch + 1) % (n_epochs // n_plot) == 0:
                # check the active components before final evaluation
                if epoch + 1 == n_epochs:
                    delete_components(model, eval_contexts, 1e-3)
                plot(model, target, device=device, contexts=plot_contexts)

            wandb.log({
                # "reverse KL between gates": kl_gate.item(),
                # "Jensen Shannon Divergence between gates": js_divergence_gates.item(),
                "Jensen Shannon Divergence": js_div.item(),
                # "Jeffreys Divergence": j_div.item()
            })
        print("current epoch:", epoch)
        model.train()
    return loss_history, history_size, adaption


def plot(model: EmbeddedConditionalGMM,
         target: AbstractTarget,
         device,
         contexts=None,
         plot_type="Evaluation",
         logging=False
         ):
    # if not logging:
    #     return

    if contexts is None:
        contexts = target.get_contexts(3).to('cpu')
    else:
        contexts = contexts.clone().detach().to('cpu')
    plot2d_matplotlib(target, model.to('cpu'), contexts, plot_type=plot_type,
                      min_x=-5, max_x=5, min_y=-5, max_y=5)
    model.to(device)


def step(model: EmbeddedConditionalGMM,
         target: AbstractTarget,
         shuffled_contexts,
         training_config,
         optimizer
         ):

    train_size = training_config["n_context"]
    batch_size = training_config["batch_size"]
    n_samples = training_config["n_samples"]

    eva_loss = 0
    for batch_idx in range(0, train_size, batch_size):
        b_contexts = shuffled_contexts[batch_idx:batch_idx + batch_size]

        # Prediction step
        gate_pred, mean_pred, chol_pred = model(b_contexts)

        # Compute ELBO loss
        loss = compute_elbo_loss(model, target, b_contexts, mean_pred, chol_pred, gate_pred, n_samples)

        # Update model
        eva_loss += get_numpy(loss)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        wandb.log({"train_loss": loss.item()})

    # Shuffle sampled contexts
    indices = ch.randperm(train_size)
    shuffled_contexts = shuffled_contexts[indices]

    return eva_loss, shuffled_contexts


def step_with_component_projection_minibatch(model: EmbeddedConditionalGMM,
                                             target: AbstractTarget,
                                             shuffled_contexts,
                                             training_config,
                                             projection_config,
                                             optimizer,
                                             old_dist,
                                             ):

    train_size = training_config["n_context"]
    batch_size = training_config["batch_size"]
    n_samples = training_config["n_samples"]
    eps_mean = projection_config["eps_mean"]
    eps_cov = projection_config["eps_cov"]
    alpha = projection_config["alpha"]

    b_gate_old, b_mean_old, b_chol_old = old_dist
    eva_loss = 0
    for batch_idx in range(0, train_size, batch_size):
        b_contexts = shuffled_contexts[batch_idx:batch_idx + batch_size]

        # Prediction
        gate_pred, mean_pred, chol_pred = model(b_contexts)

        # Projection
        batch_size, n_components, dz = mean_pred.shape

        mean_proj_flatten, chol_proj_flatten = split_kl_projection(mean_pred.view(-1, dz), chol_pred.view(-1, dz, dz),
                                                                   b_mean_old.view(-1, dz).clone().detach(),
                                                                   b_chol_old.view(-1, dz, dz).clone().detach(),
                                                                   eps_mean, eps_cov)

        mean_proj = mean_proj_flatten.view(batch_size, n_components, dz)
        chol_proj = chol_proj_flatten.view(batch_size, n_components, dz, dz)

        # Compute ELBO
        pred_dist = MultivariateNormal(loc=mean_pred, scale_tril=chol_pred)
        proj_dist = MultivariateNormal(loc=mean_proj.clone().detach(), scale_tril=chol_proj.clone().detach())
        reg_loss = alpha * kl_divergence(proj_dist, pred_dist).unsqueeze(0)
        loss = compute_elbo_loss(model, target, b_contexts, mean_proj, chol_proj, gate_pred, n_samples, reg_loss)

        with ch.no_grad():
            # get old distribution for the next batch
            if batch_idx + batch_size < len(shuffled_contexts):
                b_next_context = shuffled_contexts[batch_idx + batch_size:batch_idx + 2 * batch_size]
                b_gate_old, b_mean_old, b_chol_old = model(b_next_context)
            else:
                assert batch_idx + batch_size == len(shuffled_contexts)
                # Shuffle sampled contexts
                indices = ch.randperm(train_size, device=shuffled_contexts.device)
                new_shuffled_contexts = shuffled_contexts[indices]
                first_batch = new_shuffled_contexts[0:batch_size]
                old_dist_first_batch = model(first_batch)

        # Update model
        eva_loss = eva_loss + get_numpy(loss)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        wandb.log({"train_loss": loss.item()})

    return eva_loss, new_shuffled_contexts, old_dist_first_batch


def step_with_component_projection(model: EmbeddedConditionalGMM,
                                   target: AbstractTarget,
                                   ctx,
                                   training_config,
                                   projection_config,
                                   optimizer,
                                   old_dist,
                                   ):

    train_size = training_config["n_context"]
    batch_size = training_config["batch_size"]
    n_samples = training_config["n_samples"]
    eps_mean = projection_config["eps_mean"]
    eps_cov = projection_config["eps_cov"]
    alpha = projection_config["alpha"]
    assert batch_size == train_size

    # Prediction
    gate_pred, mean_pred, chol_pred = model(ctx)

    # Projection
    b_gate_old, b_mean_old, b_chol_old = old_dist
    batch_size, n_components, dz = mean_pred.shape

    mean_proj_flatten, chol_proj_flatten = split_kl_projection(mean_pred.view(-1, dz), chol_pred.view(-1, dz, dz),
                                                               b_mean_old.view(-1, dz).clone().detach(),
                                                               b_chol_old.view(-1, dz, dz).clone().detach(),
                                                               eps_mean, eps_cov)

    mean_proj = mean_proj_flatten.view(batch_size, n_components, dz)
    chol_proj = chol_proj_flatten.view(batch_size, n_components, dz, dz)

    # Compute ELBO
    pred_dist = MultivariateNormal(loc=mean_pred, scale_tril=chol_pred)
    proj_dist = MultivariateNormal(loc=mean_proj.clone().detach(), scale_tril=chol_proj.clone().detach())
    reg_loss = alpha * kl_divergence(proj_dist, pred_dist).unsqueeze(0)
    loss = compute_elbo_loss(model, target, ctx, mean_proj, chol_proj, gate_pred, n_samples, reg_loss)

    # Update model
    eva_loss = get_numpy(loss)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()

    # get old distribution for the next batch
    old_dist_update = gate_pred.clone().detach(), mean_pred.clone().detach(), chol_pred.clone().detach()

    wandb.log({"train_loss": loss.item()})

    return eva_loss, old_dist_update


def compute_elbo_loss(model: EmbeddedConditionalGMM,
                      target: AbstractTarget,
                      contexts,
                      mean,
                      chol,
                      gate,
                      n_samples,
                      reg_loss=0
                      ):
    model_samples = model.get_rsamples(mean, chol, n_samples)  # (S, C, O, dz)
    log_model_component = model.log_prob(mean, chol, model_samples)  # (S, C, O)
    log_responsibility = model.log_responsibilities_gmm(mean.clone().detach(), chol.clone().detach(),
                                                        gate.clone().detach(), model_samples)  # (S, C, O)
    log_target = target.log_prob_tgt(contexts, model_samples)  # (S, C, O)

    elbo = ch.exp(gate.squeeze(0)) * (log_model_component - log_target - log_responsibility +
                                      reg_loss + gate.squeeze(0))
    loss = elbo.sum(dim=-1).mean()
    return loss


def toy_task_new(config):

    model_config = config["model_config"]
    target_config = config["target_config"]
    training_config = config["training_config"]
    projection_config = config["projection"]

    # Device
    device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Target
    target = get_target(target_name=target_config["target_name"],
                        target_components=target_config["target_components"],
                        context_dim=model_config["context_dim"]
                        ).to(device)
    # Model
    model = get_model(model_config, device)

    # Training
    n_epochs = training_config["n_epochs"]
    batch_size = training_config["batch_size"]
    n_context = training_config["n_context"]
    gate_lr = training_config["gate_lr"]
    gaussian_lr = training_config["gaussian_lr"]

    adaption_config = training_config["adaption"]
    adapt = adaption_config["adapt"]
    scale = adaption_config["scale"]
    history_size = adaption_config["history_size"]

    optimizer = get_optimizer(model, gate_lr, gaussian_lr)
    contexts, eval_contexts, plot_contexts = get_all_contexts(target, n_context, device)
    loss_history = []
    adaption = False
    # eva_loss = 0

    # Initialize and plot the model
    init_some_components(model, target, contexts, plot_contexts, device, scale)
    # get old dist for first batch
    old_dist = model(contexts[0:batch_size])

    for epoch in range(n_epochs):
        # Add and delete components if indicator is True
        if adapt and adaption:
            adaptive_components(model, target, contexts[0:batch_size], plot_contexts, device, adaption_config)
            adaption = False

        # Perform training step
        component_project = projection_config["component_project"]
        gate_project = projection_config["gate_project"]
        if component_project:
            if gate_project:
                pass
            else:
                # eva_loss, old_dist = step_with_component_projection(model, target, contexts, training_config,
                #                                                              projection_config, optimizer, old_dist)
                eva_loss, contexts, old_dist = step_with_component_projection_minibatch(model, target, contexts,
                                                                                         training_config, projection_config,
                                                                                         optimizer, old_dist)
        else:
            if gate_project:
                pass
            else:
                eva_loss, contexts = step(model, target, contexts, training_config, optimizer)

        # Evaluate model and update loss history
        # loss_history.append(eva_loss)
        loss_history, history_size, adaption = evaluate_model(model, target, eval_contexts, plot_contexts, epoch,
                                                              n_epochs, adapt, adaption, loss_history, history_size, device)

    print("Training done!")


if __name__ == "__main__":
    import os
    import sys
    from omegaconf import OmegaConf
    from toy_task.GMM.utils.network_utils import set_seed

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    set_seed(1003)

    config_path = "../conf/gmm_target/10_gmm_2d_target.yaml"
    test_config = OmegaConf.load(config_path)
    gmm_config = OmegaConf.to_container(test_config, resolve=True)

    run_name = "2d_context_10_init_components_no_adaption"
    group_name = "test"
    wandb.init(project="spiral_gmm_target", group=group_name, name=run_name, config=gmm_config)
    toy_task_new(gmm_config)
