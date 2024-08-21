import torch as ch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, kl_divergence
import wandb

from toy_task.GMM.models.GMM_model import ConditionalGMM
from toy_task.GMM.models.GMM_model_2 import ConditionalGMM2
from toy_task.GMM.models.GMM_model_3 import ConditionalGMM3
from toy_task.GMM.targets.abstract_target import AbstractTarget
from toy_task.GMM.models.model_factory import get_model
from toy_task.GMM.targets.target_factory import get_target
from toy_task.GMM.algorithms.visualization.GMM_plot import plot2d_matplotlib
from toy_task.GMM.algorithms.evaluation.JensenShannon_Div import js_divergence, js_divergence_gate, kl_divergence_gate
from toy_task.GMM.algorithms.evaluation.Jeffreys_Div import jeffreys_divergence
from toy_task.GMM.projections.split_kl_projection import split_kl_projection


def delete_components(model, contexts, threshold=0.001):
    current_log_gate, _, _ = model(contexts)
    avg_gate = ch.mean(ch.exp(current_log_gate), dim=0)

    model.active_component_indices = [
        idx for i, idx in enumerate(model.active_component_indices) if avg_gate[i] >= threshold
    ]

    print(f"Deleting Step: remaining active components: {model.active_component_indices}")


def add_components(model, target, contexts):
    all_indices = set(range(model.max_components))
    active_indices = set(model.active_component_indices)
    available_indices = sorted(all_indices - active_indices)

    if not available_indices:
        raise ValueError("Adding failed. All components are active.")

    idx = available_indices[0]
    model.add_component(idx)

    device = contexts.device
    current_gate, current_mean, current_chol = model(contexts)
    new_component_gate = ch.log(ch.tensor(0.0001)).to(device)

    # strategy in VIPS++ paper
    # init_gates = ch.tensor([100, 50, 25, 10])
    # random_index = ch.randint(0, len(init_gates), (1,)).item()
    # new_component_gate = -init_gates[random_index].to(device)

    # draw samples from a basic Gaussian distribution for better exploration
    basic_mean = ch.zeros((contexts.shape[0], model.dim)).to(device)
    basic_cov = 100 * ch.eye(model.dim).unsqueeze(0).expand(contexts.shape[0], -1, -1).to(device)
    basic_samples = MultivariateNormal(loc=basic_mean, covariance_matrix=basic_cov).sample(ch.Size([10]))

    model_samples = model.get_samples_gmm(current_gate[:, :len(active_indices)], current_mean[:, :len(active_indices)],
                                          current_chol[:, :len(active_indices)], 10)  # (b,s,f)
    samples = ch.cat([basic_samples.transpose(0, 1), model_samples], dim=1)  # (b, s=s1+s2, f)
    log_model = model.log_prob_gmm(current_mean[:, :len(active_indices)], current_chol[:, :len(active_indices)],
                                   current_gate[:, :len(active_indices)], samples)  # (b,s)
    log_target = target.log_prob_tgt(contexts, samples)  # (b,s)

    # log density of new components = \log q(o_n|c) + \log q_{x_s}(x_s|o_n,c)
    log_new_o = MultivariateNormal(loc=samples, scale_tril=current_chol[:, idx].unsqueeze(1)).log_prob(samples)

    rewards = log_target - ch.max(log_model, new_component_gate + log_new_o)  # (b,s)

    chosen_mean_idx = ch.argmax(rewards)
    chosen_mean_batch_idx = chosen_mean_idx // samples.shape[1]
    chosen_mean_sample_idx = chosen_mean_idx % samples.shape[1]
    chosen_mean = samples[chosen_mean_batch_idx, chosen_mean_sample_idx]
    print("New component mean:", chosen_mean)

    # update the mean bias of the new component
    model.embedded_mean_bias[idx] = chosen_mean - current_mean[chosen_mean_batch_idx, idx]

    # update the cholesky bias of the new component
    new_component_chol = ch.eye(model.dim).to(device)
    model.embedded_chol_bias[idx] = new_component_chol - current_chol[:, idx].mean(dim=0)

    # update the gate of the new component
    model.gate.fc_gate.bias.data[idx] = new_component_gate
    model.gate.fc_gate.weight.data[idx] = ch.tensor(0, dtype=ch.float32).to(device)

    return chosen_mean


def adaptive_components(model, target, adaption_contexts, plot_contexts):
    model.eval()
    with ch.no_grad():
        delete_components(model, adaption_contexts)
        chosen_mean = add_components(model, target, adaption_contexts)

        plot(model, target, contexts=plot_contexts, plot_type="Adding",
             best_candidate=chosen_mean.clone().detach().to('cpu'))
        model.to(adaption_contexts.device)
    model.train()


def train_new_component(model, target, contexts, new_component_index, new_embedded_mean_bias, n_epochs=20):

    model.train()
    optimizer_add = optim.Adam([
        # {'params': model.gate.fc_gate.parameters(), 'lr': 0.01},
        {'params': model.gaussian_list.fc_mean.parameters(), 'lr': 0.01},
        {'params': model.gaussian_list.fc_chol.parameters(), 'lr': 0.01}
    ])
    # optimizer_add = optim.Adam(model.parameters(), lr=0.01)

    _, mean_init, _ = model(contexts)
    mean_init = mean_init.detach()
    init_bias = new_embedded_mean_bias.clone().detach()
    for epoch in range(n_epochs):
        # gradually decrease the embedded mean bias to 0
        n_steps = n_epochs // 10
        if (epoch + 1) % n_steps == 0:
            new_embedded_mean_bias = (1 - (epoch + 1) / n_epochs) * init_bias


        # _, mean, chol = model(contexts)
        # model_samples = model.get_rsamples(mean, chol, 10)
        # log_model = model.log_prob(mean, chol, model_samples)
        # log_target = target.log_prob_tgt(contexts, model_samples)
        # loss = ch.sum(log_model - log_target)
        #
        # loss = ch.mean(10*(mean[:,:new_component_index]-mean_init[:,:new_component_index]) **2)+ch.mean((mean[:,new_component_index]-mean_init[:,new_component_index]) **2)

        gate_pred, mean_pred, chol_pred = model(contexts)
        mean_pred[:, new_component_index] += new_embedded_mean_bias

        # idea 1: update only the new component, doesn't work
        mean_pred_new = mean_pred[:, new_component_index]
        chol_pred_new = chol_pred[:, new_component_index]
        model_samples = model.get_rsamples(mean_pred_new, chol_pred_new, 2)

        log_model = model.log_prob(mean_pred_new, chol_pred_new, model_samples)
        log_target = target.log_prob_tgt(contexts, model_samples)
        # loss = ch.sum(log_model - log_target) + ch.sum(5 * (mean_pred[:,:new_component_index]-mean_init[:,:new_component_index]) **2)

        aux_loss = model.auxiliary_reward(new_component_index, gate_pred.clone().detach(), mean_pred.clone().detach(),
                                      chol_pred.clone().detach(), model_samples)
        # loss = ch.sum(log_model - log_target - aux_loss) + ch.sum(5 * (mean_pred[:,:new_component_index]-mean_init[:,:new_component_index]) **2)
        gate_new = gate_pred[:, new_component_index].unsqueeze(1).expand(-1, model_samples.shape[1])
        # loss = ch.sum(ch.exp(gate_new) * (log_model - log_target - aux_loss + gate_new))
        loss = ch.sum(log_model - log_target - aux_loss + gate_new)

        # # idea 2: update same loss but detach the current component, doesn't work
        # loss_component = []
        # for j in range(model.active_components):
        #     mean_pred_j = mean_pred[:, j]  # (batched_c, 2)
        #     chol_pred_j = chol_pred[:, j]
        #
        #     # target and target log probability
        #     model_samples = model.get_rsamples(mean_pred_j, chol_pred_j, 10)
        #     log_model_j = model.log_prob(mean_pred_j, chol_pred_j, model_samples)
        #     log_target_j = target.log_prob_tgt(contexts, model_samples)
        #
        #     aux_loss = model.auxiliary_reward(j, gate_pred.clone().detach(), mean_pred.clone().detach(),
        #                                       chol_pred.clone().detach(), model_samples)
        #
        #     gate_pred_j = gate_pred[:, j].unsqueeze(1).expand(-1, model_samples.shape[1])
        #     approx_reward_j = log_model_j - log_target_j - aux_loss + gate_pred_j
        #     loss_j = ch.exp(gate_pred_j) * approx_reward_j
        #
        #     loss_component.append(loss_j)
        # loss = ch.stack(loss_component).sum() + ch.sum(5*(mean_pred[:,:new_component_index]-mean_init[:,:new_component_index]) **2)
        # # loss = ch.stack(loss_component).sum()

        optimizer_add.zero_grad()
        loss.backward()

        optimizer_add.step()


def get_optimizer(model, gate_lr, gaussian_lr):
    return optim.Adam([
        {'params': model.gate.parameters(), 'lr': gate_lr},
        {'params': model.gaussian_list.parameters(), 'lr': gaussian_lr}
    ])


def evaluate_model(model, target, eval_contexts, plot_contexts, epoch, n_epochs,
                   adaption, loss_history, history_size, stability_threshold, device):
    if len(loss_history) > history_size:
        loss_history.pop(0)

    if len(loss_history) == history_size:
        if (max(loss_history) - min(loss_history)) < stability_threshold:
            adaption = True
            if len(model.active_component_indices) < model.max_components:
                print(f"Stability reached at epoch {epoch}. Start adaption.")

            if (epoch + 1) % (n_epochs // 5)== 0:
                model.eval()
                with ch.no_grad():
                    # approx_reward = ch.cat(batched_approx_reward, dim=1)
                    # js_divergence_gates = js_divergence_gate(approx_reward, model, eval_contexts, device)
                    # kl_gate = kl_divergence_gate(approx_reward, model, eval_contexts, device)
                    js_div = js_divergence(model, target, eval_contexts, device)
                    j_div = jeffreys_divergence(model, target, eval_contexts, device)

                    plot(model, target, contexts=plot_contexts)
                    model.to(device)

                    wandb.log({
                        # "reverse KL between gates": kl_gate.item(),
                        # "Jensen Shannon Divergence between gates": js_divergence_gates.item(),
                        "Jensen Shannon Divergence": js_div.item(),
                        "Jeffreys Divergence": j_div.item()
                    })

                model.train()
    return loss_history, adaption


def train_model(model: ConditionalGMM or ConditionalGMM2 or ConditionalGMM3,
                target: AbstractTarget,
                n_epochs: int,
                batch_size: int,
                n_context: int,
                n_samples: int,
                gate_lr: float,
                gaussian_lr: float,
                device,
                project,
                eps_mean: float or None,
                eps_cov: float or None,
                alpha: int or None):

    optimizer = get_optimizer(model, gate_lr, gaussian_lr)
    contexts = target.get_contexts(n_context).to(device)
    eval_contexts = target.get_contexts(200).to(device)
    # bmm and gmm
    plot_contexts = ch.tensor([[-0.3],
                               [0.7],
                               [-1.8]])
    # funnel
    # plot_contexts = ch.tensor([[-0.3],
    #                            [0.1],
    #                            [-0.8]])
    train_size = int(n_context)
    loss_history = []
    history_size = 15
    stability_threshold = 30
    adaption = False

    for epoch in range(n_epochs):
        # plot initial model
        if epoch == 0:
            model.eval()
            with ch.no_grad():
                # funnel target has no gates
                # init_gate, _, _ = model(contexts)
                plot(model, target, contexts=plot_contexts)
                model.to(device)
            model.train()
            # wandb.log({"reverse KL between gates": kl_gate.item(),
            #            "Jensen Shannon Divergence between gates": js_divergence_gates.item()})

        # shuffle sampled contexts, since the same sample set is used.
        indices = ch.randperm(train_size)
        shuffled_contexts = contexts[indices]

        # add and delete components
        if adaption and len(model.active_component_indices) < model.max_components:
            adaptive_components(model, target, shuffled_contexts[0:batch_size], plot_contexts)
            adaption = False

        # training loop
        batch_loss = []
        if project:
            batched_approx_reward = []
            init_contexts = shuffled_contexts[0:batch_size]
            b_gate_old, b_mean_old, b_chol_old = model(init_contexts)

            for batch_idx in range(0, train_size, batch_size):
                # get old distribution for current batch
                b_contexts = shuffled_contexts[batch_idx:batch_idx+batch_size]

                # prediction step
                gate_pred, mean_pred, chol_pred = model(b_contexts)

                # projection step
                mean_proj = []
                chol_proj = []
                for j in range(model.active_components):
                    mean_proj_j, chol_proj_j = split_kl_projection(mean_pred[:, j] , chol_pred[:, j],
                                                                   b_mean_old[:, j].clone().detach(),
                                                                   b_chol_old[:, j].clone().detach(),
                                                                   eps_mean, eps_cov)
                    mean_proj.append(mean_proj_j)
                    chol_proj.append(chol_proj_j)
                mean_proj = ch.stack(mean_proj, dim=1)
                chol_proj = ch.stack(chol_proj, dim=1)

                # component-wise calculation
                loss_component = []
                approx_reward_component = []
                active_components = len(model.active_component_indices)
                for j in range(active_components):
                    mean_pred_j = mean_pred[:, j]  # (batched_c, 2)
                    chol_pred_j = chol_pred[:, j]
                    mean_proj_j = mean_proj[:, j]
                    chol_proj_j = chol_proj[:, j]

                    # target and target log probability
                    model_samples = model.get_rsamples(mean_proj_j, chol_proj_j, n_samples)
                    log_model_j = model.log_prob(mean_proj_j, chol_proj_j, model_samples)
                    log_target_j = target.log_prob_tgt(b_contexts, model_samples)

                    # regression step
                    pred_dist = MultivariateNormal(mean_pred_j, scale_tril=chol_pred_j)
                    proj_dist = MultivariateNormal(mean_proj_j.clone().detach(), scale_tril=chol_proj_j.clone().detach())
                    reg_loss = kl_divergence(proj_dist, pred_dist).unsqueeze(1).expand(-1, model_samples.shape[1])

                    aux_loss = model.auxiliary_reward(j, gate_pred.clone().detach(), mean_proj.clone().detach(),
                                                      chol_proj.clone().detach(), model_samples)

                    gate_pred_j = gate_pred[:, j].unsqueeze(1).expand(-1, model_samples.shape[1])
                    approx_reward_j = log_model_j - log_target_j - aux_loss + alpha * reg_loss + gate_pred_j
                    loss_j = ch.exp(gate_pred_j) * approx_reward_j

                    loss_component.append(loss_j)
                    approx_reward_component.append(approx_reward_j)
                if batch_idx + batch_size < train_size:
                    b_next_context = shuffled_contexts[batch_idx + batch_size:batch_idx + 2 * batch_size]
                    b_gate_old, b_mean_old, b_chol_old = model(b_next_context)

                batched_approx_reward.append(ch.stack(approx_reward_component))
                loss = ch.sum(ch.stack(loss_component))
                batch_loss.append(loss.clone().detach().item())

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                wandb.log({"train_loss": loss.item()})
        else:
            # with log responsibility but without projection
            batched_approx_reward = []
            for batch_idx in range(0, train_size, batch_size):
                # get old distribution for current batch
                b_contexts = shuffled_contexts[batch_idx:batch_idx + batch_size]

                # prediction step
                gate_pred, mean_pred, chol_pred = model(b_contexts)

                # component-wise calculation
                loss_component = []
                approx_reward_component = []
                active_components = len(model.active_component_indices)
                for j in range(active_components):
                    mean_pred_j = mean_pred[:, j]
                    chol_pred_j = chol_pred[:, j]

                    model_samples = model.get_rsamples(mean_pred_j, chol_pred_j, n_samples)
                    log_model_j = model.log_prob(mean_pred_j, chol_pred_j, model_samples)
                    log_target_j = target.log_prob_tgt(b_contexts, model_samples)

                    aux_loss = model.auxiliary_reward(j, gate_pred.clone().detach(), mean_pred.clone().detach(),
                                                      chol_pred.clone().detach(), model_samples)
                    gate_pred_j = gate_pred[:, j].unsqueeze(1).expand(-1, model_samples.shape[1])
                    approx_reward_j = log_model_j - log_target_j - aux_loss + gate_pred_j
                    loss_j = ch.exp(gate_pred_j) * approx_reward_j

                    loss_component.append(loss_j)
                    approx_reward_component.append(approx_reward_j)

                batched_approx_reward.append(ch.stack(approx_reward_component))
                loss = ch.sum(ch.stack(loss_component))
                batch_loss.append(loss.clone().detach().item())
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                wandb.log({"train_loss": loss.item()})

        # Evaluation
        a = sum(batch_loss) / len(batch_loss)
        loss_history.append(a)
        loss_history, adaption = evaluate_model(model, target, eval_contexts, plot_contexts, epoch, n_epochs,
                                                adaption, loss_history, history_size, stability_threshold, device)

        # # trick from VIPS++ paper
        # if project:
        #     current_loss = loss.item()
        #     if current_loss < prev_loss:
        #         eps_mean *= 0.8
        #         eps_cov *= 0.8
        #     else:
        #         eps_mean *= 1.1
        #         eps_cov *= 1.1
        #     prev_loss = current_loss

    print("Training done!")


def plot(model: ConditionalGMM,
         target: AbstractTarget,
         contexts=None,
         plot_type="Evaluation",
         best_candidate=None):
    if contexts is None:
        contexts = target.get_contexts(3).to('cpu')
    else:
        contexts = contexts.clone().detach().to('cpu')
    # plot2d_matplotlib(target, model.to('cpu'), contexts, min_x=-6.5, max_x=6.5, min_y=-6.5, max_y=6.5)
    plot2d_matplotlib(target, model.to('cpu'), contexts, plot_type=plot_type, best_candidate=best_candidate,
                      min_x=-15, max_x=15, min_y=-15, max_y=15)


def toy_task(config):
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    n_context = config['n_context']
    max_components = config['max_components']
    num_gate_layer = config['num_gate_layer']
    num_component_layer = config['num_component_layer']
    n_samples = config['n_samples']
    gate_lr = config['gate_lr']
    gaussian_lr = config['gaussian_lr']

    model_name = config['model_name']
    dim = config['dim']
    initialization_type = config['initialization_type']

    target_name = config['target_name']
    target_components = config['target_components']

    project = config['project']
    eps_mean = config['eps_mean']
    eps_cov = config['eps_cov']
    alpha = config['alpha']

    # Device
    device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Target
    target = get_target(target_name, target_components=target_components).to(device)

    # Model
    model = get_model(model_name,
                      target_name,
                      dim,
                      device,
                      max_components,
                      num_gate_layer,
                      num_component_layer,
                      initialization_type)

    # Training
    train_model(model, target,
                n_epochs, batch_size, n_context, n_samples, gate_lr, gaussian_lr, device,
                project, eps_mean, eps_cov, alpha)


if __name__ == "__main__":
    # test
    # funnel_config = {
    #     "n_epochs": 500,
    #     "batch_size": 128,
    #     "n_context": 1280,
    #     "max_components": 8,
    #     "num_gate_layer": 5,
    #     "num_component_layer": 7,
    #     "n_samples": 10,
    #     "gate_lr": 0.001,
    #     "gaussian_lr": 0.001,
    #     "model_name": "toy_task_model_3",
    #     "target_name": "funnel",
    #     "target_components": 10,
    #     "dim": 10,
    #     "initialization_type": "xavier",
    #     "project": False,
    #     "eps_mean": 1e-6,
    #     "eps_cov": 1e-6,
    #     "alpha": 50
    # }
    gmm_config = {
        "n_epochs": 500,
        "batch_size": 64,
        "n_context": 640,
        "max_components": 2,
        "num_gate_layer": 3,
        "num_component_layer": 5,
        "n_samples": 5,
        "gate_lr": 0.001,
        "gaussian_lr": 0.01,
        "model_name": "toy_task_model_3",
        "target_name": "gmm",
        "target_components": 3,
        "dim": 2,
        "initialization_type": "xavier",
        "project": False,
        "eps_mean": 0.5,
        "eps_cov": 0.01,
        "alpha": 50
    }
    group_name = "test"
    wandb.init(project="ELBO", group=group_name, config=gmm_config)
    toy_task(gmm_config)
