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

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


def add_components(model, target, contexts):
    model.eval()
    with ch.no_grad():
        idx = model.active_components
        model.active_components += 1

        device = contexts.device
        current_gate, current_mean, current_chol = model(contexts)
        # new_component_gate = ch.log(ch.tensor(0.001)).to(device)
        init_gates = ch.tensor([50, 25, 10, 5])
        random_index = ch.randint(0, len(init_gates), (1,)).item()
        new_component_gate = -init_gates[random_index].to(device)

        samples = model.get_samples_gmm(current_gate[:,:idx], current_mean[:,:idx], current_chol[:,:idx], 20)  # (b,s,f)
        log_model = model.log_prob_gmm(current_mean[:,:idx], current_chol[:,:idx], current_gate[:,:idx], samples)  # (b,s)
        log_target = target.log_prob_tgt(contexts, samples)  # (b,s)

        # log density of new components = \log q(o_n|c) + \log q_{x_s}(x_s|o_n,c)
        # log_new_o = MultivariateNormal(loc=samples, covariance_matrix=ch.eye(samples.shape[-1]).to(device)).log_prob(samples)
        log_new_o = MultivariateNormal(loc=samples, scale_tril=current_chol[:,idx].unsqueeze(1)).log_prob(samples)

        rewards = log_target - ch.max(log_model, new_component_gate + log_new_o)  # (b,s)
        # rewards = log_target - log_model
        # chosen_mean_idx = ch.argmax(rewards)
        # chosen_mean_batch_idx = chosen_mean_idx // samples.shape[1]
        # chosen_mean_sample_idx = chosen_mean_idx % samples.shape[1]
        # chosen_mean = samples[chosen_mean_batch_idx, chosen_mean_sample_idx]

        # DBSCAN
        _, data_idx = ch.max(rewards, dim=1)
        max_idx = data_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, current_mean.shape[-1])

        data = samples.gather(1, max_idx).squeeze(1)
        data_np = data.cpu().numpy()


        # eps = ch.mean(current_mean[:,:idx]).cpu().item()
        # dbscan = DBSCAN(eps=eps, min_samples=2).fit(data_np)
        #
        # labels = dbscan.labels_
        # unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        # max_label = unique_labels[np.argmax(counts)]
        #
        # max_class_data = data[labels == max_label]
        # # max_class_indices = ch.nonzero(ch.tensor(labels) == max_label).squeeze()

        kmeans = KMeans(n_clusters=4, random_state=0, n_init=10).fit(data_np)
        labels = kmeans.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_label = unique_labels[np.argmin(counts)]

        max_class_data = data[labels == max_label]
        chosen_mean = max_class_data.mean(dim=0)


        model.gate.fc_gate.bias.data[idx] = new_component_gate
        model.gaussian_list.fc_mean.bias.data[idx * model.dim: (idx + 1) * model.dim] = chosen_mean
        print("New component mean:", model.gaussian_list.fc_mean.bias.data[idx * model.dim: (idx + 1) * model.dim])
        # with ch.no_grad():
        #     model.gaussian_list.fc_chol.bias.data[idx * model.dim * (model.dim + 1) // 2: (idx + 1) * model.dim * (model.dim + 1) // 2] = ch.tensor([1, 0, 1])
        #     print("New component chol:", model.gaussian_list.fc_chol.bias.data[idx * model.dim * (model.dim + 1) // 2: (idx + 1) * model.dim * (model.dim + 1) // 2])
        #     model.gaussian_list.fc_mean.weight.data[idx * model.dim: (idx + 1) * model.dim].zero_()
        #     model.gaussian_list.fc_chol.weight.data[idx * model.dim * (model.dim + 1) // 2: (idx + 1) * model.dim * (model.dim + 1) // 2].zero_()
        #     model.gaussian_list.fc_chol.bias.data[idx * model.dim * (model.dim + 1) // 2: (idx + 1) * model.dim * (model.dim + 1) // 2].zero_()
        #     print("New component mean weights:", model.gaussian_list.fc_mean.weight.data[idx * model.dim: (idx + 1) * model.dim])
        #     print("New component chol weights:", model.gaussian_list.fc_chol.weight.data[idx * model.dim * (model.dim + 1) // 2: (idx + 1) * model.dim * (model.dim + 1) // 2])

    # model.register_hooks()
    return idx
def train_new_component(model, target, contexts, new_component_index, n_epochs=50):
    model.train()
    optimizer = optim.Adam([
        {'params': model.gate.fc_gate.parameters(), 'lr': 0.01},
        {'params': model.gaussian_list.fc_mean.parameters(), 'lr': 0.01},
        {'params': model.gaussian_list.fc_chol.parameters(), 'lr': 0.01}], weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        gate_pred, mean_pred, chol_pred = model(contexts)


        # # idea 1: update only the new component, doesn't work
        mean_pred_new = mean_pred[:, new_component_index]
        chol_pred_new = chol_pred[:, new_component_index]
        # model_samples = model.get_rsamples(mean_pred_new, chol_pred_new, 1)
        model_samples = mean_pred_new.unsqueeze(1)
        log_model = model.log_prob(mean_pred_new, chol_pred_new, model_samples)
        log_target = target.log_prob_tgt(contexts, model_samples)

        aux_loss = model.auxiliary_reward(new_component_index, gate_pred.clone().detach(), mean_pred.clone().detach(),
                                      chol_pred.clone().detach(), model_samples)
        loss = ch.sum(log_model - log_target - aux_loss)

        # idea 2: update same loss but detach the current component, doesn't work
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
        # loss = ch.stack(loss_component).sum()

        loss.backward()
        optimizer.step()
    # model.clear_hooks()

def train_model(model: ConditionalGMM or ConditionalGMM2 or ConditionalGMM3,
                target: AbstractTarget,
                n_epochs: int,
                batch_size: int,
                n_context: int,
                n_components: int,
                n_samples: int,
                gate_lr: float,
                gaussian_lr: float,
                device,
                project,
                eps_mean: float or None,
                eps_cov: float or None,
                alpha: int or None):

    # optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-5)
    optimizer = optim.Adam([
        {'params': model.gate.parameters(), 'lr': gate_lr},
        {'params': model.gaussian_list.parameters(), 'lr': gaussian_lr}
    ], weight_decay=1e-5)
    contexts = target.get_contexts(n_context).to(device)
    eval_contexts = target.get_contexts(200).to(device)
    # bmm and gmm
    # plot_contexts = ch.tensor([[-0.3],
    #                            [0.7],
    #                            [-1.8]])
    # funnel
    plot_contexts = ch.tensor([[-0.3],
                               [0.1],
                               [-0.8]])
    train_size = int(n_context)
    prev_loss = float('inf')

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

        # adding new components
        # model.eval()
        # with ch.no_grad():
        n_adds = n_epochs // 10
        if (epoch + 1) % n_adds == 0 and model.active_components < n_components:
            new_component_index = add_components(model, target, shuffled_contexts[0:batch_size])
            # train_new_component(model, target, shuffled_contexts[0:batch_size], new_component_index)
            model.eval()
            with ch.no_grad():
                plot(model, target, contexts=plot_contexts)
                model.to(device)
            model.train()

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
                for j in range(model.active_components):
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
                for j in range(model.active_components):
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
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                wandb.log({"train_loss": loss.item()})

        # Evaluation
        n_plot = n_epochs // 10
        if (epoch + 1) % n_plot == 0:
            model.eval()
            with ch.no_grad():
                approx_reward = ch.cat(batched_approx_reward, dim=1)  # [n_components, n_contexts, n_samples]
                js_divergence_gates = js_divergence_gate(approx_reward, model, shuffled_contexts, device)
                kl_gate = kl_divergence_gate(approx_reward, model, shuffled_contexts, device)
                js_div = js_divergence(model, target, eval_contexts, device)
                j_div = jeffreys_divergence(model, target, eval_contexts, device)

                plot(model, target, contexts=plot_contexts)
                model.to(device)

                # trick from VIPS++ paper
                if project:
                    current_loss = loss.item()
                    if current_loss < prev_loss:
                        eps_mean *= 0.8
                        eps_cov *= 0.8
                    else:
                        eps_mean *= 1.1
                        eps_cov *= 1.1
                    prev_loss = current_loss
            model.train()
            wandb.log({"reverse KL between gates": kl_gate.item(),
                       "Jensen Shannon Divergence between gates": js_divergence_gates.item(),
                       "Jensen Shannon Divergence": js_div.item(),
                       "Jeffreys Divergence": j_div.item()})

    print("Training done!")


def plot(model: ConditionalGMM,
         target: AbstractTarget,
         contexts=None):
    if contexts is None:
        contexts = target.get_contexts(3).to('cpu')
    else:
        contexts = contexts.clone().to('cpu')

    plot2d_matplotlib(target, model.to('cpu'), contexts, min_x=-6.5, max_x=6.5, min_y=-6.5, max_y=6.5)
    # plot2d_matplotlib(target, model.to('cpu'), contexts, min_x=-10, max_x=10, min_y=-10, max_y=10)


def toy_task(config):
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    n_context = config['n_context']
    n_components = config['n_components']
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
                      n_components,
                      num_gate_layer,
                      num_component_layer,
                      initialization_type)

    # Training
    train_model(model, target,
                n_epochs, batch_size, n_context, n_components, n_samples, gate_lr, gaussian_lr, device,
                project, eps_mean, eps_cov, alpha)


if __name__ == "__main__":
    # test
    # config = {
    #     "n_epochs": 500,
    #     "batch_size": 128,
    #     "n_context": 1280,
    #     "n_components": 8,
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
    config = {
        "n_epochs": 800,
        "batch_size": 64,
        "n_context": 640,
        "n_components": 4,
        "num_gate_layer": 2,
        "num_component_layer": 4,
        "n_samples": 5,
        "gate_lr": 0.01,
        "gaussian_lr": 0.01,
        "model_name": "toy_task_model_2",
        "target_name": "gmm",
        "target_components": 4,
        "dim": 2,
        "initialization_type": "xavier",
        "project": False,
        "eps_mean": 0.5,
        "eps_cov": 0.01,
        "alpha": 50
    }
    group_name = "test"
    wandb.init(project="ELBO", group=group_name, config=config)
    toy_task(config)
