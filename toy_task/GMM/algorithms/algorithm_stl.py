import torch as ch
import torch.nn as nn
import torch.optim as optim
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

from toy_task.GMM.utils.network_utils import set_seed
from toy_task.GMM.utils.torch_utils import gumbel_softmax


def train_model_3(model: ConditionalGMM or ConditionalGMM2 or ConditionalGMM3,
                  target: AbstractTarget,
                  n_epochs: int,
                  batch_size: int,
                  n_context: int,
                  n_components: int,
                  n_samples: int,
                  gate_lr: float,
                  gaussian_lr: float,
                  device):

    optimizer = optim.Adam([
        {'params': model.gate.parameters(), 'lr': gate_lr},
        {'params': model.gaussian_list.parameters(), 'lr': gaussian_lr}
    ], weight_decay=1e-5)
    contexts = target.get_contexts(n_context).to(device)
    eval_contexts = target.get_contexts(200).to(device)
    # # bmm and gmm
    # plot_contexts = ch.tensor([[-0.3],
    #                            [0.7],
    #                            [-1.8]])
    # funnel
    plot_contexts = ch.tensor([[-0.3],
                               [0.1],
                               [-0.8]])
    train_size = int(n_context)

    for epoch in range(n_epochs):
        # plot initial model
        if epoch == 0:
            model.eval()
            with ch.no_grad():
                plot(model, target, contexts=plot_contexts)
                model.to(device)
            model.train()

        # shuffle sampled contexts, since I use the same sample set
        indices = ch.randperm(train_size)
        shuffled_contexts = contexts[indices]
        batched_approx_reward = []

        for batch_idx in range(0, train_size, batch_size):
            # prediction step
            b_contexts = shuffled_contexts[batch_idx:batch_idx+batch_size]
            gate_pred, mean_pred, chol_pred = model(b_contexts)
            # # reparameterization trick for gate
            # gates = gumbel_softmax(ch.exp(gate_pred), temperature=1, hard=False)
            # gate_pred = ch.log(gates)

            # component-wise calculation
            loss_component = []
            approx_reward_component = []
            for j in range(n_components):
                mean_pred_j = mean_pred[:, j]  # (batched_contexts, 2)
                chol_pred_j = chol_pred[:, j]

                model_samples = model.get_rsamples(mean_pred_j, chol_pred_j, n_samples)
                log_target_j = target.log_prob_tgt(b_contexts, model_samples)

                # sticking the landing
                log_model_j = model.log_prob_gmm(mean_pred.clone().detach(), chol_pred.clone().detach(), gate_pred.clone().detach(), model_samples)

                gate_pred_j = gate_pred[:, j].unsqueeze(1).expand(-1, model_samples.shape[1])

                # elbo for j-th component
                approx_reward_j = log_model_j - log_target_j
                loss_j = ch.exp(gate_pred_j) * approx_reward_j
                loss_component.append(loss_j)
                approx_reward_component.append(approx_reward_j)

            batched_approx_reward.append(ch.stack(approx_reward_component))

            # # test
            # model.eval()
            # with ch.no_grad():
            #     approx_reward = ch.stack(approx_reward_component)  # [n_components, n_contexts, n_samples]
            #     js_divergence_gates = js_divergence_gate(approx_reward, model, b_contexts, device)
            #     kl_gate = kl_divergence_gate(approx_reward, model, b_contexts, device)
            #     js_div = js_divergence(model, target, eval_contexts, device)
            #     j_div = jeffreys_divergence(model, target, eval_contexts, device)
            #
            #     plot(model, target, contexts=plot_contexts)
            #     model.to(device)
            #
            # model.train()
            # wandb.log({"reverse KL between gates": kl_gate.item(),
            #           "Jensen Shannon Divergence between gates": js_divergence_gates.item(),
            #            "Jensen Shannon Divergence": js_div.item(),
            #            "Jeffreys Divergence": j_div.item()})
            # # test end
            loss = ch.sum(ch.stack(loss_component))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            wandb.log({"train_loss": loss.item()})

        # Evaluation
        n_plot = n_epochs // 20
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


def toy_task_3(config):
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
    train_model_3(model, target, n_epochs, batch_size, n_context, n_components, n_samples, gate_lr, gaussian_lr, device)

# # test
# if __name__ == "__main__":
#
#     set_seed(1002)
#     config = {
#         "n_epochs": 400,
#         "batch_size": 64,
#         "n_context": 640,
#         "n_components": 4,
#         "num_gate_layer": 2,
#         "num_component_layer": 4,
#         "n_samples": 10,
#         "gate_lr": 0.0001,
#         "gaussian_lr": 0.0001,
#         "model_name": "toy_task_model_2",
#         "target_name": "gmm",
#         "target_components": 4,
#         "dim": 2,
#         "initialization_type": "xavier"
#     }
#
#     group_name = "test"
#     wandb.init(project="ELBO", group=group_name, config=config)
#     toy_task_3(config)