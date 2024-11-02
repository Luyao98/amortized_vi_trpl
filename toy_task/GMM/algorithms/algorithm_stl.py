import torch as ch
import torch.nn as nn

from toy_task.GMM.models.GMM_model_3 import EmbeddedConditionalGMM
from toy_task.GMM.targets.abstract_target import AbstractTarget
from toy_task.GMM.models.model_factory import get_model
from toy_task.GMM.targets.target_factory import get_target

from toy_task.GMM.algorithms.algorithm import (init_some_components, adaptive_components, get_all_contexts,
                                               get_optimizer, evaluate_model)
import wandb


def step_stl(model: EmbeddedConditionalGMM,
             target: AbstractTarget,
             shuffled_contexts,
             training_config,
             optimizer
             ):

    train_size = training_config["n_context"]
    batch_size = training_config["batch_size"]

    eva_loss = ch.zeros(1, device=shuffled_contexts.device)
    total_time = 0  # Accumulate total time for all batches
    num_batches = 0  # Track the number of batches
    for batch_idx in range(0, train_size, batch_size):
        b_contexts = shuffled_contexts[batch_idx:batch_idx + batch_size]

        # Start timing
        start_event = ch.cuda.Event(enable_timing=True)
        end_event = ch.cuda.Event(enable_timing=True)
        start_event.record()

        # Prediction
        gate_pred, mean_pred, chol_pred = model(b_contexts)

        # End timing
        end_event.record()
        ch.cuda.synchronize()
        batch_time = start_event.elapsed_time(end_event)
        total_time += batch_time
        num_batches += 1

        # Compute ELBO loss
        loss = compute_elbo_loss_stl(model, target, b_contexts, mean_pred, chol_pred, gate_pred,
                                     training_config["n_samples"])

        # Update model
        # eva_loss += get_numpy(loss)  # reduce cpu consumption
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        wandb.log({"negative ELBO": loss.detach()})

    # Shuffle sampled contexts
    indices = ch.randperm(train_size)
    shuffled_contexts = shuffled_contexts[indices]
    avg_time = total_time / num_batches if num_batches > 0 else 0
    return eva_loss, shuffled_contexts, avg_time


def compute_elbo_loss_stl(model: EmbeddedConditionalGMM,
                          target: AbstractTarget,
                          contexts,
                          mean,
                          chol,
                          gate,
                          n_samples
                          ):
    model_samples = model.get_rsamples(mean, chol, n_samples)  # (S, C, O, dz)
    _, _, n_components, _ = model_samples.shape
    log_model = [
        model.log_prob_gmm(mean.clone().detach(), chol.clone().detach(), gate.clone().detach(), model_samples[:, :, o])
        for o in range(n_components)]
    log_model = ch.stack(log_model, dim=-1)  # (S, C, O)
    log_target = target.log_prob_tgt(contexts, model_samples)  # (S, C, O)

    elbo = ch.exp(gate.squeeze(0)) * (log_model - log_target)
    loss = elbo.sum(dim=-1).mean()
    return loss


def toy_task_stl(config):

    model_config = config["model_config"]
    target_config = config["target_config"]
    training_config = config["training_config"]

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

    adaption_config = training_config["adaption"]
    adapt = adaption_config["adapt"]
    history_size = adaption_config["history_size"]

    optimizer = get_optimizer(model, training_config["gate_lr"], training_config["gaussian_lr"])
    contexts, eval_contexts, plot_contexts = get_all_contexts(target, training_config["n_context"] , device)
    loss_history = []
    adaption = False
    infer_time = 0

    # Initialize and plot the model
    init_some_components(model, target, contexts, plot_contexts, device, adaption_config["scale"],
                         adaption_config["lr"], adaption_config["itr"])

    for epoch in range(n_epochs):
        # Add and delete components if indicator is True
        if adapt and adaption:
            adaptive_components(model, target, contexts[0:batch_size], plot_contexts, device, adaption_config)
            adaption = False

        # Perform training step
        eva_loss, contexts, inference_time = step_stl(model, target, contexts, training_config, optimizer)

        # Evaluate model and update loss history
        loss_history.append(eva_loss)
        loss_history, history_size, adaption = evaluate_model(model, target, eval_contexts, plot_contexts, epoch,
                                                              n_epochs, adapt, adaption, loss_history, history_size,
                                                              device)
        infer_time += inference_time
    wandb.log({"inference_time/ms": float(infer_time/n_epochs)})


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
    toy_task_stl(gmm_config)