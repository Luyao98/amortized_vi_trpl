import numpy as np
from toy_task.GMM.models.GMM_model import ConditionalGMM
from toy_task.GMM.models.GMM_model_2 import ConditionalGMM2
from toy_task.GMM.models.GMM_model_3 import EmbeddedConditionalGMM
from toy_task.GMM.utils.network_utils import initialize_weights, generate_init_biases


def get_model(config, device):

    model_name = config["model_name"]
    dim = config["dim"]
    context_dim = config["context_dim"]
    random_init = config["random_init"]
    init_scale = config["init_scale"]
    init_std = config["init_std"]
    max_components = config["max_components"]
    init_components = config["init_components"]
    num_gate_layer = config["num_gate_layer"]
    num_component_layer = config["num_component_layer"]
    num_neuron_gate_layer = config["num_neuron_gate_layer"]
    num_neuron_component_layer = config["num_neuron_component_layer"]
    initialization_type = config["initialization_type"]

    if model_name == "toy_task_model_1":
        init_bias_mean_list = generate_init_biases(max_components, dim, 3)
        init_bias_gate = [1.0] * max_components

        model = ConditionalGMM(num_gate_layer,
                               num_neuron_gate_layer,
                               num_component_layer,
                               num_neuron_component_layer,
                               max_components,
                               dim=dim,
                               init_bias_gate=init_bias_gate,
                               init_bias_mean_list=init_bias_mean_list)
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean', 'fc_gate'])

    elif model_name == "toy_task_model_2":

        init_bias_mean_list = generate_init_biases(max_components, dim, 3)
        init_bias_mean_array = np.array(init_bias_mean_list).flatten()
        init_bias_gate = [1.0] * max_components
        model = ConditionalGMM2(num_gate_layer=num_gate_layer,
                                num_neuron_gate_layer=num_neuron_gate_layer,
                                num_component_layer=num_component_layer,
                                num_neuron_component_layer=num_neuron_component_layer,
                                max_components=max_components,
                                active_components=1,
                                dim=dim,
                                init_bias_gate=init_bias_gate,
                                init_bias_mean=init_bias_mean_array)
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean', 'fc_gate'])

    elif model_name == "embedded_model":
        model = EmbeddedConditionalGMM(num_gate_layer=num_gate_layer,
                                num_neuron_gate_layer=num_neuron_gate_layer,
                                num_component_layer=num_component_layer,
                                num_neuron_component_layer=num_neuron_component_layer,
                                max_components=max_components,
                                init_components=init_components,
                                dim=dim,
                                context_dim=context_dim,
                                random_init=random_init,
                                init_scale=init_scale,
                                init_bias_gate=None,
                                init_bias_mean=[0.0] * dim,
                                init_std=init_std)
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean', 'embedded'])
    else:
        raise ValueError(f"Invalid model name {model_name}. Choose one from 'toy_task_model_1', 'toy_task_model_2', toy_task_model_3, toy_task_model_4.")

    return model.to(device)


# test
# model = get_model("toy_task_model_3", "gmm", 2, 'cuda', 4,2, 5, "xavier")
# print(model.dim)