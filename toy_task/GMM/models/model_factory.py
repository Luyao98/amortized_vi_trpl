import numpy as np
from toy_task.GMM.models.GMM_model import ConditionalGMM
from toy_task.GMM.models.GMM_model_2 import ConditionalGMM2
from toy_task.GMM.models.GMM_model_3 import ConditionalGMM3
from toy_task.GMM.utils.network_utils import initialize_weights, generate_init_biases


def get_model(model_name,target_name, dim, device, max_components, gate_layer, com_layer, initialization_type):
    if model_name == "toy_task_model_1":
        if target_name == "gmm":
            scale = 5.0
        else:
            scale = 3.0
        num_layers_gate = gate_layer
        gate_size = 128
        num_layers_gaussian = com_layer
        gaussian_size = 256
        init_bias_mean_list = generate_init_biases(max_components, dim, scale)
        init_bias_gate = [1.0] * max_components

        model = ConditionalGMM(num_layers_gate,
                               gate_size,
                               num_layers_gaussian,
                               gaussian_size,
                               max_components,
                               dim=dim,
                               init_bias_gate=init_bias_gate,
                               init_bias_mean_list=init_bias_mean_list)
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean', 'fc_gate'])

    elif model_name == "toy_task_model_2":
        if target_name == "gmm":
            scale = 5.0
        else:
            scale = 3.0

        init_bias_mean_list = generate_init_biases(max_components, dim, scale)
        init_bias_mean_array = np.array(init_bias_mean_list).flatten()
        init_bias_gate = [1.0] * max_components
        model = ConditionalGMM2(num_layers_gate=gate_layer,
                                gate_size=128,
                                num_layers_gaussian=com_layer,
                                gaussian_size=256,
                                max_components=max_components,
                                active_components=1,
                                dim=dim,
                                init_bias_gate=init_bias_gate,
                                init_bias_mean=init_bias_mean_array)
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean', 'fc_gate'])

    elif model_name == "toy_task_model_3":
        model = ConditionalGMM3(num_layers_gate=gate_layer,
                                gate_size=128,
                                num_layers_gaussian=com_layer,
                                gaussian_size=256,
                                max_components=max_components,
                                init_components=30,
                                dim=dim,
                                context_dim=1,
                                init_bias_gate=None,
                                init_bias_mean=[0.0] * dim,
                                init_std=3)
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean', 'embedded'])
    else:
        raise ValueError(f"Invalid model name {model_name}. Choose one from 'toy_task_model_1', 'toy_task_model_2', toy_task_model_3, toy_task_model_4.")

    return model.to(device)


# test
# model = get_model("toy_task_model_3", "gmm", 2, 'cuda', 4,2, 5, "xavier")
# print(model.dim)