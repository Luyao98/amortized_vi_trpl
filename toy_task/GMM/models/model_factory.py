import numpy as np
from toy_task.GMM.models.GMM_model import ConditionalGMM
from toy_task.GMM.utils.network_utils import initialize_weights


def get_model(model_name, dim, device, fc_layer_size, n_components, initialization_type):
    if model_name == "toy_task_2d_model":
        init_bias_mean_list = generate_init_biases(n_components)
        init_bias_gate = [0.0] * n_components

        model = ConditionalGMM(fc_layer_size, n_components, dim=dim,
                               init_bias_gate=init_bias_gate,
                               init_bias_mean_list=init_bias_mean_list)
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean', 'fc_gate'])
    else:
        raise ValueError(f"Invalid model name {model_name}. Now only support 'toy_task_2d_model'.")

    return model.to(device)


def generate_init_biases(n_components, scale=3.0):
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    init_bias_mean_list = [[scale * np.cos(angle), scale * np.sin(angle)] for angle in angles]
    return init_bias_mean_list
