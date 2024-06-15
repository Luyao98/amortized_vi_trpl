import numpy as np
from toy_task.GMM.models.GMM_model import ConditionalGMM
from toy_task.GMM.models.GMM_model_2 import ConditionalGMM2
from toy_task.GMM.models.GMM_model_3 import ConditionalGMM3
from toy_task.GMM.utils.network_utils import initialize_weights


def get_model(model_name,target_name, dim, device, n_components, gate_layer, com_layer, initialization_type):
    if model_name == "toy_task_model_1":
        if target_name == "gmm":
            scale = 5.0
        else:
            scale = 3.0
        num_layers_gate = gate_layer
        gate_size = 128
        num_layers_gaussian = com_layer
        gaussian_size = 256
        init_bias_mean_list = generate_init_biases(n_components, dim, scale)
        init_bias_gate = [0.0] * n_components

        model = ConditionalGMM(num_layers_gate,
                               gate_size,
                               num_layers_gaussian,
                               gaussian_size,
                               n_components,
                               dim=int(dim),
                               init_bias_gate=init_bias_gate,
                               init_bias_mean_list=init_bias_mean_list)
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean', 'fc_gate'])
    elif model_name == "toy_task_model_2":
        if target_name == "gmm":
            scale = 5.0
        else:
            scale = 3.0
        num_layers_gate = gate_layer
        gate_size = 128
        num_layers_gaussian = com_layer
        gaussian_size = 256
        init_bias_mean_list = generate_init_biases(n_components, dim, scale)
        init_bias_mean_array = np.array(init_bias_mean_list).flatten()
        init_bias_gate = [0.0] * n_components
        model = ConditionalGMM2(num_layers_gate=num_layers_gate,
                                gate_size=gate_size,
                                num_layers_gaussian=num_layers_gaussian,
                                gaussian_size=gaussian_size,
                                n_components=n_components,
                                dim=int(dim),
                                init_bias_gate=init_bias_gate,
                                init_bias_mean=init_bias_mean_array)
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean', 'fc_gate'])
    elif model_name == "toy_task_model_3":
        if target_name == "gmm":
            scale = 5.0
        else:
            scale = 3.0
        num_layers_gate = gate_layer
        gate_size = 128
        num_layers_gaussian = com_layer
        gaussian_size = 256
        init_bias_mean_list = generate_init_biases(n_components, dim, scale)
        init_bias_mean_array = np.array(init_bias_mean_list).flatten()
        init_bias_gate = [0.0] * n_components
        model = ConditionalGMM3(num_layers_gate=num_layers_gate,
                                gate_size=gate_size,
                                num_layers_gaussian=num_layers_gaussian,
                                gaussian_size=gaussian_size,
                                n_components=n_components,
                                dim=int(dim),
                                init_bias_gate=init_bias_gate,
                                init_bias_mean=init_bias_mean_array)
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean', 'fc_gate'])
    else:
        raise ValueError(f"Invalid model name {model_name}. Choose one from 'toy_task_model_1', 'toy_task_model_2', toy_task_model_3, toy_task_model_4.")

    return model.to(device)


def generate_init_biases(n_components, dim, scale):
    if dim == 2:
        angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
        init_bias_mean_list = [[scale * np.cos(angle), scale * np.sin(angle)] for angle in angles]
    elif dim == 10:
        # dummy version for testing
        angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
        init_bias_mean_list = [[scale * np.cos(angle), scale * np.sin(angle),
                                scale * np.cos(angle), scale * np.sin(angle),
                                scale * np.cos(angle), scale * np.sin(angle),
                                scale * np.cos(angle), scale * np.sin(angle),
                                scale * np.cos(angle), scale * np.sin(angle)] for angle in angles]
    else:
        raise ValueError(f"Invalid dim {dim}. Now only support 2 or 10.")
    return init_bias_mean_list


# test
# model = get_model("toy_task_model_2", 2, 'cuda', 128, 5, "xavier")
