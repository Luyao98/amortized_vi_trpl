from toy_task.GMM.models.GMM_model import ConditionalGMM
from toy_task.GMM.utils.network_utils import initialize_weights


def get_model(model_name,
              dim,
              device,
              fc_layer_size,
              n_components,
              initialization_type,
              ):

    if model_name == "toy_task_2d_model":
        if n_components == 4:
            init_bias_mean_list = [
                [5.0, 5.0],
                [-5.0, -5.0],
                [5.0, -5.0],
                [-5.0, 5.0]]
            init_bias_gate = [0.0, 0.0, 0.0, 0.0]
        elif n_components == 1:
            init_bias_mean_list = [[-5.0, 5.0]]
            init_bias_gate = [0.0]
        else:
            raise ValueError(f"Invalid n_components {n_components}. Now only support 1 or 4.")
        model = ConditionalGMM(fc_layer_size, n_components, dim=dim,
                               init_bias_gate=init_bias_gate, init_bias_mean_list=init_bias_mean_list)
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean', 'fc_gate'])
    else:
        raise ValueError(f"Invalid model name {model_name}. Now only support 'toy_task_2d_model'.")

    return model.to(device)
