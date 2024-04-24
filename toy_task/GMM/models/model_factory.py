from toy_task.GMM.models.GMM_model import ConditionalGMM
from toy_task.GMM.utils.network_utils import initialize_weights


def get_model(model_name,
              device,
              fc_layer_size,
              n_components,
              initialization_type,
              ):

    if model_name == "toy_task_model":
        init_bias_mean_list = [
            [10.0, 10.0],
            [-10.0, -10.0],
            [10.0, -10.0],
            [-10.0, 10.0]]
        model = ConditionalGMM(fc_layer_size, n_components, init_bias_mean_list)
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean', 'fc_gate'])
    else:
        raise ValueError(f"Invalid model name {model_name}. Now only support 'toy_task_model'.")

    return model.to(device)
