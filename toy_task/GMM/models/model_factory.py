from toy_task.GMM.models.GMM_model import ConditionalGMM
from toy_task.GMM.models.GMM_model_2 import ConditionalGMM2
from toy_task.GMM.models.GMM_model_3 import EmbeddedConditionalGMM
from toy_task.GMM.utils.network_utils import initialize_weights


def get_model(config, device):
    """
    Creates and initializes a model based on the provided configuration.

    Parameters:
    - config (dict): A dictionary containing model configuration parameters.
    - device (str or torch.device): The device (e.g., 'cpu' or 'cuda') on which the model should be loaded.

    Returns:
    - torch.nn.Module: An initialized model instance.

    Raises:
    - ValueError: If an invalid model name is provided in the configuration.
    """

    # Extract model configuration parameters from the config dictionary
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

    # Create and initialize the model based on the model_name
    if model_name == "toy_task_model_1":
        # Create an instance of ConditionalGMM, non-embedded seperate expert model
        model = ConditionalGMM(
            num_gate_layer=num_gate_layer,
            num_neuron_gate_layer=num_neuron_gate_layer,
            num_component_layer=num_component_layer,
            num_neuron_component_layer=num_neuron_component_layer,
            n_components=init_components,
            dim=dim,
            context_dim=context_dim,
            random_init=random_init,
            init_scale=init_scale
        )
        # Initialize model weights with specific settings
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean'])

    elif model_name == "toy_task_model_2":
        # Create an instance of ConditionalGMM2, non-embedded shared expert model
        model = ConditionalGMM2(
            num_gate_layer=num_gate_layer,
            num_neuron_gate_layer=num_neuron_gate_layer,
            num_component_layer=num_component_layer,
            num_neuron_component_layer=num_neuron_component_layer,
            n_components=init_components,
            dim=dim,
            context_dim=context_dim,
            random_init=random_init,
            init_scale=init_scale
        )
        # Initialize model weights with specific settings
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean'])

    elif model_name == "embedded_model":
        # Create an instance of EmbeddedConditionalGMM, embedded and shared expert model
        model = EmbeddedConditionalGMM(
            num_gate_layer=num_gate_layer,
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
            init_std=init_std
        )
        # Initialize model weights with more layers to preserve bias
        initialize_weights(model, initialization_type, preserve_bias_layers=['fc_mean', 'embedded'])

    else:
        raise ValueError(
            f"Invalid model name {model_name}. Choose one from 'toy_task_model_1', 'toy_task_model_2', embedded_model.")

    return model.to(device)
