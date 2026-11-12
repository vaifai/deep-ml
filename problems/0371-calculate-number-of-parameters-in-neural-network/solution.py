def calculate_parameters(layers: list[dict]) -> int:
    """
    Calculate the total number of trainable parameters in a neural network.

    Args:
        layers: List of dictionaries, each describing a layer.

    Returns:
        Total number of trainable parameters (int).
    """
    # Your code here
    params = 0
    for layer in layers:
        if layer["type"] == "dense":
            params+=layer["input_size"]*layer["output_size"]
            if "bias" not in layer or layer["bias"]:
                params+=layer["output_size"]
        else:
            params+=layer["in_channels"]*layer["out_channels"]*layer["kernel_size"]*layer["kernel_size"]
            if "bias" not in layer or layer["bias"]:
                params+=layer["out_channels"]
    return params