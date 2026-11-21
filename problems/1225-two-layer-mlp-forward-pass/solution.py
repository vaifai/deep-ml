import torch
import torch.nn as nn


def two_layer_mlp_forward(x, w1, b1, w2, b2):
    """Build a 2-layer MLP, set fixed weights, return scalar output.

    Args:
        x (torch.Tensor): Input of shape (1, 2).
        w1 (torch.Tensor): First Linear weight, shape (2, 2).
        b1 (torch.Tensor): First Linear bias, shape (2,).
        w2 (torch.Tensor): Second Linear weight, shape (1, 2).
        b2 (torch.Tensor): Second Linear bias, shape (1,).

    Returns:
        float: Scalar network output.
    """
    # TODO
    model = nn.Sequential(
        nn.Linear(2,2),
        nn.ReLU(),
        nn.Linear(2,1)
    )

    with torch.no_grad():
        model[0].weight.copy_(torch.as_tensor(w1,dtype=model[0].weight.dtype))
        model[2].weight.copy_(torch.as_tensor(w2,dtype=model[2].weight.dtype))
        model[0].bias.copy_(torch.as_tensor(b1,dtype=model[0].bias.dtype))
        model[2].bias.copy_(torch.as_tensor(b2,dtype=model[2].bias.dtype))
    return model(x).item()

