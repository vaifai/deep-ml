import torch
import torch.nn.functional as F
import math

def softplus(x: float) -> torch.Tensor:
    """
    Compute the softplus activation function using PyTorch.

    Args:
        x: Input value

    Returns:
        The softplus value: log(1 + e^x) as a torch.Tensor
    """
    # Your code here
    x_tensor = torch.tensor(x, dtype=torch.float32)
    return F.softplus(x_tensor)