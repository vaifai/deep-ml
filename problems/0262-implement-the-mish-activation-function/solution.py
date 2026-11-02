import torch,torch.nn.functional as F

def mish(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Mish activation function.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Mish activation value rounded to 4 decimal places
    """
    # Your code here
    return x * torch.tanh(F.softplus(x))