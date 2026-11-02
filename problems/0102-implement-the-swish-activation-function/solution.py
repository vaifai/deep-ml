import torch,torch.nn.functional as F

def swish(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the Swish activation function.

    Args:
        x: Input tensor

    Returns:
        The Swish activation value as a tensor
    """
    # Your code here
    return torch.mul(x,F.sigmoid(x))