import torch,torch.nn.functional as F

def selu(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the SELU (Scaled Exponential Linear Unit) activation function.

    Args:
        x: Input tensor

    Returns:
        SELU activation tensor
    """
    alpha = torch.tensor(1.6732632423543772)
    scale = torch.tensor(1.0507009873554804)
    # Your code here
    if x > 0 :
        return torch.mul(scale,x)
    else:
        return torch.mul(scale,torch.mul(alpha,torch.exp(x)-1))