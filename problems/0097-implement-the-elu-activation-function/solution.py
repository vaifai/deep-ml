import torch
import torch.nn.functional as F
import math

def elu(x: float, alpha: float = 1.0) -> float:
    """
    Compute the ELU activation function using PyTorch.

    Args:
        x (float): Input value
        alpha (float): ELU parameter for negative values (default: 1.0)

    Returns:
        float: ELU activation value rounded to 4 decimal places
    """
    # Your code here
    if x>0:
        return x
    x = alpha*(math.exp(x) - 1)
    return round(x,4)