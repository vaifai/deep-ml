import torch,torch.nn.functional as F

def softsign(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the Softsign activation function.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: The Softsign of the input
    """
    # Your code here
    x_tensor = torch.tensor(x, dtype=torch.float32)
    return F.softsign(x)