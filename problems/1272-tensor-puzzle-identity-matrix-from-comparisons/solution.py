import torch

def eye(n: int) -> torch.Tensor:
    """n x n identity via comparisons (no torch.eye)."""
    # Your code here
    tensor = torch.zeros(n,n, dtype=torch.float64)
    tensor[torch.arange(n),torch.arange(n)] = 1.
    return tensor
