import torch

def roll(a: torch.Tensor) -> torch.Tensor:
    """Circular left-shift by one via index arithmetic (no torch.roll)."""
    # Your code here
    n = len(a)
    tensor = torch.zeros(n)
    for i in range(1,n):
        tensor[i-1]=a[i]
    tensor[n-1]=a[0]
    return tensor
