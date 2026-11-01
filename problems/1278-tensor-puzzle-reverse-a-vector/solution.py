import torch

def flip(a: torch.Tensor) -> torch.Tensor:
    """Reverse a vector via index arithmetic (no torch.flip)."""
    # Your code here
    n = len(a)
    rev = n-1-torch.arange(n)
    return a[rev]

