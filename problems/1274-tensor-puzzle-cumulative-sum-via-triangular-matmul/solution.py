import torch

def cumsum(a: torch.Tensor) -> torch.Tensor:
    """Prefix sums via lower-triangular matmul (no torch.cumsum)."""
    # Your code here
    n = len(a)
    tensor = torch.zeros(n)
    tensor[0]=a[0]
    for i in range(1,n):
        tensor[i] = tensor[i-1] + a[i]
    return tensor
