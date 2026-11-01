import torch

def compress(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Keep values where g is True, packed left; pad right with zeros."""
    # Your code here
    size = len(v)
    tensor,ind = torch.zeros(size), 0
    for i in range(size):
        if g[i]:
            tensor[ind]=v[i]
            ind+=1
    return tensor
