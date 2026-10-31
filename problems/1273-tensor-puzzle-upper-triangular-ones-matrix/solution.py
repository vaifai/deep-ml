import torch

def triu_ones(n: int) -> torch.Tensor:
    """Upper-triangular ones matrix via comparisons (no torch.triu)."""
    # Your code here
    tensor = torch.zeros((n,n),dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            if(j>=i):
                tensor[(i,j)]=1.
    return tensor
