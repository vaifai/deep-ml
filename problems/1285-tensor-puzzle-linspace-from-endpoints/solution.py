import torch

def linspace(start, stop, n: int) -> torch.Tensor:
    """n evenly spaced values from start to stop inclusive (no torch.linspace)."""
    # Your code here
    if(n==1):
        return torch.tensor([start], dtype = torch.float64)
    step = (stop - start)/(n-1)
    tensor = torch.zeros(n, dtype=torch.float64)
    ind = 0
    while(ind < n):
        tensor[ind]=start
        ind+=1
        start+=step
    return tensor
