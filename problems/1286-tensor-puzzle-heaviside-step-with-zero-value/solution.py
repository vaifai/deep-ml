import torch

def heaviside(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Heaviside step: 0 if a<0, b if a==0, 1 if a>0 (no torch.heaviside)."""
    # Your code here
    n = len(a)
    tensor = torch.zeros(n,dtype = torch.float64)
    for i in range(n):
        if(a[i] == 0):
            tensor[i] = b[i]
        if(a[i] > 0):
            tensor[i] = 1.
    return tensor
