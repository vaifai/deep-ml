import torch

def diff(a: torch.Tensor) -> torch.Tensor:
    """First-order difference; out[0]=a[0], out[i]=a[i]-a[i-1] (no torch.diff)."""
    # Your code here
    n = len(a)
    out = torch.zeros(n)
    out[0]=a[0]
    for i in range(1,n):
        out[i] = a[i] - a[i-1]
    return out
