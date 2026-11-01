import torch

def pad_to(a: torch.Tensor, j: int) -> torch.Tensor:
    """Pad with zeros or truncate a so its length is exactly j."""
    # Your code here
    size = len(a)
    if(j<=size):
        return a[:j].to(torch.float64)
    return torch.cat((a,torch.tensor([0]*(j-size)))).to(torch.float64)

