import torch

def vstack(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Stack two equal-length vectors as rows without torch.stack/vstack."""
    # Your code here
    n = len(a)
    tensor = torch.empty((2,n), dtype = a.dtype)
    tensor[0]=a
    tensor[1]=b
    return tensor
