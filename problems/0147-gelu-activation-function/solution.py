import torch
import torch.nn.functional as F

def GeLU(x: torch.Tensor) -> torch.Tensor:
    # Your code here
    return F.gelu(x)