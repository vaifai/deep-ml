import torch

def outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Outer product a_i * b_j via broadcasting (no torch.outer)."""
    # Your code here
    return a.unsqueeze(1) @ b.unsqueeze(0)
