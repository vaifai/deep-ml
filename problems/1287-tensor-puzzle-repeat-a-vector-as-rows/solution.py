import torch

def repeat_rows(a: torch.Tensor, d: int) -> torch.Tensor:
    """Repeat vector a as d identical rows (no torch.repeat/tile)."""
    # Your code here
    tensor = torch.ones((d,1), dtype=a.dtype)
    return tensor*a
