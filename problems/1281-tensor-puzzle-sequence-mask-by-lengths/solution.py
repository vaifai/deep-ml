import torch

def sequence_mask(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Zero out positions at or beyond each row's length."""
    # Your code here
    rows,cols = values.shape
    tensor = torch.zeros((rows,cols), dtype = values.dtype)
    for i in range(rows):
        for j in range(cols):
            tensor[i][j] = values[i][j] if j <lengths[i] else 0
    return tensor
