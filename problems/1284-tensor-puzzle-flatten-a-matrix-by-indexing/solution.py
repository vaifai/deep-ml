import torch

def flatten(A: torch.Tensor) -> torch.Tensor:
    """Row-major flatten via index arithmetic (no reshape/view/flatten)."""
    # Your code here
    rows, cols = A.shape
    tensor = torch.zeros((rows*cols))
    for i in range(rows):
        for j in range(cols):
            tensor[(i*cols + j)] = A[i][j]
    return tensor
