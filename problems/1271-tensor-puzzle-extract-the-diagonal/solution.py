import torch

def diag(A: torch.Tensor) -> torch.Tensor:
    """Extract the main diagonal of square A without torch.diag."""
    # Your code here
    n = A.size(0)
    return A[torch.arange(n), torch.arange(n)]
