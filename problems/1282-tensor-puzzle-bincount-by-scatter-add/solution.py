import torch

def bincount(a: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Count occurrences of each integer 0..n_bins-1 (no torch.bincount)."""
    # Your code here
    size = len(a)
    tensor = torch.zeros(n_bins,dtype=torch.int64)
    count = {}
    for item in a:
        if item.item() in count:
            count[item.item()]+=1
        else:
            count[item.item()]=1

    for i in range(n_bins):
        tensor[i] = count[i] if i in count else 0
    return tensor
