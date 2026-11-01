import torch

def scatter_add(values: torch.Tensor, indices: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Sum values into bins given by indices (no Tensor.scatter_add)."""
    # Your code here
    size = len(values)
    count = {}
    tensor = torch.zeros(n_bins)
    for i in range(size):
        ind = indices[i].item()
        if ind in count:
            count[ind]+=values[i].item()
        else:
            count[ind]=values[i].item()
    for i in range(n_bins):
        tensor[i] = count[i] if i in count else 0
    return tensor
