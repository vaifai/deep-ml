import torch

def bucketize(v: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """Bucket index = how many boundaries each value is >= (no torch.bucketize)."""
    # Your code here
    size = len(v)
    tensor = torch.zeros(size, dtype=torch.int64)
    for i in range(size):
        val=0
        for j in range(len(boundaries)):
            if(v[i].item() >= boundaries[j].item()):
                val+=1
            else:
                break
        tensor[i]=val
    return tensor

