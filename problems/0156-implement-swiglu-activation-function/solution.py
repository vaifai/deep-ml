import torch,torch.nn.functional as F

def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor of shape (batch_size, 2d)
    Returns:
        torch.Tensor of shape (batch_size, d)
    """
    # Your code here
    rows,cols = x.shape
    tensor = torch.zeros((rows, int(cols/2)), dtype=torch.float64)
    for i in range(rows):
        x1,x2 = x[i][:int(cols/2)],x[i][int(cols/2):]
        tensor[i] = torch.mul(x1,F.silu(x2))
    return tensor