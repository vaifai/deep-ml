import torch

def linear_backward(grad_output, x, W):
    # TODO: return (grad_input, grad_W, grad_b) for y = x @ W.T + b
    x = torch.tensor(x,dtype=torch.float32,requires_grad=True)
    W = torch.tensor(W,dtype=torch.float32,requires_grad=True)
    W_shape = W.shape
    b = torch.zeros((W_shape[0],), requires_grad=True)
    grad_out_t = torch.as_tensor(grad_output, dtype=torch.float32)
    y = x @ W.T + b
    y.backward(grad_out_t)
    return x.grad, W.grad,b.grad
