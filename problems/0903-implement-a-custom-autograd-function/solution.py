import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: save anything backward will need, then return the ReLU output
        ctx.relu = torch.relu(x)
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: return dL/dx using the saved tensors and grad_output
        return grad_output * (ctx.relu > 0).to(ctx.relu.dtype)
