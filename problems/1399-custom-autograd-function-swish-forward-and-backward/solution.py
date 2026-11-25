import torch


class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: save x; return x * sigmoid(x)
        ctx.save_for_backward = {}
        ctx.save_for_backward["x"] = x
        ctx.save_for_backward["sigmoid_x"] = torch.sigmoid(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: recover x from ctx, recompute the sigmoid, return grad_output times the local derivative
        return grad_output * ctx.save_for_backward["sigmoid_x"]*(1 + ctx.save_for_backward["x"]*(1 - ctx.save_for_backward["sigmoid_x"]))


def swish(x):
    # TODO: SwishFunction.apply(x)
    return SwishFunction.apply(x)
