import torch

def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked leaf for x, compute f(x), run backprop, return df/dx as a float
    x = torch.tensor(float(x_value), requires_grad=True)
    f_x = x**2 + 3*x + 2
    f_x.backward()
    return x.grad.item()
