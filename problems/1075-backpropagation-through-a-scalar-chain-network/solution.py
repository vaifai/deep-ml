import numpy as np
import math

def backprop_chain(a0: float, y: float, w: list, b: list) -> dict:
    """
    Compute gradients of C = (a_L - y)^2 with respect to each weight and bias
    in a chain network where every layer has a single sigmoid neuron.

    Args:
        a0: input activation (scalar)
        y: target value (scalar)
        w: list of L weights
        b: list of L biases

    Returns:
        Dictionary with keys 'dW' and 'dB', each a list of L floats rounded to 6 places.
    """
    n = len(w)
    dW, dB = [0.0]*n, [0.0]*n
    act,inp,z = [], a0,[]
    for i in range(n):
        act.append(inp)
        out = w[i]*inp + b[i]
        z.append(out)
        inp = sigmoid(out)
    current_grad = 2*(inp - y)
    #print(act, z, inp)
    for i in range(n-1,-1,-1):
        local_grad = sigmoid(z[i]) * (1-sigmoid(z[i]))
        dW[i] = round(current_grad*local_grad*act[i],6)
        dB[i] = round(current_grad*local_grad,6)
        current_grad=current_grad*local_grad*w[i]
        #print(local_grad, dW[i], current_grad_w)

    return {
        "dW": dW,
        "dB": dB
        }
def sigmoid(x:float) -> float:
    return 1/(1 + math.exp(-x))