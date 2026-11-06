import numpy as np

def reshape_backward(grad_output: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Backward pass for a reshape/view operation.

    Args:
        grad_output: numpy array, gradient w.r.t. the reshape output.
        original_shape: tuple of ints, shape of the input to the forward reshape.

    Returns:
        numpy array of shape `original_shape` containing the gradient
        w.r.t. the input tensor.
    """
    mul,mul_grad = 1.0,1.0
    for item in original_shape:
        mul*=item
    for item in grad_output.shape:
        mul_grad*=item
    #print(mul, mul_grad)
    if(mul != mul_grad):
        raise ValueError(" grad output size differs from original shape")
    return grad_output.reshape(original_shape)
