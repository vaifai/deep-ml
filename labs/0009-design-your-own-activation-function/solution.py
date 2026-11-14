# --- cell 1 ---
import numpy as np

def activation(x):
    '''
    Apply an activation function element-wise.
    
    Args:
        x: numpy array of any shape (raw neuron outputs)
    
    Returns:
        numpy array of same shape (activated outputs)
    
    Requirements:
        - Must be non-linear (not just returning x)
        - Must work on arrays of any shape
        - Must be deterministic
    '''
    # TODO: Implement your activation function
    
    #sigmoid = np.exp(-np.logaddexp(0, -x))  # Replace with your activation
    result = 0.5 * x * (
        1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) *
            (x + 0.044715 * x**3)
        )
    )
    #result = x * sigmoid
    
    return result