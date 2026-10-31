import numpy as np

def vector_sum(a: np.ndarray):
    """Sum elements of 1-D array a without np.sum / loops."""
    # Your code here
    n = len(a)
    return np.dot(a, np.arange(n)*0 + 1)
