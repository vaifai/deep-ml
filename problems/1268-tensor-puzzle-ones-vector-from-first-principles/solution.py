import numpy as np

def ones(n: int) -> np.ndarray:
    """Return a length-n float vector of ones without calling np.ones."""
    # Your code here
    return (np.arange(n)* 0)+1.
