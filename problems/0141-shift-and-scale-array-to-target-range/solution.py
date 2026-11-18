import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    # Your code here
    m1,m2 = np.min(values).item(),np.max(values).item()
    return c + (d-c)/(m2-m1)*(values - m1)