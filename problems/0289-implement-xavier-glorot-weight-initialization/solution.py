import numpy as np

def xavier_initialization(shape: tuple, mode: str = 'uniform', seed: int = None) -> np.ndarray:
    """
    Implement Xavier/Glorot weight initialization.
    
    Args:
        shape: Tuple of (fan_in, fan_out) representing weight matrix dimensions
        mode: 'uniform' or 'normal' initialization
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Initialized weight matrix as numpy array
    """
    # Your code here
    if seed is not None:
        np.random.seed(seed)
    if mode == "uniform":
        limit = np.sqrt(6/(shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, shape)
    elif mode == "normal":
        sigma = np.sqrt(2/(shape[0] + shape[1]))
        return np.random.normal(0, sigma, shape)
    else:
        raise ValueError("Mode not correct")