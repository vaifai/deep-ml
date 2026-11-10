import numpy as np

def he_initialize(layer_dims: list, method: str = 'normal', seed: int = 42) -> list:
    """
    Initialize weight matrices for a neural network using He initialization.
    
    Args:
        layer_dims: List of integers representing neurons per layer
        method: 'normal' or 'uniform' sampling distribution
        seed: Random seed for reproducibility
    
    Returns:
        List of numpy arrays, one weight matrix per adjacent layer pair
    """
    if seed is not None:
        np.random.seed(seed)
    weights = []
    for i in range(0,len(layer_dims)-1,1):
        val = np.sqrt(2/layer_dims[i]) if method == "normal" else np.sqrt(6/layer_dims[i])
        if method == "normal":
            weights.append(np.random.normal(0,val,(layer_dims[i], layer_dims[i+1])))
        else:
            weights.append(np.random.uniform(-val,val,(layer_dims[i], layer_dims[i+1])))
    return weights