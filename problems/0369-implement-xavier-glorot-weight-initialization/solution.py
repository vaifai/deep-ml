import numpy as np

def xavier_init(fan_in: int, fan_out: int, mode: str = 'uniform', seed: int = 42) -> dict:
    """
    Perform Xavier/Glorot weight initialization.

    Args:
        fan_in (int): Number of input units.
        fan_out (int): Number of output units.
        mode (str): 'uniform' or 'normal'.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Contains 'weights' (nested list), 'shape' (list), and 'param' (float).
    """
    # Your code here
    if seed is not None:
        np.random.seed(seed)
    d = {
        "weights": None,
        "shape":None,
        "param":None
    }
    if mode == "uniform":
        val = np.sqrt(6/(fan_in + fan_out))
        d["weights"] = np.random.uniform(-val,val,(fan_in,fan_out)).tolist()
        d["shape"] = [fan_in,fan_out]
        d["param"] = val
        return d
    elif mode == "normal":
        val = np.sqrt(2/(fan_in + fan_out))
        d["weights"] = np.random.uniform(0,val,(fan_in,fan_out)).tolist()
        d["shape"] = [fan_in,fan_out]
        d["param"] = val
        return d
    else:
        raise ValueError("mode must be 'uniform' or 'normal'")