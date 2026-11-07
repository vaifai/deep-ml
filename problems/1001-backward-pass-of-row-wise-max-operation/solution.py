import numpy as np

def max_backward(logits: np.ndarray, dlogit_maxes: np.ndarray) -> np.ndarray:
    """
    Backward pass for row-wise max.

    Args:
        logits: array of shape (N, C) -- input to the forward max
        dlogit_maxes: array of shape (N, 1) -- upstream gradient

    Returns:
        dlogits: array of shape (N, C) -- gradient w.r.t. logits
    """
    dlogits = np.zeros_like(logits)
    x,y = logits.shape
    #print(x,y , dlogit_maxes.shape)
    for i in range(x):
        col = np.argmax(logits[i])
        dlogits[i][col] = dlogit_maxes[i][0]
    return dlogits
