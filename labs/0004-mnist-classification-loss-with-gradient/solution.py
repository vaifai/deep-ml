# --- cell 1 ---
import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: how to aggregate per-sample losses:
               "mean" → average over batch (gradient scaled by 1/N)
               "sum"  → sum over batch (gradient unscaled)
               "none" → return per-sample loss vector (no aggregation)
    **kwargs:  absorbs any extra arguments from the training harness

    Returns: (loss, grad) where grad has the same shape as preds
    """
    # Your implementation here
    assert preds.ndim == 2
    N, C = preds.shape
    assert target.shape == (N,)
    assert np.issubdtype(target.dtype, np.integer)
    assert np.all(target >= 0) and np.all(target < C)
    
    T = np.zeros_like(preds)
    T[np.arange(N), target] = 1.0
    
    p = np.clip(preds, 1e-12, 1.0)
    per_sample = -np.sum(T * np.log(p), axis=1)
    
    d_preds = -T/p
    
    if reduction == "mean":
        loss = float(np.mean(per_sample))
        d_preds = d_preds/N
    elif reduction == "sum":
        loss = float(np.sum(per_sample))
    elif reduction == "none":
        loss = per_sample
    return loss, d_preds