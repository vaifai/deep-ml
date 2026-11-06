import numpy as np

def manual_cross_entropy_backward(logits: np.ndarray, y: np.ndarray) -> dict:
    """
    Manually compute gradients through every intermediate of softmax + NLL loss.

    Args:
        logits: numpy array of shape (n, C)
        y:      numpy array of integer class indices, shape (n,)

    Returns:
        Dictionary with keys 'dlogprobs', 'dprobs', 'dcounts_sum_inv',
        'dcounts_sum', 'dcounts', 'dnorm_logits', 'dlogit_maxes', 'dlogits'.
    """
    n = y.shape[0]
    logit_maxes = logits.max(axis=1, keepdims=True)
    norm_logits = logits - logit_maxes
    counts = np.exp(norm_logits)
    counts_sum = counts.sum(axis=1, keepdims=True)
    counts_sum_inv = counts_sum ** -1
    probs = counts * counts_sum_inv
    logprobs = np.log(probs)
    loss = -np.mean(logprobs[range(y.shape[0]),y])
    #print(norm_logits)
    dlogprobs = np.zeros_like(logits)
    dlogprobs[np.arange(n),y] = -1.0/n
    dprobs = dlogprobs*(1/probs)
    dcounts_sum_inv = np.sum(
        dprobs * counts,
        axis=1,
        keepdims=True
        )
    dcounts_sum = -1*dcounts_sum_inv*(counts_sum ** -2)
    dcounts = dprobs*counts_sum_inv + dcounts_sum
    dnorm_logits = dcounts*np.exp(norm_logits)
    dlogit_maxes = -np.sum(
        dnorm_logits,
        axis=1,
        keepdims=True
        ) + 0.0
    dlogits = dnorm_logits.copy()
    max_indices = np.argmax(logits, axis=1)
    dlogits[np.arange(n), max_indices] += dlogit_maxes[:, 0]
    return {
        "dlogprobs": dlogprobs,
        "dprobs": dprobs,
        "dcounts_sum_inv": dcounts_sum_inv,
        "dcounts_sum": dcounts_sum,
        "dcounts": dcounts,
        "dnorm_logits": dnorm_logits,
        "dlogit_maxes": dlogit_maxes,
        "dlogits": dnorm_logits
    }
