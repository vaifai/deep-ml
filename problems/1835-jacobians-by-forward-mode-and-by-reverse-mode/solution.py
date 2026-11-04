import numpy as np

def jacobian_two_ways(layers, x):
    # layers: list of (W, b, act); return dict(J_fwd, J_rev, n_jvp, n_vjp, macs_fwd, macs_rev)
    n = len(x)
    m=layers[-1][0].shape[0]
    n_input = len(x)
    J_fwd,J_rev = np.zeros((m, n)), np.zeros((m,n))

    current = np.asarray(x, dtype=float).copy()
    derivatives = []
    for W,b,act in layers:
        z = W @ current + b
        current,d = activation_and_derivative(z, act)
        derivatives.append(d)
    
    for j in range(n):
        v = np.zeros(n)
        v[j] = 1.0

        for layer_idx, (W,b,act) in enumerate(layers):
            d = derivatives[layer_idx]
            v = d * (W @ v)
        J_fwd[:, j] = v

    for i in range(m):
        u = np.zeros(m)
        u[i] = 1.0

        for layer_idx in range(len(layers)-1,-1,-1):
            d = derivatives[layer_idx]
            W,b,act = layers[layer_idx]
            u = W.T @ (d * u)
        J_rev[i, :] = u
    
    P = sum(W.shape[0] * W.shape[1] for W,b,act in layers)
    return {
        "J_fwd": J_fwd,
        "J_rev": J_rev,
        "n_jvp": n,
        "n_vjp": m,
        "macs_fwd": n*P,
        "macs_rev": m*P
        }
def activation_and_derivative(z, act):
    if act == "relu":
        return np.maximum(0,z), (z>0).astype(float)
    elif act == "sigmoid":
        a = 1/(1 + np.exp(-z))
        d = a * (1-a)
        return a,d
    elif act == "linear":
        a = z
        d = np.ones_like(z)
        return a,d
    else:
        a = np.tanh(z)
        d = 1 - a**2
        return a,d
