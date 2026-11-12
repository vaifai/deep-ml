import numpy as np

def lsuv_init(layer_dims, X_batch, act, seed, tol=0.1, max_iter=10):
    # return (Ws, variances, n_rescales)
    rng = np.random.default_rng(seed)

    Ws = []
    variances = []
    n_rescales = []

    for l in range(1, len(layer_dims)):
        d_in, d_out = layer_dims[l-1], layer_dims[l]
        G = rng.standard_normal((d_out, d_in))

        if d_out >= d_in:
            Q, R = np.linalg.qr(G, mode="reduced")

            signs = np.sign(np.diag(R))
            signs[signs == 0] = 1

            W = Q * signs
        else:
            Q, R = np.linalg.qr(G.T, mode="reduced")

            signs = np.sign(np.diag(R))
            signs[signs == 0] = 1

            W = (Q * signs).T
        Ws.append(W)
    h = np.asarray(X_batch, dtype=float)

    for l in range(len(Ws)):
        W = Ws[l]

        z = h @ W.T
        v = np.var(z)

        count = 0
        while abs(v - 1.0) >= tol and count < max_iter:
            if v <= 0:
                raise ValueError(
                    f"Layer {l + 1}: cannot rescale zero variance"
                )
            W = W / np.sqrt(v)

            z = h @ W.T
            v = np.var(z)

            count += 1
        Ws[l] = W
        variances.append(float(v))
        n_rescales.append(count)

        # Step 3: Apply activation for the next layer
        if act == "relu":
            h = np.maximum(0, z)
        elif act == "tanh":
            h = np.tanh(z)
        elif act == "linear":
            h = z
        else:
            raise ValueError(f"Unknown activation: {act}")
    return Ws, variances, n_rescales
