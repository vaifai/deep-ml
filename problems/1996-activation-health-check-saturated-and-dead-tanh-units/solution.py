import numpy as np


def activation_health(acts, threshold=0.97):
    """One dict per layer: mean, std, saturated (% of values with |t| > threshold) and dead (neuron indices)."""
    answer = []
    for layer in acts:
        d = {}
        d["mean"] = round(np.mean(layer),4).item()
        d["std"] = round(np.std(layer),4).item()
        d["saturated"] = round((np.sum(np.abs(layer) > threshold)/layer.size)*100 , 2).item()
        count = [0]*layer.shape[1]
        for i in range(layer.shape[0]):
            for j in range(layer.shape[1]):
                if np.abs(layer[i][j]) > threshold:
                    count[j]+=1
        d["dead"] = [i for i in range(layer.shape[1]) if count[i] == layer.shape[0]]
        answer.append(d)
    return answer
