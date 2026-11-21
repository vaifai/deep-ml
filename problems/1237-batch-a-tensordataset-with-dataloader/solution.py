import torch
import math
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    # TODO
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    num_batches = math.ceil(len(dataset)/4)
    first_batch_X_shape_tuple = None
    for batch_X, _ in loader:
        first_batch_X_shape_tuple = tuple(batch_X.shape)
        break
    return (num_batches, first_batch_X_shape_tuple)


