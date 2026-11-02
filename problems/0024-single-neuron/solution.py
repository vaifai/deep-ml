import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE
    features_tensor = torch.tensor(features, dtype=torch.float64)
    labels_tensor = torch.tensor(labels, dtype=torch.float64)
    weights_tensor = torch.tensor(weights, dtype=torch.float64)
    bias_tensor = torch.tensor(bias, dtype=torch.float64)
    z_tensor = torch.sigmoid(features_tensor @ weights_tensor.T + bias_tensor)
    preds,loss=[],0.0
    return (z_tensor.tolist(),torch.nn.functional.mse_loss(z_tensor, labels_tensor).item())
