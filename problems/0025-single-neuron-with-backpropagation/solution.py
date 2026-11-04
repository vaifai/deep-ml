import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	mse_values = []
	updated_weights = initial_weights.astype(float).copy()
	
	updated_bias = float(initial_bias)
	for _ in range(epochs):
		output = np.dot(features, updated_weights) + updated_bias
		sigmoid = 1 / (1 + np.exp(-output))
		sigmoid_der = sigmoid * (1 - sigmoid)
		mse = np.mean((sigmoid - labels)**2)
		mse_values.append(float(mse))
		bias_der = np.sum((sigmoid - labels) * sigmoid_der)
		bias_der = 2 * bias_der / len(features)
		for j in range(len(updated_weights)):
			derivative = (2 / len(features)) * np.sum(
                (sigmoid - labels) * sigmoid_der * features[:, j]
            )
			updated_weights[j] -= learning_rate * derivative

        # Update bias
		updated_bias -= learning_rate * bias_der
		
	return updated_weights, updated_bias, mse_values