import math
def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	sigmoid = 1/(1+math.exp(-x))
	sig_der = sigmoid*(1-sigmoid)
	tanh_der = (1/math.cosh(x))**2
	relu_der = 1.0 if x > 0 else 0.0
	return {
		"sigmoid": sig_der,
		"tanh": tanh_der,
		"relu": relu_der
	}