import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	# Your code here
	n = len(x)
	fx = f(x.copy())
	m = len(fx)
	J = [[0.0 for _ in range(n)] for _ in range(m)]
	for j in range(n):
		x_perturbed = x.copy()
		x_perturbed[j] += h
		fx_perturbed = f(x_perturbed)
		for i in range(m):
			J[i][j] = (fx_perturbed[i] - fx[i]) / h
	return J
	