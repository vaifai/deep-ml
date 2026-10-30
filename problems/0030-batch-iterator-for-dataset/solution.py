import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	num_items = len(X)
	answer, n = [], len(X)
	for i in range(0,n,batch_size):
		current = []
		current.append(X[i:min(i+batch_size,n)].tolist())
		if y is not None:
			current.append(y[i:min(i+batch_size,n)].tolist())
		answer.append(current)
	return answer