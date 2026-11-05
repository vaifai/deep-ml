class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
	def __repr__(self):
		def fmt(x):
			return int(x) if float(x).is_integer() else round(x, 4)
		return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

	def __add__(self, other):
		 # Implement addition here
		value = Value(self.data + other.data, (self,other),'+')
		def _backward():
			self.grad+= 1.0*value.grad
			other.grad+= 1.0*value.grad
			#print("+ grad", self.data,self.grad, other.data,other.grad)
		value._backward = _backward
		return value

	def __mul__(self, other):
		# Implement multiplication here
		value = Value(self.data * other.data, (self,other),'*')
		def _backward():
			#print(self.data,other.data,value.grad,value.data)
			self.grad+= other.data*value.grad
			other.grad+= self.data*value.grad
		value._backward = _backward
		return value

	def relu(self):
		# Implement ReLU here
		value = Value(self.data if self.data > 0 else 0, (self,), 'relu')
		def _backward():
			self.grad+= (self.data > 0)*value.grad
		value._backward = _backward
		return value

	def backward(self):
		# Implement backward pass here
		topo,visited=[],set()
		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build_topo(child)
				#print(v.data,v._op)
				topo.append(v)
		build_topo(self)
		self.grad = 1.0
		for v in reversed(topo):
			v._backward()
			#print(v.data,v._op,v.grad)