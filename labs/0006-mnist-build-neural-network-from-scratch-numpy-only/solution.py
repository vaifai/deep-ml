# --- cell 1 ---
import numpy as np

class NeuralNetwork:
    '''
    Build a neural network from scratch using only NumPy.
    Required architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        '''
        Initialize network parameters.
        Use small random initialization (e.g., Xavier/He initialization).
        '''
        self.lr = lr
        # TODO: Initialize weights and biases
        # Hint: Use np.random.randn() with proper scaling
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / 784)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / 784)
        self.b1 = np.zeros((1,128))
        self.b2 = np.zeros((1,10))
        self.cache = {
            "W1": self.W1,
            "W2": self.W2
        }
    
    def forward(self, X):
        '''
        Forward pass through the network.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            probs: Class probabilities, shape (N, 10)
        
        Must cache intermediate values for backward pass!
        '''
        # TODO: Implement forward pass
        # Layer 1: X @ W1 + b1, then ReLU
        # Layer 2: hidden @ W2 + b2, then Softmax
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(0,z1)
        z2 = h1 @ self.W2 + self.b2
        e_x = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = e_x / np.sum(e_x, axis=1, keepdims=True)
        self.cache["z1"] = z1
        self.cache["h1"] = h1
        self.cache["z2"] = z2
        self.cache["probs"] = probs
        return probs
    
    def backward(self, X, y, probs):
        '''
        Backward pass - compute gradients for all parameters.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
            probs: Predicted probabilities from forward pass, shape (N, 10)
        
        Returns:
            loss: Scalar cross-entropy loss
        
        Must update self.W1, self.b1, self.W2, self.b2 using computed gradients!
        '''
        # TODO: Implement backward pass
        # 1. Compute loss
        # 2. Compute gradient of loss w.r.t. softmax output
        # 3. Backprop through linear layer 2
        # 4. Backprop through ReLU
        # 5. Backprop through linear layer 1
        # 6. Update all parameters using gradients
        lr = 0.05
        N = y.shape[0]
        correct_probs = probs[np.arange(N), y]
        # print(correct_probs.shape, probs.shape,y.shape, y)
        loss = -np.mean(np.log(np.clip(correct_probs, 1e-12,1 )))
        dz2 = probs.copy()
        dz2[np.arange(N), y] -= 1
        dz2 /= N
        dw2 = self.cache["h1"].T @ dz2
        db2 = np.sum(dz2,axis=0,keepdims=True)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (self.cache["z1"] > 0)
        dw1 = X.T @ dz1
        db1 =  np.sum(dz1,axis=0,keepdims=True)
        self.W1 = self.W1 - lr * dw1
        self.W2 = self.W2 - lr * dw2
        self.b1 = self.b1 - lr * db1
        self.b2 = self.b2 - lr * db2
        return loss 
    
    def train_step(self, X, y):
        '''
        Complete training step: forward + backward + update.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
        
        Returns:
            loss: Scalar loss value
        '''
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss
    
    def predict(self, X):
        '''
        Predict class labels.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            predictions: Predicted class labels, shape (N,)
        '''
        probs = self.forward(X)
        return np.argmax(probs, axis=1)