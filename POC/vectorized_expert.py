import numpy as np

class VectorizedExpertNeuron:
    """Vectorized expert neuron for batch processing"""
    def __init__(self, in_dim, hidden, out_dim):
        self.W1 = np.random.randn(in_dim, hidden) * np.sqrt(2/in_dim)
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, out_dim) * np.sqrt(2/hidden)
        self.b2 = np.zeros((1, out_dim))
    def fwd(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        return z2
