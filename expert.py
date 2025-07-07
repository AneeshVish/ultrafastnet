import numpy as np
import math

class Act:
    @staticmethod
    def relu(x):   return np.maximum(0, x)

class ExpertNeuron:
    def __init__(self, in_dim, hidden, out_dim):
        self.W1 = np.random.randn(in_dim, hidden) * math.sqrt(2/in_dim)
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, hidden//2) * math.sqrt(2/hidden)
        self.b2 = np.zeros((1, hidden//2))
        self.W3 = np.random.randn(hidden//2, out_dim) * math.sqrt(2/(hidden//2))
        self.b3 = np.zeros((1, out_dim))
    def fwd(self, x):
        self.x, self.z1 = x, x @ self.W1 + self.b1
        self.a1         = Act.relu(self.z1)
        self.z2         = self.a1 @ self.W2 + self.b2
        self.a2         = Act.relu(self.z2)
        self.z3         = self.a2 @ self.W3 + self.b3
        return self.z3
