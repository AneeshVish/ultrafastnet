import numpy as np

class Act:
    @staticmethod
    def relu(x):   return np.maximum(0, x)
    @staticmethod
    def softmax(x):
        e = np.exp(x - x.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

class PacketRouter:
    """Return a (batch, N_neuron) routing matrix."""
    def __init__(self, in_dim, n_neuron, mode="learned"):
        self.mode, self.n = mode, n_neuron
        if mode == "learned":
            self.W1 = np.random.randn(in_dim, 32) * 0.1
            self.b1 = np.zeros((1, 32))
            self.W2 = np.random.randn(32, n_neuron) * 0.1
            self.b2 = np.zeros((1, n_neuron))
        elif mode == "hash":
            self.key = np.random.randn(in_dim)

    def __call__(self, x):
        if self.mode == "learned":
            h = Act.relu(x @ self.W1 + self.b1)
            return Act.softmax(h @ self.W2 + self.b2)
        if self.mode == "hash":
            idx = (x @ self.key).astype(int) % self.n
            m   = np.zeros((len(x), self.n)); m[np.arange(len(x)), idx] = 1
            return m
        if self.mode == "random":
            idx = np.random.randint(0, self.n, len(x))
            m   = np.zeros((len(x), self.n)); m[np.arange(len(x)), idx] = 1
            return m
