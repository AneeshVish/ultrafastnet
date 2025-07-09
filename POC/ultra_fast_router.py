import numpy as np

class OptimizedPacketRouter:
    """Ultra-fast packet router with frequency-based routing"""
    def __init__(self, in_dim, n_neuron, mode="frequency_hash"):
        self.mode = mode
        self.n = n_neuron
        if mode == "frequency_hash":
            self.freq_key = np.random.randn(min(8, in_dim//4))
    def __call__(self, x):
        freq_data = np.fft.rfft(x, axis=1)
        magnitude = np.abs(freq_data)
        low_freq = magnitude[:, :len(self.freq_key)]
        idx = (low_freq @ self.freq_key).astype(int) % self.n
        routing = np.zeros((len(x), self.n))
        routing[np.arange(len(x)), idx] = 1
        return routing
