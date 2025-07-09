import numpy as np

class FastOutputAggregator:
    """Ultra-fast output aggregator with vectorized deduplication"""
    def __init__(self, n_neuron, n_in, n_out):
        self.n = n_neuron
        self.th = 0.95
        self.attW = np.random.randn(n_in, 8) * 0.1
        self.attb = np.zeros((1, 8))
        self.attV = np.random.randn(8, 1) * 0.1
        self.outW = np.random.randn(n_in, n_out) * 0.1
        self.outb = np.zeros((1, n_out))
    def fast_dedup(self, O):
        B, N, D = O.shape
        mask = np.ones((B, N), dtype=bool)
        norms = np.linalg.norm(O, axis=2, keepdims=True) + 1e-8
        O_normalized = O / norms
        similarities = np.einsum('bnd,bmd->bnm', O_normalized, O_normalized)
        for b in range(B):
            high_sim = similarities[b] > self.th
            np.fill_diagonal(high_sim, False)
            high_sim = np.triu(high_sim, k=1)
            duplicates = np.any(high_sim, axis=0)
            mask[b, duplicates] = False
        return mask
    def __call__(self, O, P):
        mask = self.fast_dedup(O)
        P_masked = P * mask
        P_masked /= P_masked.sum(axis=1, keepdims=True) + 1e-8
        att_features = np.maximum(0, O @ self.attW + self.attb)
        attention = att_features @ self.attV
        attention = np.exp(attention - attention.max(axis=1, keepdims=True))
        attention = attention.squeeze(-1)
        attention /= attention.sum(axis=1, keepdims=True) + 1e-8
        weights = attention * P_masked
        Z = np.einsum('bn,bnd->bd', weights, O)
        return Z @ self.outW + self.outb
