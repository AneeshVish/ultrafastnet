import numpy as np
from ultra_fast_router import OptimizedPacketRouter
from vectorized_expert import VectorizedExpertNeuron
from fast_aggregator import FastOutputAggregator

class UltraFastPacketNet:
    """Production-ready ultra-fast packet-switching neural network"""
    def __init__(self, in_dim, n_blocks=8, n_neuron=6, hid=32, neur_out=16, final=1):
        assert in_dim % n_blocks == 0
        self.block = in_dim // n_blocks
        self.n_neuron = n_neuron
        self.router = [OptimizedPacketRouter(self.block, n_neuron, "frequency_hash")
                      for _ in range(n_blocks)]
        self.expert = [VectorizedExpertNeuron(self.block, hid, neur_out)
                      for _ in range(n_neuron)]
        self.aggr = FastOutputAggregator(n_neuron, neur_out, final)
        self.stats = {}
    def __call__(self, x):
        B = len(x)
        N = self.n_neuron
        all_outputs = []
        all_probs = []
        for b in range(0, x.shape[1], self.block):
            blk = x[:, b:b+self.block]
            idx = b // self.block
            P = self.router[idx](blk)
            all_probs.append(P)
            expert_outputs = np.zeros((B, N, self.expert[0].W2.shape[1]))
            for n, expert in enumerate(self.expert):
                expert_outputs[:, n] = expert.fwd(blk)
            weighted_outputs = expert_outputs * P[:, :, np.newaxis]
            all_outputs.append(weighted_outputs)
        total_outputs = np.sum(all_outputs, axis=0)
        total_probs = np.mean(all_probs, axis=0)
        y_hat = self.aggr(total_outputs, total_probs)
        self.stats = {
            'utilisation': np.count_nonzero(total_probs.mean(0) > 1e-3) / N,
            'entropy': -(total_probs * np.log(total_probs + 1e-9)).sum(1).mean(),
            'efficiency': 1.0 - np.std(total_probs.mean(0)) / (np.mean(total_probs.mean(0)) + 1e-8)
        }
        return y_hat
