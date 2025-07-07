import numpy as np
from router import PacketRouter
from expert import ExpertNeuron
from aggregator import OutputAggregator

class PacketNet:
    def __init__(self, in_dim, n_blocks=8, n_neuron=6,
                 hid=64, neur_out=32, final=1,
                 router_mode="learned"):
        assert in_dim % n_blocks == 0
        self.block   = in_dim//n_blocks
        self.router  = [PacketRouter(self.block, n_neuron, router_mode)
                        for _ in range(n_blocks)]
        self.expert  = [ExpertNeuron(self.block, hid, neur_out)
                        for _ in range(n_neuron)]
        self.aggr    = OutputAggregator(n_neuron, neur_out, final)
        self.stats   = {}
    def __call__(self, x):
        B = len(x); N = len(self.expert)
        out = np.zeros((B,N,self.expert[0].W3.shape[1]))
        Ptot= np.zeros((B,N))
        for b in range(0, x.shape[1], self.block):
            blk      = x[:, b:b+self.block]
            idx      = b//self.block
            P        = self.router[idx](blk)              # (B,N)
            for n,e in enumerate(self.expert):
                out[:,n] += e.fwd(blk)*P[:,n:n+1]         # weighted add
            Ptot += P
        Ptot /= (x.shape[1]//self.block)
        y_hat   = self.aggr(out, Ptot)
        self.stats = dict(utilisation=np.count_nonzero(Ptot.mean(0)>1e-3)/N,
                          entropy=-(Ptot*np.log(Ptot+1e-9)).sum(1).mean())
        return y_hat
