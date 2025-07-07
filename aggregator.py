import numpy as np

class Act:
    @staticmethod
    def relu(x):   return np.maximum(0, x)
    @staticmethod
    def softmax(x):
        e = np.exp(x - x.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

class OutputAggregator:
    def __init__(self, n_neuron, n_in, n_out):
        self.attW = np.random.randn(n_in, 32)*0.1
        self.attb = np.zeros((1, 32))
        self.attV = np.random.randn(32, 1)*0.1
        self.outW = np.random.randn(n_in, n_out)*0.1
        self.outb = np.zeros((1, n_out))
        self.th   = 0.95   # duplicate threshold
        self.n    = n_neuron
    def dedup(self, O):
        """Mask duplicates by cosine similarity."""
        mask = np.ones((len(O), self.n), bool)
        for b, row in enumerate(O):
            nrm = row/ (np.linalg.norm(row, axis=1, keepdims=True)+1e-8)
            sim = nrm @ nrm.T
            for i in range(self.n):
                for j in range(i+1, self.n):
                    if sim[i,j] > self.th: mask[b,j] = False
        return mask
    def __call__(self, O, P):          # O=(B,N,D)  P=(B,N)
        mask   = self.dedup(O)
        P      = P*mask;   P /= P.sum(1, keepdims=True)+1e-8
        att    = Act.softmax( Act.relu(O @ self.attW + self.attb) @ self.attV )
        W      = att.squeeze(-1)*P
        Z      = (W[:,:,None]*O).sum(1)            # (B,D)
        return Z @ self.outW + self.outb           # (B,final)
