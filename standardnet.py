import numpy as np
import math

class Act:
    @staticmethod
    def relu(x):   return np.maximum(0, x)

class StandardMLP:
    def __init__(self, in_dim, hidden=[128,64,32], out_dim=1):
        self.W=[]; self.b=[]
        prev=in_dim
        for h in hidden:
            self.W.append(np.random.randn(prev,h)*math.sqrt(2/prev))
            self.b.append(np.zeros((1,h))); prev=h
        self.W.append(np.random.randn(prev,out_dim)*math.sqrt(2/prev))
        self.b.append(np.zeros((1,out_dim)))
    def __call__(self,x):
        for W,b in zip(self.W[:-1],self.b[:-1]):
            x = Act.relu(x@W+b)
        return x@self.W[-1]+self.b[-1]
