import numpy as np
import time
from psinetwork import PacketNet
from standardnet import StandardMLP

def mse(a, b):
    return ((a-b)**2).mean()

np.random.seed(0)
X = np.random.randn(2000,128)
w = np.random.randn(128,1)
y = X@w + 0.1*np.random.randn(2000,1)

Xtr,Xte = X[:1600], X[1600:]
ytr,yte = y[:1600], y[1600:]

packet = PacketNet(128)
plain  = StandardMLP(128)

def speed(model, X, n=20):
    t = [time.time()]
    for _ in range(n):
        model(X)
        t.append(time.time())
    return (t[-1] - t[0]) / n

print("Packet-Net  params :", sum(p.size for p in (
    [item for sublist in [[r.W1, r.b1, r.W2, r.b2] if hasattr(r, 'W1') else [r.key] for r in packet.router] for item in sublist] +
    [item for sublist in [[e.W1, e.b1, e.W2, e.b2, e.W3, e.b3] for e in packet.expert] for item in sublist] +
    [packet.aggr.attW, packet.aggr.attb, packet.aggr.attV, packet.aggr.outW, packet.aggr.outb]
)))
print("Standard MLP params:", sum(W.size+b.size for W,b in zip(plain.W,plain.b)))

print("Inference speed (1000 samples):")
print("  Packet-Net  ", f"{speed(packet, Xte):.4f}s")
print("  Standard MLP", f"{speed(plain , Xte):.4f}s")

print("MSE on test set:")
print("  Packet-Net  ", f"{mse(packet(Xte),yte):.4f}")
print("  Standard MLP", f"{mse(plain(Xte), yte):.4f}")
print("Packet routing entropy:", f"{packet.stats['entropy']:.3f}")
