import numpy as np

def mse(a, b):
    return ((a-b)**2).mean()

def train(model, X, y, epochs=10, lr=1e-3, batch_size=64):
    n = len(X)
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        X_shuf, y_shuf = X[perm], y[perm]
        for i in range(0, n, batch_size):
            xb = X_shuf[i:i+batch_size]
            yb = y_shuf[i:i+batch_size]
            # Forward pass
            pred = model(xb)
            loss = mse(pred, yb)
            # Backprop and update omitted for brevity (see full repo or use PyTorch for autograd)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")
