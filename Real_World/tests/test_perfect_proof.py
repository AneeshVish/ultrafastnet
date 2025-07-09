"""
Test: Perfect Proof for Hard Neural Network Problem

This test demonstrates:
- Solving a challenging neural network problem with two approaches:
    1. Classical Feedforward Training (CFT, e.g., standard MLP)
    2. Your UltraFastNet approach (PacketNet)
- Compares time taken, accuracy, and displays graphs for both approaches.

Problem: Classification of a highly non-linear synthetic dataset (e.g., two interleaving moons or circles)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from ultrafastnet import PacketNet


def solve_with_cft(X_train, y_train, X_test, y_test):
    start = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=300, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start
    return acc, elapsed, y_pred


def solve_with_ultrafastnet(X_train, y_train, X_test, y_test):
    # Pure forward pass, no learning
    start = time.time()
    net = PacketNet(input_dim=2, n_blocks=2, n_neurons=2, hidden_dim=16, neuron_output_dim=2, final_output_dim=2)
    out = net(X_test)
    y_pred = np.argmax(out, axis=1)
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start
    return acc, elapsed, y_pred


def plot_results(X_test, y_test, y_pred_cft, y_pred_ufn, acc_cft, acc_ufn):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"CFT (MLP) | Acc: {acc_cft:.2f}")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_cft, cmap="coolwarm", alpha=0.7)
    plt.subplot(1, 2, 2)
    plt.title(f"UltraFastNet | Acc: {acc_ufn:.2f}")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_ufn, cmap="coolwarm", alpha=0.7)
    plt.tight_layout()
    plt.show()


def test_perfect_proof():
    # Generate hard synthetic dataset
    X, y = make_moons(n_samples=1200, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    acc_cft, time_cft, y_pred_cft = solve_with_cft(X_train, y_train, X_test, y_test)
    acc_ufn, time_ufn, y_pred_ufn = solve_with_ultrafastnet(X_train, y_train, X_test, y_test)

    print("CFT (MLP) Accuracy:", acc_cft)
    print("CFT (MLP) Time taken (s):", time_cft)
    print("UltraFastNet Accuracy:", acc_ufn)
    print("UltraFastNet Time taken (s):", time_ufn)

    plot_results(X_test, y_test, y_pred_cft, y_pred_ufn, acc_cft, acc_ufn)

if __name__ == "__main__":
    test_perfect_proof()
