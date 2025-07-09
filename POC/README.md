# PacketNet vs StandardMLP Benchmark

This project benchmarks a modular neural network architecture called **PacketNet** against a standard multilayer perceptron (**StandardMLP**) using synthetic regression data. It is a pure NumPy implementation and does not rely on deep learning frameworks.

## Project Structure

- `benchmark.py` — Main script for benchmarking models (parameter count, speed, MSE, entropy).
- `trainer.py` — Contains a simple training loop and MSE function (for reference).
- `standardnet.py` — Implements a standard MLP (StandardMLP).
- `psinetwork.py` — Implements PacketNet, a modular neural network with routers, experts, and aggregator.
- `router.py` — Implements PacketRouter for routing logic.
- `expert.py` — Implements ExpertNeuron, a mini-MLP block.
- `aggregator.py` — Implements OutputAggregator, which deduplicates and aggregates expert outputs.
- `requirements.txt` — Python dependencies (NumPy).

## How to Run

1. **Install dependencies** (NumPy):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the benchmark:**
   ```bash
   python benchmark.py
   ```

## What It Does
- Generates synthetic regression data.
- Initializes both PacketNet and StandardMLP models.
- Compares parameter counts, inference speed, and mean squared error (MSE) on a test set.
- Reports routing entropy for PacketNet.

## Output Example
```
Packet-Net  params : <number>
Standard MLP params: <number>
Inference speed (1000 samples):
  Packet-Net   <seconds>s
  Standard MLP <seconds>s
MSE on test set:
  Packet-Net   <mse>
  Standard MLP <mse>
Packet routing entropy: <entropy>
```

## Notes
- The code is educational and focuses on architecture comparison, not on training performance.
- Backpropagation and model training are not implemented in detail (see `trainer.py` for a stub).
- All code is pure NumPy for clarity and simplicity.

## License
This project is provided for educational and research purposes.
#
