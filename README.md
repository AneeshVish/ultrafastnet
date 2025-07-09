# UltraFastNet

UltraFastNet is a high-performance, modular packet-switching neural network library for research and production, featuring vectorized routers, experts, and aggregators. It is designed for speed, flexibility, and ease of use in modern machine learning workflows.

---

## 🚀 Features
- **Production-ready**: Modern packaging, type hints, and robust error handling
- **High Performance**: Vectorized NumPy operations and efficient routing
- **Developer Friendly**: Clean API, comprehensive tests, and logging support
- **Extensible**: Modular core for easy customization
- **CI/CD Ready**: GitHub Actions workflows for testing and publishing

---

## 📦 Installation

UltraFastNet is designed to be easy to install and use in any Python project.

### Install from PyPI (recommended)
Once published, simply run:

```bash
pip install ultrafastnet
```

### Install from source (development mode)
If you want the latest features or to contribute:

```bash
# Clone the repository
$ git clone https://github.com/AneeshVish/ultrafastnet.git
$ cd ultrafastnet

# Install in development mode (with all dev and docs extras)
$ pip install -e ".[dev,docs,examples]"
```

---

## 🧑‍💻 Basic Usage

```python
import numpy as np
from ultrafastnet import PacketNet

# Create a network
network = PacketNet(input_dim=128, n_neurons=8, hidden_dim=64)

# Generate random input data
X = np.random.randn(100, 128).astype(np.float32)

# Run inference
predictions = network(X)
print(f"Output shape: {predictions.shape}")
print(f"Stats: {network.stats}")
```

---

## 📚 Using UltraFastNet as a Library

You can use UltraFastNet as a dependency in your own Python projects. Just install it (from PyPI or source) and import the classes you need:

```python
from ultrafastnet import PacketNet, Router, Expert, Aggregator

# Example: Build your own custom network
class MyCustomNet(PacketNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add your custom logic here

# Or use the core components directly
router = Router(...)
expert = Expert(...)
aggregator = Aggregator(...)
```

You can integrate UltraFastNet into any machine learning pipeline, use it in scripts or notebooks, and extend its core classes for advanced use cases.

---

## 🏗️ Library Architecture Overview

UltraFastNet is organized for clarity and extensibility:

```
ultrafastnet/
├── core/
│   ├── network.py      # PacketNet: main network class
│   ├── router.py       # Router: packet routing logic
│   ├── expert.py       # Expert: expert neuron logic
│   └── aggregator.py   # Aggregator: output aggregation
├── utils/
│   ├── exceptions.py   # Custom exceptions
│   ├── validators.py   # Input/config validation
│   └── logging_config.py # Logging setup
```

- **PacketNet** ties together routing, experts, and aggregation.
- Each component can be used or extended independently.
- Utilities provide validation, error handling, and logging.

---

## 📚 Examples

- See [`examples/basic_usage.py`](examples/basic_usage.py) for a simple demo
- See [`examples/benchmarking.py`](examples/benchmarking.py) for performance tests
- See [`examples/advanced_usage.py`](examples/advanced_usage.py) for advanced features

---

## 🧪 Running Tests

Run the full test suite using:

```bash
pytest
```

Check code formatting and types:

```bash
black .
isort .
mypy .
```

---

## 📝 Documentation

- Full documentation: *(coming soon)*
- API Reference: *(coming soon)*

---

## 🤝 Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

1. Fork the repo and create your branch
2. Make your changes with clear commit messages
3. Add or update tests as needed
4. Ensure all tests pass
5. Open a pull request

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## 🔗 Useful Links
- GitHub: [https://github.com/AneeshVish/ultrafastnet.git](https://github.com/AneeshVish/ultrafastnet.git)

