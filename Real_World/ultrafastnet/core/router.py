"""
Packet router implementation for directing inputs to expert neurons.
"""

import numpy as np
from typing import Literal, Optional
from numpy.typing import NDArray

from ..utils.exceptions import ConfigError, ValidationError
from ..utils.validators import validate_input_shape


class Router:
    """
    Ultra-fast packet router with multiple routing strategies.
    """
    def __init__(
        self,
        input_dim: int,
        n_neurons: int,
        mode: Literal["frequency_hash"] = "frequency_hash"
    ) -> None:
        if input_dim <= 0:
            raise ConfigError(f"input_dim must be positive, got {input_dim}")
        if n_neurons <= 0:
            raise ConfigError(f"n_neurons must be positive, got {n_neurons}")
        
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.mode = mode
        
        # Initialize routing strategy
        self._initialize_routing()
        
    def _initialize_routing(self) -> None:
        """Initialize routing strategy based on mode."""
        if self.mode == "frequency_hash":
            # Create frequency keys for hashing
            key_size = min(8, self.input_dim // 4)
            if key_size <= 0:
                key_size = 1
            self.freq_keys = np.random.randn(key_size).astype(np.float32)
        else:
            raise ConfigError(f"Unknown routing mode: {self.mode}")
    
    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Route input to expert neurons.
        """
        # Validate input
        validate_input_shape(x, expected_features=self.input_dim)
        
        if self.mode == "frequency_hash":
            return self._frequency_hash_routing(x)
        else:
            raise ConfigError(f"Unknown routing mode: {self.mode}")
    
    def __call__(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Make the router callable."""
        return self.forward(x)
    
    def _frequency_hash_routing(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Route based on frequency domain features.
        """
        batch_size = x.shape[0]
        
        # Compute FFT and extract magnitude spectrum
        freq_data = np.fft.rfft(x, axis=1)
        magnitude = np.abs(freq_data)
        
        # Extract low frequency components
        n_freq_components = len(self.freq_keys)
        if magnitude.shape[1] < n_freq_components:
            low_freq = np.zeros((batch_size, n_freq_components), dtype=np.float32)
            low_freq[:, :magnitude.shape[1]] = magnitude
        else:
            low_freq = magnitude[:, :n_freq_components]
        
        # Hash to neuron indices
        hash_values = low_freq @ self.freq_keys
        neuron_indices = np.abs(hash_values.astype(np.int32)) % self.n_neurons
        
        # Create one-hot routing probabilities
        routing_probs = np.zeros((batch_size, self.n_neurons), dtype=np.float32)
        routing_probs[np.arange(batch_size), neuron_indices] = 1.0
        
        return routing_probs
    
    def get_routing_entropy(self, routing_probs: NDArray[np.float32]) -> float:
        """
        Calculate routing entropy to measure load balancing.
        """
        avg_probs = routing_probs.mean(axis=0)
        entropy = -(avg_probs * np.log(avg_probs + 1e-9)).sum()
        return float(entropy)
    
    def get_load_balance_stats(self, routing_probs: NDArray[np.float32]) -> dict:
        """
        Get detailed load balancing statistics.
        """
        avg_probs = routing_probs.mean(axis=0)
        return {
            'entropy': self.get_routing_entropy(routing_probs),
            'std_dev': float(np.std(avg_probs)),
            'max_usage': float(np.max(avg_probs)),
            'min_usage': float(np.min(avg_probs)),
            'utilization': float(np.count_nonzero(avg_probs > 1e-3) / self.n_neurons)
        }
    
    def count_parameters(self) -> int:
        """Count number of parameters in the router."""
        if self.mode == "frequency_hash":
            return len(self.freq_keys)
        return 0
    
    def get_config(self) -> dict:
        """Get router configuration."""
        return {
            'input_dim': self.input_dim,
            'n_neurons': self.n_neurons,
            'mode': self.mode
        }
