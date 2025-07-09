"""
Expert neuron implementation for specialized processing.
"""

import numpy as np
from typing import Literal, Optional, Tuple
from numpy.typing import NDArray

from ..utils.exceptions import ConfigError, ValidationError
from ..utils.validators import validate_input_shape


class Expert:
    """
    Vectorized expert neuron for batch processing.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: Literal["relu"] = "relu",
        init_method: Literal["he", "xavier", "normal"] = "he",
        dropout_rate: float = 0.0
    ) -> None:
        # Validate parameters
        if input_dim <= 0:
            raise ConfigError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ConfigError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ConfigError(f"output_dim must be positive, got {output_dim}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ConfigError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.init_method = init_method
        self.dropout_rate = dropout_rate
        
        # Initialize weights and biases
        self._initialize_parameters()
        
    def _initialize_parameters(self) -> None:
        """Initialize network parameters using specified method."""
        if self.init_method == "he":
            # He initialization (good for ReLU)
            self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)
            self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2.0 / self.hidden_dim)
        elif self.init_method == "xavier":
            # Xavier initialization
            self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(1.0 / self.input_dim)
            self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(1.0 / self.hidden_dim)
        elif self.init_method == "normal":
            # Standard normal initialization
            self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.1
            self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * 0.1
        else:
            raise ConfigError(f"Unknown initialization method: {self.init_method}")
        
        # Initialize biases to zero
        self.b1 = np.zeros((1, self.hidden_dim), dtype=np.float32)
        self.b2 = np.zeros((1, self.output_dim), dtype=np.float32)
        
        # Ensure float32 dtype for all parameters
        self.W1 = self.W1.astype(np.float32)
        self.W2 = self.W2.astype(np.float32)
        
    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Forward pass through the expert network.
        """
        # Validate input
        validate_input_shape(x, expected_features=self.input_dim)
        
        # First layer: linear transformation + activation
        z1 = x @ self.W1 + self.b1
        a1 = self._apply_activation(z1)
        
        # Second layer: linear transformation
        z2 = a1 @ self.W2 + self.b2
        
        return z2
    
    def __call__(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Make the expert callable."""
        return self.forward(x)
    
    def _apply_activation(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply activation function."""
        if self.activation == "relu":
            return np.maximum(0, x)
        else:
            raise ConfigError(f"Unknown activation: {self.activation}")
    
    def get_activations(self, x: NDArray[np.float32]) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Get intermediate activations for analysis.
        """
        validate_input_shape(x, expected_features=self.input_dim)
        
        # First layer
        z1 = x @ self.W1 + self.b1
        a1 = self._apply_activation(z1)
        
        # Second layer
        z2 = a1 @ self.W2 + self.b2
        
        return a1, z2
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return (
            self.W1.size + self.b1.size + 
            self.W2.size + self.b2.size
        )
    
    def get_parameter_stats(self) -> dict:
        """Get statistics about network parameters."""
        return {
            'W1_mean': float(np.mean(self.W1)),
            'W1_std': float(np.std(self.W1)),
            'W2_mean': float(np.mean(self.W2)),
            'W2_std': float(np.std(self.W2)),
            'b1_mean': float(np.mean(self.b1)),
            'b2_mean': float(np.mean(self.b2)),
            'total_params': self.count_parameters()
        }
    
    def get_config(self) -> dict:
        """Get expert configuration."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'activation': self.activation,
            'init_method': self.init_method,
            'dropout_rate': self.dropout_rate
        }
    
    def reset_parameters(self) -> None:
        """Reset all parameters to new random values."""
        self._initialize_parameters()
    
    def get_weight_norms(self) -> dict:
        """Get L2 norms of weight matrices."""
        return {
            'W1_norm': float(np.linalg.norm(self.W1)),
            'W2_norm': float(np.linalg.norm(self.W2))
        }
    
    def apply_weight_decay(self, decay_rate: float = 0.01) -> None:
        """
        Apply weight decay to parameters.
        """
        if not 0.0 <= decay_rate <= 1.0:
            raise ConfigError(f"decay_rate must be in [0, 1], got {decay_rate}")
        
        self.W1 *= (1.0 - decay_rate)
        self.W2 *= (1.0 - decay_rate)
    
    def get_sparsity_stats(self) -> dict:
        """Get sparsity statistics for weights."""
        w1_sparsity = float(np.count_nonzero(np.abs(self.W1) < 1e-6) / self.W1.size)
        w2_sparsity = float(np.count_nonzero(np.abs(self.W2) < 1e-6) / self.W2.size)
        
        return {
            'W1_sparsity': w1_sparsity,
            'W2_sparsity': w2_sparsity,
            'overall_sparsity': (w1_sparsity + w2_sparsity) / 2.0
        }
