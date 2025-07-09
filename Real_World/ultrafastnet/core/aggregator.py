"""
Output aggregation with attention mechanism and deduplication.
"""

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray

from ..utils.exceptions import ConfigError, ValidationError
from ..utils.validators import validate_input_shape


class Aggregator:
    """
    Ultra-fast output aggregator with vectorized deduplication and attention.
    """
    def __init__(
        self,
        n_neurons: int,
        input_dim: int,
        output_dim: int,
        dedup_threshold: float = 0.95,
        attention_hidden_dim: int = 8
    ) -> None:
        # Validate parameters
        if n_neurons <= 0:
            raise ConfigError(f"n_neurons must be positive, got {n_neurons}")
        if input_dim <= 0:
            raise ConfigError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ConfigError(f"output_dim must be positive, got {output_dim}")
        if not 0.0 <= dedup_threshold <= 1.0:
            raise ConfigError(f"dedup_threshold must be in [0, 1], got {dedup_threshold}")
        if attention_hidden_dim <= 0:
            raise ConfigError(f"attention_hidden_dim must be positive, got {attention_hidden_dim}")
        
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dedup_threshold = dedup_threshold
        self.attention_hidden_dim = attention_hidden_dim
        
        # Initialize parameters
        self._initialize_parameters()
        
    def _initialize_parameters(self) -> None:
        """Initialize attention and output transformation parameters."""
        # Attention mechanism parameters
        self.attention_W = np.random.randn(self.input_dim, self.attention_hidden_dim) * 0.1
        self.attention_b = np.zeros((1, self.attention_hidden_dim))
        self.attention_v = np.random.randn(self.attention_hidden_dim, 1) * 0.1
        
        # Output transformation parameters
        self.output_W = np.random.randn(self.input_dim, self.output_dim) * 0.1
        self.output_b = np.zeros((1, self.output_dim))
        
        # Ensure float32 dtype
        self.attention_W = self.attention_W.astype(np.float32)
        self.attention_b = self.attention_b.astype(np.float32)
        self.attention_v = self.attention_v.astype(np.float32)
        self.output_W = self.output_W.astype(np.float32)
        self.output_b = self.output_b.astype(np.float32)
        
    def forward(
        self, 
        outputs: NDArray[np.float32], 
        routing_probs: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Aggregate expert outputs with attention and deduplication.
        """
        # Validate inputs
        if outputs.ndim != 3:
            raise ValidationError(f"outputs must be 3D tensor, got {outputs.ndim}D")
        if routing_probs.ndim != 2:
            raise ValidationError(f"routing_probs must be 2D tensor, got {routing_probs.ndim}D")
        
        batch_size, n_neurons, feature_dim = outputs.shape
        
        if n_neurons != self.n_neurons:
            raise ValidationError(f"Expected {self.n_neurons} neurons, got {n_neurons}")
        if feature_dim != self.input_dim:
            raise ValidationError(f"Expected {self.input_dim} features, got {feature_dim}")
        if routing_probs.shape != (batch_size, n_neurons):
            raise ValidationError(f"routing_probs shape mismatch: expected {(batch_size, n_neurons)}, got {routing_probs.shape}")
        
        # Step 1: Apply deduplication mask
        dedup_mask = self._compute_deduplication_mask(outputs)
        
        # Step 2: Apply deduplication to routing probabilities
        masked_probs = routing_probs * dedup_mask
        
        # Step 3: Renormalize probabilities
        prob_sums = masked_probs.sum(axis=1, keepdims=True)
        masked_probs = masked_probs / (prob_sums + 1e-8)
        
        # Step 4: Compute attention weights
        attention_weights = self._compute_attention_weights(outputs)
        
        # Step 5: Combine routing probabilities with attention
        combined_weights = attention_weights * masked_probs
        
        # Step 6: Aggregate outputs using combined weights
        aggregated_features = np.einsum('bn,bnd->bd', combined_weights, outputs)
        
        # Step 7: Apply final linear transformation
        final_output = aggregated_features @ self.output_W + self.output_b
        
        return final_output
    
    def __call__(
        self, 
        outputs: NDArray[np.float32], 
        routing_probs: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Make the aggregator callable."""
        return self.forward(outputs, routing_probs)
    
    def _compute_deduplication_mask(self, outputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Compute deduplication mask to remove similar outputs.
        """
        batch_size, n_neurons, feature_dim = outputs.shape
        
        # Initialize mask to keep all outputs
        mask = np.ones((batch_size, n_neurons), dtype=bool)
        
        # Compute normalized outputs for similarity comparison
        norms = np.linalg.norm(outputs, axis=2, keepdims=True) + 1e-8
        normalized_outputs = outputs / norms
        
        # Compute pairwise similarities
        similarities = np.einsum('bnd,bmd->bnm', normalized_outputs, normalized_outputs)
        
        # For each batch item, find and mask duplicates
        for b in range(batch_size):
            # Find high similarity pairs (excluding self-similarity)
            high_sim = similarities[b] > self.dedup_threshold
            np.fill_diagonal(high_sim, False)
            
            # Keep only upper triangle to avoid double-counting
            high_sim = np.triu(high_sim, k=1)
            
            # Mark the second element of each duplicate pair for removal
            duplicates = np.any(high_sim, axis=0)
            mask[b, duplicates] = False
        
        return mask.astype(np.float32)
    
    def _compute_attention_weights(self, outputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Compute attention weights for each expert output.
        """
        batch_size, n_neurons, _ = outputs.shape
        
        # Reshape for matrix multiplication
        outputs_reshaped = outputs.reshape(-1, self.input_dim)
        
        # Compute attention features
        attention_features = np.maximum(0, outputs_reshaped @ self.attention_W + self.attention_b)
        
        # Compute attention scores
        attention_scores = attention_features @ self.attention_v
        attention_scores = attention_scores.reshape(batch_size, n_neurons)
        
        # Apply softmax to get attention weights
        attention_scores = attention_scores - attention_scores.max(axis=1, keepdims=True)
        attention_weights = np.exp(attention_scores)
        attention_weights = attention_weights / (attention_weights.sum(axis=1, keepdims=True) + 1e-8)
        
        return attention_weights
    
    def get_attention_stats(self, outputs: NDArray[np.float32]) -> dict:
        """
        Get statistics about attention weights.
        """
        attention_weights = self._compute_attention_weights(outputs)
        return {
            'mean_attention': float(np.mean(attention_weights)),
            'std_attention': float(np.std(attention_weights)),
            'max_attention': float(np.max(attention_weights)),
            'min_attention': float(np.min(attention_weights)),
            'attention_entropy': float(-(attention_weights * np.log(attention_weights + 1e-9)).sum(axis=1).mean())
        }
    
    def get_deduplication_stats(self, outputs: NDArray[np.float32]) -> dict:
        """
        Get statistics about deduplication.
        """
        mask = self._compute_deduplication_mask(outputs)
        return {
            'kept_ratio': float(np.mean(mask)),
            'removed_ratio': float(1.0 - np.mean(mask)),
            'kept_per_batch': float(np.mean(mask.sum(axis=1)))
        }
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return (
            self.attention_W.size + self.attention_b.size + self.attention_v.size +
            self.output_W.size + self.output_b.size
        )
    
    def get_config(self) -> dict:
        """Get aggregator configuration."""
        return {
            'n_neurons': self.n_neurons,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'dedup_threshold': self.dedup_threshold,
            'attention_hidden_dim': self.attention_hidden_dim
        }
    
    def reset_parameters(self) -> None:
        """Reset all parameters to new random values."""
        self._initialize_parameters()
