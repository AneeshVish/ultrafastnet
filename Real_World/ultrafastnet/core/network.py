"""
Main PacketNet implementation - the core ultra-fast packet-switching neural network.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from numpy.typing import NDArray

from .router import Router
from .expert import Expert
from .aggregator import Aggregator
from ..utils.validators import validate_input_shape, validate_config
from ..utils.exceptions import ConfigError, ValidationError


class PacketNet:
    """
    Production-ready ultra-fast packet-switching neural network.
    
    This class implements a packet-switching neural network that divides input
    into blocks, routes each block to specialized expert neurons, and aggregates
    the outputs for final prediction.
    """
    def __init__(
        self,
        input_dim: int,
        n_blocks: int = 8,
        n_neurons: int = 6,
        hidden_dim: int = 32,
        neuron_output_dim: int = 16,
        final_output_dim: int = 1,
        routing_mode: str = "frequency_hash",
        random_seed: Optional[int] = None
    ) -> None:
        # Validate configuration
        validate_config(
            input_dim=input_dim,
            n_blocks=n_blocks,
            n_neurons=n_neurons,
            hidden_dim=hidden_dim,
            neuron_output_dim=neuron_output_dim,
            final_output_dim=final_output_dim
        )
        
        if input_dim % n_blocks != 0:
            raise ConfigError(
                f"input_dim ({input_dim}) must be divisible by n_blocks ({n_blocks})"
            )
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Store configuration
        self.input_dim = input_dim
        self.n_blocks = n_blocks
        self.n_neurons = n_neurons
        self.hidden_dim = hidden_dim
        self.neuron_output_dim = neuron_output_dim
        self.final_output_dim = final_output_dim
        self.routing_mode = routing_mode
        
        # Calculate block size
        self.block_size = input_dim // n_blocks
        
        # Initialize components
        self._initialize_components()
        
        # Statistics from last forward pass
        self.stats: Dict[str, float] = {}
        
    def _initialize_components(self) -> None:
        """Initialize routers, experts, and aggregator."""
        # Create routers for each block
        self.routers: List[Router] = [
            Router(
                input_dim=self.block_size,
                n_neurons=self.n_neurons,
                mode=self.routing_mode
            )
            for _ in range(self.n_blocks)
        ]
        
        # Create expert neurons
        self.experts: List[Expert] = [
            Expert(
                input_dim=self.block_size,
                hidden_dim=self.hidden_dim,
                output_dim=self.neuron_output_dim
            )
            for _ in range(self.n_neurons)
        ]
        
        # Create output aggregator
        self.aggregator = Aggregator(
            n_neurons=self.n_neurons,
            input_dim=self.neuron_output_dim,
            output_dim=self.final_output_dim
        )
        
    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Forward pass through the packet-switching network.
        """
        # Validate input
        validate_input_shape(x, expected_features=self.input_dim)
        
        batch_size = x.shape[0]
        
        # Storage for outputs and routing probabilities
        all_outputs: List[NDArray[np.float32]] = []
        all_routing_probs: List[NDArray[np.float32]] = []
        
        # Process each block
        for block_idx in range(self.n_blocks):
            # Extract block
            start_idx = block_idx * self.block_size
            end_idx = start_idx + self.block_size
            block_input = x[:, start_idx:end_idx]
            
            # Route block to neurons
            routing_probs = self.routers[block_idx](block_input)
            all_routing_probs.append(routing_probs)
            
            # Process block through all experts
            expert_outputs = np.zeros(
                (batch_size, self.n_neurons, self.neuron_output_dim),
                dtype=np.float32
            )
            
            for neuron_idx, expert in enumerate(self.experts):
                expert_outputs[:, neuron_idx] = expert(block_input)
            
            # Weight expert outputs by routing probabilities
            weighted_outputs = expert_outputs * routing_probs[:, :, np.newaxis]
            all_outputs.append(weighted_outputs)
        
        # Combine outputs from all blocks
        total_outputs = np.sum(all_outputs, axis=0)
        total_routing_probs = np.mean(all_routing_probs, axis=0)
        
        # Aggregate final output
        final_output = self.aggregator(total_outputs, total_routing_probs)
        
        # Update statistics
        self._update_stats(total_routing_probs)
        
        return final_output
    
    def __call__(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Make the network callable."""
        return self.forward(x)
    
    def _update_stats(self, routing_probs: NDArray[np.float32]) -> None:
        """Update performance statistics."""
        mean_probs = routing_probs.mean(axis=0)
        
        # Utilization: fraction of neurons with significant activation
        utilization = np.count_nonzero(mean_probs > 1e-3) / self.n_neurons
        
        # Entropy: measure of routing diversity
        entropy = -(routing_probs * np.log(routing_probs + 1e-9)).sum(axis=1).mean()
        
        # Efficiency: 1 - coefficient of variation of neuron usage
        efficiency = 1.0 - np.std(mean_probs) / (np.mean(mean_probs) + 1e-8)
        
        self.stats = {
            'utilization': float(utilization),
            'entropy': float(entropy),
            'efficiency': float(efficiency)
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get network configuration."""
        return {
            'input_dim': self.input_dim,
            'n_blocks': self.n_blocks,
            'n_neurons': self.n_neurons,
            'hidden_dim': self.hidden_dim,
            'neuron_output_dim': self.neuron_output_dim,
            'final_output_dim': self.final_output_dim,
            'routing_mode': self.routing_mode,
            'block_size': self.block_size
        }
    
    def summary(self) -> str:
        """Get a summary of the network architecture."""
        config = self.get_config()
        total_params = self._count_parameters()
        
        summary = f"""
PacketNet Architecture Summary
==============================
Input Dimension: {config['input_dim']}
Blocks: {config['n_blocks']} (size: {config['block_size']} each)
Expert Neurons: {config['n_neurons']}
Hidden Dimension: {config['hidden_dim']}
Neuron Output Dimension: {config['neuron_output_dim']}
Final Output Dimension: {config['final_output_dim']}
Routing Mode: {config['routing_mode']}

Total Parameters: {total_params:,}
==============================
        """
        return summary.strip()
    
    def _count_parameters(self) -> int:
        """Count total number of parameters in the network."""
        total = 0
        
        # Expert parameters
        for expert in self.experts:
            total += expert.count_parameters()
        
        # Aggregator parameters
        total += self.aggregator.count_parameters()
        
        # Router parameters (minimal)
        for router in self.routers:
            total += router.count_parameters()
        
        return total
