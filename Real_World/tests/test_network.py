"""
Tests for the main PacketNet class.
"""

import numpy as np
import pytest
from ultrafastnet import PacketNet
from ultrafastnet.utils.exceptions import ConfigError, ValidationError


class TestPacketNet:
    """Test suite for PacketNet class."""
    
    def test_init_default_params(self):
        """Test network initialization with default parameters."""
        network = PacketNet(input_dim=128)
        
        assert network.input_dim == 128
        assert network.n_blocks == 8
        assert network.n_neurons == 6
        assert network.hidden_dim == 32
        assert network.neuron_output_dim == 16
        assert network.final_output_dim == 1
        assert network.block_size == 16
        
    def test_init_custom_params(self):
        """Test network initialization with custom parameters."""
        network = PacketNet(
            input_dim=64,
            n_blocks=4,
            n_neurons=8,
            hidden_dim=128,
            neuron_output_dim=32,
            final_output_dim=2
        )
        
        assert network.input_dim == 64
        assert network.n_blocks == 4
        assert network.n_neurons == 8
        assert network.hidden_dim == 128
        assert network.neuron_output_dim == 32
        assert network.final_output_dim == 2
        assert network.block_size == 16
        
    def test_init_invalid_input_dim(self):
        """Test initialization with invalid input dimension."""
        with pytest.raises(ConfigError):
            PacketNet(input_dim=0)
            
        with pytest.raises(ConfigError):
            PacketNet(input_dim=-10)
            
    def test_init_indivisible_input_dim(self):
        """Test initialization with input_dim not divisible by n_blocks."""
        with pytest.raises(ConfigError):
            PacketNet(input_dim=127, n_blocks=8)  # 127 not divisible by 8
            
    def test_forward_valid_input(self):
        """Test forward pass with valid input."""
        network = PacketNet(input_dim=128, random_seed=42)
        X = np.random.randn(10, 128).astype(np.float32)
        
        output = network(X)
        
        assert output.shape == (10, 1)
        assert output.dtype == np.float32
        assert np.isfinite(output).all()
        
    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        network = PacketNet(input_dim=64, random_seed=42)
        
        batch_sizes = [1, 5, 10, 50, 100]
        
        for batch_size in batch_sizes:
            X = np.random.randn(batch_size, 64).astype(np.float32)
            output = network(X)
            
            assert output.shape == (batch_size, 1)
            assert np.isfinite(output).all()
            
    def test_forward_invalid_input_shape(self):
        """Test forward pass with invalid input shape."""
        network = PacketNet(input_dim=128)
        
        # Wrong feature dimension
        X_wrong_features = np.random.randn(10, 64).astype(np.float32)
        with pytest.raises(ValidationError):
            network(X_wrong_features)
            
        # Wrong number of dimensions
        X_wrong_dims = np.random.randn(10, 128, 1).astype(np.float32)
        with pytest.raises(ValidationError):
            network(X_wrong_dims)
            
    def test_forward_invalid_input_dtype(self):
        """Test forward pass with invalid input dtype."""
        network = PacketNet(input_dim=128)
        
        # Integer input
        X_int = np.random.randint(0, 10, (10, 128))
        with pytest.raises(ValidationError):
            network(X_int)
            
    def test_forward_non_finite_input(self):
        """Test forward pass with non-finite input."""
        network = PacketNet(input_dim=128)
        
        # NaN input
        X_nan = np.random.randn(10, 128).astype(np.float32)
        X_nan[0, 0] = np.nan
        with pytest.raises(ValidationError):
            network(X_nan)
            
        # Inf input
        X_inf = np.random.randn(10, 128).astype(np.float32)
        X_inf[0, 0] = np.inf
        with pytest.raises(ValidationError):
            network(X_inf)
            
    def test_stats_update(self):
        """Test that statistics are updated after forward pass."""
        network = PacketNet(input_dim=128, random_seed=42)
        X = np.random.randn(10, 128).astype(np.float32)
        
        # Initially stats should be empty
        assert network.stats == {}
        
        # After forward pass, stats should be populated
        network(X)
        
        assert 'utilization' in network.stats
        assert 'entropy' in network.stats
        assert 'efficiency' in network.stats
        
        # All stats should be valid numbers
        for key, value in network.stats.items():
            assert isinstance(value, float)
            assert np.isfinite(value)
            
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        X = np.random.randn(10, 128).astype(np.float32)
        
        network1 = PacketNet(input_dim=128, random_seed=42)
        network2 = PacketNet(input_dim=128, random_seed=42)
        
        output1 = network1(X)
        output2 = network2(X)
        
        np.testing.assert_array_almost_equal(output1, output2)
        
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        X = np.random.randn(10, 128).astype(np.float32)
        
        network1 = PacketNet(input_dim=128, random_seed=42)
        network2 = PacketNet(input_dim=128, random_seed=123)
        
        output1 = network1(X)
        output2 = network2(X)
        
        # Results should be different
        assert not np.allclose(output1, output2)
        
    def test_get_config(self):
        """Test get_config method."""
        network = PacketNet(
            input_dim=64,
            n_blocks=4,
            n_neurons=8,
            hidden_dim=128
        )
        
        config = network.get_config()
        
        assert config['input_dim'] == 64
        assert config['n_blocks'] == 4
        assert config['n_neurons'] == 8
        assert config['hidden_dim'] == 128
        assert config['block_size'] == 16
        
    def test_summary(self):
        """Test summary method."""
        network = PacketNet(input_dim=128)
        summary = network.summary()
        
        assert isinstance(summary, str)
        assert 'PacketNet Architecture Summary' in summary
        assert 'Input Dimension: 128' in summary
        assert 'Total Parameters:' in summary
        
    def test_count_parameters(self):
        """Test parameter counting."""
        network = PacketNet(input_dim=128, n_neurons=4, hidden_dim=32)
        param_count = network._count_parameters()
        
        assert isinstance(param_count, int)
        assert param_count > 0
        
    def test_multiple_forward_passes(self):
        """Test multiple forward passes on same network."""
        network = PacketNet(input_dim=128, random_seed=42)
        
        X1 = np.random.randn(10, 128).astype(np.float32)
        X2 = np.random.randn(20, 128).astype(np.float32)
        
        output1 = network(X1)
        output2 = network(X2)
        
        assert output1.shape == (10, 1)
        assert output2.shape == (20, 1)
        
        # Stats should be updated after each forward pass
        assert network.stats != {}
        
    def test_edge_cases(self):
        """Test edge cases."""
        # Minimum input dimension
        network = PacketNet(input_dim=8, n_blocks=8)
        X = np.random.randn(1, 8).astype(np.float32)
        output = network(X)
        assert output.shape == (1, 1)
        
        # Single sample
        network = PacketNet(input_dim=128)
        X = np.random.randn(1, 128).astype(np.float32)
        output = network(X)
        assert output.shape == (1, 1)
        
    def test_large_batch_size(self):
        """Test with large batch size."""
        network = PacketNet(input_dim=128, random_seed=42)
        X = np.random.randn(1000, 128).astype(np.float32)
        
        output = network(X)
        
        assert output.shape == (1000, 1)
        assert np.isfinite(output).all()
        
    @pytest.mark.parametrize("input_dim,n_blocks", [
        (32, 4),
        (64, 8),
        (128, 16),
        (256, 32)
    ])
    def test_various_configurations(self, input_dim, n_blocks):
        """Test various valid configurations."""
        network = PacketNet(input_dim=input_dim, n_blocks=n_blocks)
        X = np.random.randn(10, input_dim).astype(np.float32)
        
        output = network(X)
        
        assert output.shape == (10, 1)
        assert np.isfinite(output).all()
