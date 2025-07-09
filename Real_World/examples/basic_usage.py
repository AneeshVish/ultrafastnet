"""
Basic usage example for UltrafastNet.

This example demonstrates how to:
1. Create a PacketNet instance
2. Generate sample data
3. Perform inference
4. Analyze results
"""

import numpy as np
import time
from ultrafastnet import PacketNet


def main():
    """Main function demonstrating basic usage."""
    print("UltrafastNet Basic Usage Example")
    print("=" * 40)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configuration
    input_dim = 128
    batch_size = 100
    n_neurons = 8
    hidden_dim = 64
    
    print(f"Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of neurons: {n_neurons}")
    print(f"  Hidden dimension: {hidden_dim}")
    print()
    
    # Create network
    print("Creating PacketNet...")
    network = PacketNet(
        input_dim=input_dim,
        n_neurons=n_neurons,
        hidden_dim=hidden_dim,
        neuron_output_dim=32,
        final_output_dim=1,
        random_seed=42
    )
    
    print("Network created successfully!")
    print(f"Network summary:")
    print(network.summary())
    print()
    
    # Generate sample data
    print("Generating sample data...")
    X = np.random.randn(batch_size, input_dim).astype(np.float32)
    print(f"Generated data shape: {X.shape}")
    print()
    
    # Perform inference
    print("Performing inference...")
    start_time = time.perf_counter()
    
    predictions = network(X)
    
    end_time = time.perf_counter()
    inference_time = (end_time - start_time) * 1000  # Convert to ms
    
    print(f"Inference completed!")
    print(f"Inference time: {inference_time:.2f}ms")
    print(f"Throughput: {batch_size / (inference_time / 1000):.1f} samples/second")
    print()
    
    # Analyze results
    print("Results Analysis:")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Output range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
    print(f"  Output mean: {np.mean(predictions):.4f}")
    print(f"  Output std: {np.std(predictions):.4f}")
    print()
    
    # Network statistics
    print("Network Statistics:")
    stats = network.stats
    for key, value in stats.items():
        print(f"  {key.capitalize()}: {value:.4f}")
    print()
    
    # Test with different input sizes
    print("Testing with different input sizes...")
    test_sizes = [10, 50, 200, 500]
    
    for size in test_sizes:
        test_X = np.random.randn(size, input_dim).astype(np.float32)
        
        start_time = time.perf_counter()
        test_predictions = network(test_X)
        end_time = time.perf_counter()
        
        test_time = (end_time - start_time) * 1000
        throughput = size / (test_time / 1000)
        
        print(f"  Batch size {size:3d}: {test_time:6.2f}ms, {throughput:8.1f} samples/sec")
    
    print()
    print("Basic usage example completed successfully!")


if __name__ == "__main__":
    main()
