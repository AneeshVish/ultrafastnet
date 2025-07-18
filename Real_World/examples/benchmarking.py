"""
Benchmarking example for UltrafastNet.

This example demonstrates performance benchmarking and optimization
analysis for the packet-switching neural network.
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from ultrafastnet import PacketNet


def benchmark_network_performance(
    input_dim: int = 128,
    batch_sizes: List[int] = None,
    n_neurons: int = 6,
    hidden_dim: int = 32,
    neuron_output_dim: int = 16,
    n_runs: int = 10
) -> Dict[str, List[float]]:
    """
    Benchmark network performance across different batch sizes.
    """
    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100, 200, 500, 1000]
    
    print("Running Performance Benchmarks")
    print("=" * 50)
    
    # Create network
    network = PacketNet(
        input_dim=input_dim,
        n_neurons=n_neurons,
        hidden_dim=hidden_dim,
        neuron_output_dim=neuron_output_dim,
        final_output_dim=1,
        random_seed=42
    )
    
    print(f"Network Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Expert neurons: {n_neurons}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Total parameters: {network._count_parameters():,}")
    print()
    
    results = {
        'batch_sizes': [],
        'mean_times': [],
        'std_times': [],
        'throughput': [],
        'utilization': [],
        'efficiency': []
    }
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Generate test data
        X = np.random.randn(batch_size, input_dim).astype(np.float32)
        
        # Warm up
        _ = network(X)
        
        # Benchmark
        times = []
        utilizations = []
        efficiencies = []
        
        for run in range(n_runs):
            start_time = time.perf_counter()
            _ = network(X)
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
            utilizations.append(network.stats['utilization'])
            efficiencies.append(network.stats['efficiency'])
        
        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = batch_size / (mean_time / 1000)  # samples/second
        mean_utilization = np.mean(utilizations)
        mean_efficiency = np.mean(efficiencies)
        
        results['batch_sizes'].append(batch_size)
        results['mean_times'].append(mean_time)
        results['std_times'].append(std_time)
        results['throughput'].append(throughput)
        results['utilization'].append(mean_utilization)
        results['efficiency'].append(mean_efficiency)
        
        print(f"  Time: {mean_time:.2f} B{std_time:.2f}ms")
        print(f"  Throughput: {throughput:.1f} samples/second")
        print(f"  Utilization: {mean_utilization:.1%}")
        print(f"  Efficiency: {mean_efficiency:.1%}")
        print()
    
    return results


def compare_configurations() -> None:
    """Compare different network configurations."""
    print("Comparing Network Configurations")
    print("=" * 50)
    
    configs = [
        {"n_neurons": 4, "hidden_dim": 16, "name": "Small"},
        {"n_neurons": 6, "hidden_dim": 32, "name": "Medium"},
        {"n_neurons": 8, "hidden_dim": 64, "name": "Large"},
        {"n_neurons": 12, "hidden_dim": 128, "name": "Extra Large"}
    ]
    
    input_dim = 128
    batch_size = 200
    n_runs = 5
    
    # Generate test data
    X = np.random.randn(batch_size, input_dim).astype(np.float32)
    
    print(f"Test Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of runs: {n_runs}")
    print()
    
    for config in configs:
        print(f"Testing {config['name']} configuration...")
        
        # Create network
        network = PacketNet(
            input_dim=input_dim,
            n_neurons=config['n_neurons'],
            hidden_dim=config['hidden_dim'],
            neuron_output_dim=16,
            final_output_dim=1,
            random_seed=42
        )
        
        # Warm up
        _ = network(X)
        
        # Benchmark
        times = []
        for run in range(n_runs):
            start_time = time.perf_counter()
            _ = network(X)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        mean_time = np.mean(times)
        throughput = batch_size / (mean_time / 1000)
        params = network._count_parameters()
        
        print(f"  Results:")
        print(f"    Parameters: {params:,}")
        print(f"    Time: {mean_time:.2f}ms")
        print(f"    Throughput: {throughput:.1f} samples/second")
        print(f"    Utilization: {network.stats['utilization']:.1%}")
        print(f"    Efficiency: {network.stats['efficiency']:.1%}")
        print()


def analyze_routing_behavior() -> None:
    """Analyze packet routing behavior."""
    print("Analyzing Routing Behavior")
    print("=" * 50)
    
    input_dim = 64
    batch_size = 100
    n_neurons = 8
    
    # Create network
    network = PacketNet(
        input_dim=input_dim,
        n_neurons=n_neurons,
        hidden_dim=32,
        neuron_output_dim=16,
        final_output_dim=1,
        random_seed=42
    )
    
    # Generate different types of test data
    test_cases = [
        ("Random Normal", np.random.randn(batch_size, input_dim)),
        ("Random Uniform", np.random.uniform(-1, 1, (batch_size, input_dim))),
        ("Sine Wave", np.sin(np.linspace(0, 10*np.pi, batch_size*input_dim)).reshape(batch_size, input_dim)),
        ("Constant", np.ones((batch_size, input_dim))),
        ("Sparse", np.random.choice([0, 1], (batch_size, input_dim), p=[0.9, 0.1]))
    ]
    
    print(f"Test Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of neurons: {n_neurons}")
    print()
    
    for name, data in test_cases:
        print(f"Testing with {name} data...")
        
        # Ensure float32 dtype
        data = data.astype(np.float32)
        
        # Process data
        _ = network(data)
        stats = network.stats
        
        # Analyze individual router behavior
        routing_stats = []
        for i, router in enumerate(network.routers):
            block_start = i * network.block_size
            block_end = block_start + network.block_size
            block_data = data[:, block_start:block_end]
            
            routing_probs = router(block_data)
            router_stats = router.get_load_balance_stats(routing_probs)
            routing_stats.append(router_stats)
        
        avg_router_entropy = np.mean([rs['entropy'] for rs in routing_stats])
        avg_router_utilization = np.mean([rs['utilization'] for rs in routing_stats])
        
        print(f"  Network Statistics:")
        print(f"    Overall Utilization: {stats['utilization']:.1%}")
        print(f"    Overall Efficiency: {stats['efficiency']:.1%}")
        print(f"    Overall Entropy: {stats['entropy']:.3f}")
        print(f" Router Statistics:")
        print(f"    Average Entropy: {avg_router_entropy:.3f}")
        print(f"    Average Utilization: {avg_router_utilization:.1%}")
        print()


def main():
    """Main benchmarking function."""
    print("ULTRA-FAST NEURAL NETWORK")
    print("COMPREHENSIVE BENCHMARKING SUITE")
    print("=" * 60)
    print()
    
    # Run performance benchmarks
    benchmark_results = benchmark_network_performance()
    
    # Display summary
    print("Performance Summary:")
    max_throughput = max(benchmark_results['throughput'])
    optimal_batch = benchmark_results['batch_sizes'][
        benchmark_results['throughput'].index(max_throughput)
    ]
    print(f" Maximum throughput: {max_throughput:.1f} samples/second")
    print(f" Optimal batch size: {optimal_batch}")
    print()
    
    # Compare configurations
    compare_configurations()
    
    # Analyze routing behavior
    analyze_routing_behavior()
    
    print("Benchmarking completed successfully!")
    print("UltrafastNet is ready for production use!")


if __name__ == "__main__":
    main()
