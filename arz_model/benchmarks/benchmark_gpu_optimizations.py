"""
GPU Kernel Optimization Performance Benchmark
==============================================

This script benchmarks the performance improvements from Phase 1 and Phase 2
GPU optimizations:
- Phase 1.1: fastmath=True on all kernels
- Phase 1.2: WENO division optimization and constant memory
- Phase 2.1-2.2: Device function fastmath optimization

Expected improvement: 12-30% speedup (conservative estimate)
Target: 30-50% speedup

Usage:
    python arz_model/benchmarks/benchmark_gpu_optimizations.py
    
Author: GPU Optimization Task (2025-11-17)
"""

import time
import numpy as np
import cupy as cp
from numba import cuda
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from arz_model.config.config_factory import create_victoria_island_config
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner


def benchmark_simulation_step(runner, num_steps=100, warmup_steps=10):
    """
    Benchmark the average time per simulation step.
    
    Args:
        runner: SimulationRunner instance
        num_steps: Number of steps to benchmark
        warmup_steps: Number of warmup steps to skip
        
    Returns:
        dict: Benchmark results with timing statistics
    """
    print(f"🔥 Warming up with {warmup_steps} steps...")
    
    # Warmup to compile kernels
    for _ in range(warmup_steps):
        runner.step()
    
    cuda.synchronize()  # Ensure all GPU work is complete
    
    print(f"⏱️  Running {num_steps} benchmark steps...")
    step_times = []
    
    for i in range(num_steps):
        start = time.perf_counter()
        runner.step()
        cuda.synchronize()  # Wait for GPU to finish
        end = time.perf_counter()
        
        step_time_ms = (end - start) * 1000
        step_times.append(step_time_ms)
        
        if (i + 1) % 20 == 0:
            print(f"  Step {i+1}/{num_steps}: {step_time_ms:.3f} ms")
    
    results = {
        'mean_ms': np.mean(step_times),
        'median_ms': np.median(step_times),
        'std_ms': np.std(step_times),
        'min_ms': np.min(step_times),
        'max_ms': np.max(step_times),
        'total_time_s': np.sum(step_times) / 1000,
        'steps_per_sec': 1000 / np.mean(step_times),
    }
    
    return results


def print_results(results, title="Benchmark Results"):
    """Pretty print benchmark results."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Mean step time:      {results['mean_ms']:8.3f} ms")
    print(f"  Median step time:    {results['median_ms']:8.3f} ms")
    print(f"  Std deviation:       {results['std_ms']:8.3f} ms")
    print(f"  Min step time:       {results['min_ms']:8.3f} ms")
    print(f"  Max step time:       {results['max_ms']:8.3f} ms")
    print(f"  Total time:          {results['total_time_s']:8.3f} s")
    print(f"  Throughput:          {results['steps_per_sec']:8.2f} steps/sec")
    print(f"{'='*60}\n")


def calculate_speedup(baseline_ms, optimized_ms):
    """Calculate speedup percentage."""
    speedup = ((baseline_ms - optimized_ms) / baseline_ms) * 100
    return speedup


def main():
    """Main benchmark routine."""
    print("=" * 60)
    print("  GPU Kernel Optimization Performance Benchmark")
    print("  Phase 1 + Phase 2 (Partial) Optimizations")
    print("=" * 60)
    print()
    
    # Configuration
    csv_path = "arz_model/data/fichier_de_travail_corridor_utf8.csv"
    
    print("📋 Configuration:")
    print(f"  CSV file: {csv_path}")
    print(f"  Simulation time: 30.0 seconds")
    print(f"  Benchmark steps: 100")
    print(f"  Warmup steps: 10")
    print()
    
    # Create simulation configuration
    print("🏗️  Building simulation configuration...")
    config_factory = create_victoria_island_config(
        csv_path=csv_path,
        t_final=30.0,
        output_dt=5.0,
        cells_per_100m=4,  # dx=25m
        default_density=25.0
    )
    
    # Build network grid
    print("🌐 Building network grid...")
    network_grid = NetworkGrid.from_config(config_factory)
    
    # Create simulation runner
    print("🚀 Initializing simulation runner...")
    runner = SimulationRunner(network_grid, config_factory)
    
    # Run benchmark
    print("\n" + "="*60)
    print("  STARTING BENCHMARK")
    print("="*60 + "\n")
    
    try:
        results = benchmark_simulation_step(runner, num_steps=100, warmup_steps=10)
        print_results(results, "GPU Optimized Performance")
        
        # Save results to file
        results_file = "arz_model/benchmarks/gpu_optimization_benchmark_results.txt"
        with open(results_file, 'w') as f:
            f.write("GPU Kernel Optimization Benchmark Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Date: 2025-11-17\n")
            f.write(f"Optimizations: Phase 1 (fastmath + division) + Phase 2 (device func)\n\n")
            f.write(f"Mean step time:      {results['mean_ms']:.3f} ms\n")
            f.write(f"Median step time:    {results['median_ms']:.3f} ms\n")
            f.write(f"Std deviation:       {results['std_ms']:.3f} ms\n")
            f.write(f"Min step time:       {results['min_ms']:.3f} ms\n")
            f.write(f"Max step time:       {results['max_ms']:.3f} ms\n")
            f.write(f"Total time:          {results['total_time_s']:.3f} s\n")
            f.write(f"Throughput:          {results['steps_per_sec']:.2f} steps/sec\n")
        
        print(f"✅ Results saved to: {results_file}")
        
        # Performance analysis
        print("\n" + "="*60)
        print("  PERFORMANCE ANALYSIS")
        print("="*60)
        print("\n📊 Optimization Summary:")
        print("  ✅ Phase 1.1: fastmath=True (8 kernels)")
        print("  ✅ Phase 1.2: WENO division reduction (85%)")
        print("  ✅ Phase 2.1: Device function fastmath")
        print("  ✅ Phase 2.2: Coupling kernel integration")
        print("  ⏸️  Phase 2.3: SSP-RK3 fusion (deferred)")
        print()
        print("💡 Expected Speedup: 12-30% (conservative)")
        print("🎯 Target Speedup: 30-50%")
        print()
        print("📝 To compare with baseline:")
        print("  1. Revert GPU optimizations (git checkout baseline)")
        print("  2. Run this benchmark again")
        print("  3. Compare mean step times")
        print()
        print("✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
