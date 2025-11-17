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


def benchmark_simulation_run(runner, t_final):
    """
    Benchmark the total time for a complete simulation run.
    
    Args:
        runner: SimulationRunner instance
        t_final: The final simulation time (in seconds).
        
    Returns:
        dict: Benchmark results with timing statistics.
    """
    print(f"⏱️  Running benchmark for a simulation of {t_final} seconds...")
    
    # Warmup (optional, but good practice for JIT compilation)
    # A very short run can ensure kernels are compiled.
    print("🔥 Warming up kernels with a very short run...")
    try:
        runner.run(timeout=1.0) # Run for 1 simulation second to warm up
    except Exception as e:
        # This might fail if the timeout is too short for even one step, which is fine.
        print(f"   (Warmup run interrupted as expected: {e})")
        pass
    
    cuda.synchronize()
    
    # Main benchmark run
    start_time = time.perf_counter()
    results_dict = runner.run(timeout=None) # Run to completion based on config t_final
    cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time_s = end_time - start_time
    total_steps = results_dict.get('total_steps', 0)
    
    if total_steps == 0:
        raise RuntimeError("Simulation reported 0 steps. Cannot calculate performance.")

    results = {
        'total_time_s': total_time_s,
        'total_steps': total_steps,
        'mean_ms_per_step': (total_time_s / total_steps) * 1000,
        'steps_per_sec': total_steps / total_time_s,
    }
    
    return results


def print_results(results, title="Benchmark Results"):
    """Pretty print benchmark results."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Total simulation time: {results['total_time_s']:8.3f} s")
    print(f"  Total steps executed:  {results['total_steps']:8d} steps")
    print(f"  Mean time per step:    {results['mean_ms_per_step']:8.3f} ms")
    print(f"  Throughput:            {results['steps_per_sec']:8.2f} steps/sec")
    print(f"{'='*60}\n")


def calculate_speedup(baseline_ms, optimized_ms):
    """Calculate speedup percentage."""
    speedup = ((baseline_ms - optimized_ms) / baseline_ms) * 100
    return speedup


def main():
    """Main benchmark routine."""
    print("=" * 60)
    print("  GPU Kernel Optimization Performance Benchmark")
    print("  (Corrected Method: Timing full runner.run())")
    print("=" * 60)
    print()
    
    # Configuration
    csv_path = "arz_model/data/fichier_de_travail_corridor_utf8.csv"
    simulation_duration = 60.0 # seconds
    
    print("📋 Configuration:")
    print(f"  CSV file: {csv_path}")
    print(f"  Simulation duration for benchmark: {simulation_duration} seconds")
    print()
    
    # Create simulation configuration
    print("🏗️  Building simulation configuration...")
    config = create_victoria_island_config(
        csv_path=csv_path,
        t_final=simulation_duration,
        output_dt=5.0,
        cells_per_100m=4,  # dx=25m
        default_density=25.0
    )
    
    # Build network grid
    print("🌐 Building network grid...")
    network_grid = NetworkGrid.from_config(config)
    
    # Create simulation runner
    print("🚀 Initializing simulation runner...")
    runner = SimulationRunner(network_grid, config)
    
    # Run benchmark
    print("\n" + "="*60)
    print("  STARTING BENCHMARK")
    print("="*60 + "\n")
    
    try:
        results = benchmark_simulation_run(runner, t_final=simulation_duration)
        print_results(results, "GPU Optimized Performance")
        
        # Save results to file
        results_file = "arz_model/benchmarks/gpu_optimization_benchmark_results.txt"
        with open(results_file, 'w') as f:
            f.write("GPU Kernel Optimization Benchmark Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Date: 2025-11-17\n")
            f.write(f"Optimizations: Phase 1 (fastmath + division) + Phase 2 (device func)\n\n")
            f.write(f"Total simulation time: {results['total_time_s']:.3f} s\n")
            f.write(f"Total steps executed:  {results['total_steps']} steps\n")
            f.write(f"Mean time per step:    {results['mean_ms_per_step']:.3f} ms\n")
            f.write(f"Throughput:            {results['steps_per_sec']:.2f} steps/sec\n")
        
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
        print("  3. Compare throughput (steps/sec)")
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
