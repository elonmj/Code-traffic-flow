"""
GPU Kernel Optimization Performance Benchmark - BASELINE
=======================================================

This script benchmarks the performance of the baseline GPU kernel optimizations.
No actual optimizations are applied in this run; it serves as a performance reference.

Expected improvement from future optimizations: 12-30% speedup (conservative estimate)
Target: 30-50% speedup

Usage:
    python arz_model/benchmarks/benchmark_gpu_optimizations.py
    
Author: GPU Optimization Task (2025-11-17)
"""

import sys
import time
import numpy as np
from numba import cuda

# Make sure the script can find the arz_model package
sys.path.append(r'd:\Projets\Alibi\Code project')

from arz_model.config import create_victoria_island_config
from arz_model.network import NetworkGrid
from arz_model.simulation import SimulationRunner


def benchmark_simulation_run(runner, t_final):
    """
    Benchmark the total time for a complete simulation run.
    
    Args:
        runner: SimulationRunner instance
        t_final: The final simulation time (in seconds).
        ion run.
    Returns:
        dict: Benchmark results with timing statistics.
    """
    print(f"⏱️  Running benchmark for a simulation of {t_final} seconds...")seconds).
    
    # Warmup (optional, but good practice for JIT compilation)rns:
    # A very short run can ensure kernels are compiled.: Benchmark results with timing statistics.
    print("🔥 Warming up kernels with a very short run...")
    try:nt(f"⏱️  Running benchmark for a simulation of {t_final} seconds...")
        # Create a new runner for the warmup to not affect the main run state
        warmup_runner = SimulationRunner(runner.network_grid, runner.simulation_config)# Warmup (optional, but good practice for JIT compilation)
        warmup_runner.run(timeout=1.0) # Run for 1 simulation second to warm upre kernels are compiled.
    except Exception as e:with a very short run...")
        # This might fail if the timeout is too short for even one step, which is fine.
        print(f"   (Warmup run interrupted as expected: {e})")    runner.run(timeout=1.0) # Run for 1 simulation second to warm up
        pass
        # This might fail if the timeout is too short for even one step, which is fine.
    cuda.synchronize(){e})")
    
    # Main benchmark run
    start_time = time.perf_counter()
    results_dict = runner.run(timeout=None) # Run to completion based on config t_final
    cuda.synchronize()run
    end_time = time.perf_counter()
    eout=None) # Run to completion based on config t_final
    total_time_s = end_time - start_time.synchronize()
    total_steps = results_dict.get('total_steps', 0)
    
    if total_steps == 0:l_time_s = end_time - start_time
        raise RuntimeError("Simulation reported 0 steps. Cannot calculate performance.")t.get('total_steps', 0)

    results = {if total_steps == 0:
        'total_time_s': total_time_s,untimeError("Simulation reported 0 steps. Cannot calculate performance.")
        'total_steps': total_steps,
        'mean_ms_per_step': (total_time_s / total_steps) * 1000,
        'steps_per_sec': total_steps / total_time_s,
    }
    me_s / total_steps) * 1000,
    return resultss,


def print_results(results, title="Benchmark Results"):return results
    """Pretty print benchmark results."""
    print(f"\n{'='*60}")
    print(f"  {title}")def print_results(results, title="Benchmark Results"):
    print(f"{'='*60}")
    print(f"  Total wall time:     {results['total_time_s']:8.3f} s")
    print(f"  Total steps executed:  {results['total_steps']:8d} steps")
    print(f"  Mean time per step:    {results['mean_ms_per_step']:8.3f} ms")
    print(f"  Throughput:            {results['steps_per_sec']:8.2f} steps/sec")mulation time: {results['total_time_s']:8.3f} s")
    print(f"{'='*60}\n")steps")
.3f} ms")
.2f} steps/sec")
def calculate_speedup(baseline_ms, optimized_ms):
    """Calculate speedup percentage."""
    speedup = ((baseline_ms - optimized_ms) / baseline_ms) * 100
    return speedup
 percentage."""
    speedup = ((baseline_ms - optimized_ms) / baseline_ms) * 100
def main():    return speedup
    """Main benchmark routine."""
    print("=" * 60)
    print("  GPU Kernel Performance Benchmark - BASELINE")
    print("  (No Optimizations Applied)")ark routine."""
    print("=" * 60)    print("=" * 60)
    print()    print("  GPU Kernel Optimization Performance Benchmark")
      (Corrected Method: Timing full runner.run())")
    # Configuration
    csv_path = "arz_model/data/fichier_de_travail_corridor_utf8.csv"
    simulation_duration = 60.0 # seconds
    
    print("📋 Configuration:")_model/data/fichier_de_travail_corridor_utf8.csv"
    print(f"  CSV file: {csv_path}")ion_duration = 60.0 # seconds
    print(f"  Simulation duration for benchmark: {simulation_duration} seconds")
    print()guration:")
    
    # Create simulation configurationprint(f"  Simulation duration for benchmark: {simulation_duration} seconds")
    print("🏗️  Building simulation configuration...")
    config = create_victoria_island_config(
        csv_path=csv_path,
        t_final=simulation_duration,configuration...")
        output_dt=5.0,and_config(
        cells_per_100m=4,  # dx=25m_path=csv_path,
        default_density=25.0    t_final=simulation_duration,
    )
    
    # Build network grid
    print("🌐 Building network grid...")
    network_grid = NetworkGrid.from_config(config)
    id
    # Create simulation runner...")
    print("🚀 Initializing simulation runner...")id.from_config(config)
    runner = SimulationRunner(network_grid, config)
    # Create simulation runner
    # Run benchmarkng simulation runner...")
    print("\n" + "="*60)id, config)
    print("  STARTING BASELINE BENCHMARK")
    print("="*60 + "\n")# Run benchmark
    
    try:
        results = benchmark_simulation_run(runner, t_final=simulation_duration)
        print_results(results, "GPU Baseline Performance")
        
        # Save results to fileark_simulation_run(runner, t_final=simulation_duration)
        results_file = "arz_model/benchmarks/gpu_baseline_benchmark_results.txt"PU Optimized Performance")
        with open(results_file, 'w') as f:
            f.write("GPU Kernel Baseline Benchmark Results\n")    # Save results to file
            f.write("=" * 60 + "\n")results_file = "arz_model/benchmarks/gpu_optimization_benchmark_results.txt"
            f.write(f"Date: 2025-11-17\n")
            f.write(f"Optimizations: None\n\n")lts\n")
            f.write(f"Total wall time:       {results['total_time_s']:.3f} s\n")    f.write("=" * 60 + "\n")
            f.write(f"Total steps executed:  {results['total_steps']} steps\n")25-11-17\n")
            f.write(f"Mean time per step:    {results['mean_ms_per_step']:.3f} ms\n") func)\n\n")
            f.write(f"Throughput:            {results['steps_per_sec']:.2f} steps/sec\n")e: {results['total_time_s']:.3f} s\n")
        ']} steps\n")
        print(f"✅ Results saved to: {results_file}")step:    {results['mean_ms_per_step']:.3f} ms\n")
        print("\n✅ Baseline benchmark completed successfully!")   {results['steps_per_sec']:.2f} steps/sec\n")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0
print("  ✅ Phase 1.1: fastmath=True (8 kernels)")
n (85%)")
if __name__ == "__main__":print("  ✅ Phase 2.1: Device function fastmath")
    sys.exit(main())Coupling kernel integration")
