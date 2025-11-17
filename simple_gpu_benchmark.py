"""
Simple GPU Optimization Benchmark
==================================

Minimal standalone script to measure GPU kernel performance on Kaggle P100.
This version directly reports results without complex infrastructure.
"""

import time
import sys
import os
import numpy as np
from numba import cuda

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from arz_model.config.config_factory import create_victoria_island_config
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner


def main():
    print("=" * 70)
    print("  GPU KERNEL OPTIMIZATION BENCHMARK - SIMPLIFIED")
    print("  NVIDIA Tesla P100 (Kaggle)")
    print("=" * 70)
    print()
    
    # Configuration
    csv_path = "arz_model/data/fichier_de_travail_corridor_utf8.csv"
    
    print("📋 Configuration:")
    print(f"  CSV: {csv_path}")
    print(f"  Grid: 4 cells/100m (dx=25m)")
    print(f"  Density: 25 veh/km")
    print(f"  Benchmark: 50 time steps")
    print()
    
    try:
        # Create simulation
        print("🏗️  Building simulation...")
        config = create_victoria_island_config(
            csv_path=csv_path,
            t_final=30.0,
            output_dt=5.0,
            cells_per_100m=4,
            default_density=25.0
        )
        
        network_grid = NetworkGrid.from_config(config)
        runner = SimulationRunner(network_grid, config)
        print(f"✅ Simulation ready: {len(network_grid.segments)} segments")
        print()
        
        # Warmup
        print("🔥 Warmup (5 steps)...")
        for _ in range(5):
            runner.step()
        cuda.synchronize()
        print("✅ Warmup complete")
        print()
        
        # Benchmark
        print("⏱️  Benchmarking (50 steps)...")
        step_times = []
        
        for i in range(50):
            t_start = time.perf_counter()
            runner.step()
            cuda.synchronize()
            t_end = time.perf_counter()
            
            step_time_ms = (t_end - t_start) * 1000
            step_times.append(step_time_ms)
            
            if (i + 1) % 10 == 0:
                print(f"  Step {i+1}/50: {step_time_ms:.2f} ms")
        
        print()
        
        # Statistics
        step_times = np.array(step_times)
        mean_ms = np.mean(step_times)
        median_ms = np.median(step_times)
        std_ms = np.std(step_times)
        min_ms = np.min(step_times)
        max_ms = np.max(step_times)
        
        print("=" * 70)
        print("  BENCHMARK RESULTS")
        print("=" * 70)
        print()
        print(f"📊 Step Time Statistics:")
        print(f"  Mean:   {mean_ms:.3f} ms")
        print(f"  Median: {median_ms:.3f} ms")
        print(f"  Std:    {std_ms:.3f} ms")
        print(f"  Min:    {min_ms:.3f} ms")
        print(f"  Max:    {max_ms:.3f} ms")
        print()
        
        throughput = 1000.0 / mean_ms  # steps per second
        print(f"🚀 Throughput: {throughput:.2f} steps/sec")
        print()
        
        # Interpretation
        print("=" * 70)
        print("  INTERPRETATION")
        print("=" * 70)
        print()
        print("ℹ️  Optimizations Applied:")
        print("  ✅ Phase 1.1: fastmath=True on 8 kernels")
        print("  ✅ Phase 1.2: WENO division reduction (85%)")
        print("  ✅ Phase 2.1: Device function fastmath")
        print("  ✅ Phase 2.2: Coupling kernel integration")
        print()
        print("📈 Expected Performance:")
        print("  Conservative: 12-30% speedup")
        print("  Target:       30-50% speedup")
        print()
        
        # Success
        print("=" * 70)
        print("  ✅ BENCHMARK COMPLETE")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
