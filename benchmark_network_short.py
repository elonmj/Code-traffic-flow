"""
Short benchmark script for Victoria Island network simulation.
Runs only 240 seconds of simulation time to quickly measure performance.

This script is identical to main_network_simulation.py but with:
- Reduced t_final: 240s instead of 1800s
- Enhanced progress logging with wall time tracking
- Automatic results saving with timing metrics
"""

import os
import sys
import time
import pickle
import traceback

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from arz_model.config.config_factory import create_victoria_island_config
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner

def main():
    """Main benchmark execution."""
    print("=" * 70, flush=True)
    print("=== BENCHMARK: Victoria Island Network (240s simulation) ===", flush=True)
    print("=" * 70, flush=True)
    
    wall_start_total = time.time()
    
    # --- 1. Generate Configuration ---
    print("\n[PHASE 1] Generating network configuration...", flush=True)
    try:
        config = create_victoria_island_config(
            topology_csv="arz_model/data/fichier_de_travail_corridor_utf8.csv",
            default_density=20.0,
            default_velocity=50.0,
            inflow_density=30.0,
            inflow_velocity=40.0,
            t_final=240.0,  # SHORT BENCHMARK: 4 minutes instead of 30
            output_dt=10.0,
            cells_per_100m=10
        )
        print("✅ Configuration generated (t_final=240s)", flush=True)
    except Exception as e:
        print(f"❌ Error generating configuration: {e}", flush=True)
        traceback.print_exc()
        return

    # --- 2. Build NetworkGrid ---
    print("\n[PHASE 2] Building NetworkGrid...", flush=True)
    try:
        network_grid = NetworkGrid.from_config(config)
        print(f"✅ NetworkGrid built", flush=True)
        print(f"   Segments: {len(network_grid.segments)}", flush=True)
        print(f"   Nodes: {len(network_grid.nodes)}", flush=True)
    except Exception as e:
        print(f"❌ Error building NetworkGrid: {e}", flush=True)
        traceback.print_exc()
        return

    # --- 3. Initialize Simulation Runner ---
    print("\n[PHASE 3] Initializing simulation runner...", flush=True)
    try:
        runner = SimulationRunner(
            network_grid=network_grid,
            simulation_config=config,
            debug=False
        )
        print("✅ Simulation runner initialized", flush=True)
    except Exception as e:
        print(f"❌ Error initializing runner: {e}", flush=True)
        traceback.print_exc()
        return

    # --- 4. Run Simulation with Timing ---
    print("\n[PHASE 4] Running simulation (240s, no timeout)...", flush=True)
    print("=" * 70, flush=True)
    
    wall_start_sim = time.time()
    
    try:
        results = runner.run(timeout=None)
        wall_end_sim = time.time()
        
        wall_elapsed = wall_end_sim - wall_start_sim
        
        print("=" * 70, flush=True)
        print("✅ Simulation finished", flush=True)
        print(f"\n⏱️  BENCHMARK RESULTS:", flush=True)
        print(f"   Simulation time: 240.0 s", flush=True)
        print(f"   Wall clock time: {wall_elapsed:.1f} s", flush=True)
        print(f"   Time per sim second: {wall_elapsed/240.0:.3f} s/s", flush=True)
        print(f"   Speedup ratio: {240.0/wall_elapsed:.2f}x", flush=True)
        
        # Add timing to results
        results['benchmark'] = {
            't_sim': 240.0,
            'wall_time': wall_elapsed,
            'time_per_step': wall_elapsed / 240.0,
            'speedup_ratio': 240.0 / wall_elapsed
        }
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}", flush=True)
        traceback.print_exc()
        return

    # --- 5. Save Results ---
    print("\n[PHASE 5] Saving results...", flush=True)
    
    if os.path.exists('/kaggle/working/'):
        output_path = '/kaggle/working/benchmark_results.pkl'
    else:
        output_dir = os.path.join(project_root, 'results')
        output_path = os.path.join(output_dir, 'benchmark_results.pkl')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"✅ Results saved to {output_path}", flush=True)
        
        # Also save timing summary as text
        summary_path = output_path.replace('.pkl', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("BENCHMARK SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Simulation time: 240.0 s\n")
            f.write(f"Wall clock time: {wall_elapsed:.1f} s\n")
            f.write(f"Time per sim second: {wall_elapsed/240.0:.3f} s/s\n")
            f.write(f"Speedup ratio: {240.0/wall_elapsed:.2f}x\n")
        print(f"✅ Summary saved to {summary_path}", flush=True)
        
    except Exception as e:
        print(f"❌ Error saving results: {e}", flush=True)
        traceback.print_exc()
    
    wall_total = time.time() - wall_start_total
    print(f"\n⏱️  Total wall time (including setup): {wall_total:.1f} s", flush=True)
    print("\n✅ Benchmark complete.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
