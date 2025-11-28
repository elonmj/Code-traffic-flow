"""
Main script to run a full corridor network simulation.

NOTE: This is a CUDA-accelerated implementation.
It MUST be executed using the `kaggle_runner` to ensure proper GPU environment setup.

Usage:
    # Victoria Island (Lagos) - default
    python kaggle_runner/executor.py --target arz_model/main_network_simulation.py
    
    # Cotonou Vedoko (Benin)
    python kaggle_runner/executor.py --target arz_model/main_network_simulation.py --args "--corridor cotonou"

⚠️ IMPORTANT: GPU/CUDA IMPLEMENTATION - USE KAGGLE RUNNER ⚠️
=============================================================
This script requires CUDA GPU support and should ALWAYS be executed via:

    python -m kaggle_runner.executor --script arz_model/main_network_simulation.py

Do NOT run this script directly on local machines without CUDA GPU.
The kaggle_runner automatically uploads to Kaggle and executes on P100 GPU.

=============================================================

This script demonstrates the complete automated workflow:
1. Uses ConfigFactory to automatically generate network configuration from CSV topology
2. Builds the NetworkGrid from the generated configuration
3. Initializes the SimulationRunner with the NetworkGrid and config
4. Runs the simulation (delegated to NetworkSimulator on GPU)
5. Saves the results dictionary

This is a REUSABLE system - no manual segment-by-segment configuration needed!

SUPPORTED CORRIDORS:
- victoria_island (default): Lagos, Nigeria
- cotonou: Cotonou Vedoko "Triangle de la Mort", Benin
"""
import os
import sys
import pickle
import time
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the ConfigFactory for automated configuration generation
from arz_model.config.config_factory import (
    create_victoria_island_config,
    create_cotonou_vedoko_config
)
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner

# ============================================================================
# CORRIDOR CONFIGURATIONS REGISTRY
# ============================================================================
# Each corridor has its own scenario parameters optimized for its topology.
# The cache system will automatically handle config reuse.

CORRIDOR_CONFIGS = {
    'victoria_island': {
        'factory': create_victoria_island_config,
        'name': 'Victoria Island (Lagos)',
        'scenario': 'Rush Hour Congestion Build-up',
        'params': {
            'default_density': 60.0,   # veh/km - moderate-heavy traffic
            'default_velocity': 35.0,  # km/h - moderate speed
            'inflow_density': 120.0,   # veh/km - heavy entry density
            'inflow_velocity': 15.0,   # km/h - slow entry speed
            't_final': 300.0,          # 5 minutes
            'output_dt': 5.0,          # Output every 5 seconds
            'cells_per_100m': 8        # Good resolution
        }
    },
    'cotonou': {
        'factory': create_cotonou_vedoko_config,
        'name': 'Cotonou Vedoko "Triangle de la Mort" (Benin)',
        'scenario': 'Triangle de la Mort - Rush Hour',
        'params': {
            'default_density': 80.0,   # veh/km - heavy traffic (Cotonou reality)
            'default_velocity': 25.0,  # km/h - slow flow
            'inflow_density': 140.0,   # veh/km - jam density at entries
            'inflow_velocity': 10.0,   # km/h - crawling speed
            't_final': 300.0,          # 5 minutes
            'output_dt': 5.0,          # Output every 5 seconds
            'cells_per_100m': 8        # Good resolution
        }
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run ARZ network simulation for a corridor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Victoria Island (default)
  python main_network_simulation.py
  
  # Run Cotonou Vedoko
  python main_network_simulation.py --corridor cotonou
  
  # List available corridors
  python main_network_simulation.py --list
        """
    )
    parser.add_argument(
        '--corridor', '-c',
        type=str,
        default='victoria_island',
        choices=list(CORRIDOR_CONFIGS.keys()),
        help='Corridor to simulate (default: victoria_island)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available corridors and exit'
    )
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    # Handle --list option
    if args.list:
        print("\n🌍 AVAILABLE CORRIDORS:")
        print("=" * 60)
        for key, cfg in CORRIDOR_CONFIGS.items():
            print(f"  {key:20s} - {cfg['name']}")
            print(f"                       Scenario: {cfg['scenario']}")
        print("=" * 60)
        return
    
    # Get corridor configuration
    corridor_key = args.corridor
    corridor_cfg = CORRIDOR_CONFIGS[corridor_key]
    
    wall_start = time.time()
    
    print("=" * 70, flush=True)
    print(f"=== CORRIDOR: {corridor_cfg['name'].upper()} ===", flush=True)
    print(f"=== SCENARIO: {corridor_cfg['scenario']} ===", flush=True)
    print("=" * 70, flush=True)
    sys.stdout.flush()

    # --- 1. Generate Configuration from CSV Using ConfigFactory ---
    print("\n[PHASE 1] Generating network configuration from topology CSV...", flush=True)
    print("             (Using automated ConfigFactory with CACHE support)", flush=True)
    try:
        # Get the factory function and parameters for this corridor
        factory_fn = corridor_cfg['factory']
        params = corridor_cfg['params']
        
        # The factory reads the CSV and automatically generates the complete config
        # Cache is automatically checked - if params match, cached config is reused!
        config = factory_fn(**params)
        
        print("✅ Network configuration ready (from cache or freshly generated).", flush=True)
        print(f"   Scenario: {corridor_cfg['scenario']}", flush=True)
        print(f"   - Initial density: {params['default_density']} veh/km", flush=True)
        print(f"   - Initial velocity: {params['default_velocity']} km/h", flush=True)
        print(f"   - Inflow density: {params['inflow_density']} veh/km", flush=True)
        print(f"   - Inflow velocity: {params['inflow_velocity']} km/h", flush=True)
    except Exception as e:
        print(f"❌ Error generating configuration: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # --- 2. Build the NetworkGrid from Configuration ---
    print("\n[PHASE 2] Building NetworkGrid from Pydantic config...", flush=True)
    try:
        network_grid = NetworkGrid.from_config(config)
        print("✅ NetworkGrid built successfully.", flush=True)
        print(f"   - Segments: {len(network_grid.segments)} segments", flush=True)
        print(f"   - Nodes: {len(network_grid.nodes)} nodes", flush=True)
    except Exception as e:
        print(f"❌ Error building NetworkGrid: {e}", flush=True)
        return

    # --- 3. Initialize the Simulation Runner ---
    print("\n[PHASE 3] Initializing simulation runner...", flush=True)
    try:
        # The runner now requires both the grid and the config
        runner = SimulationRunner(network_grid=network_grid, simulation_config=config, debug=False)
        print("✅ Simulation runner initialized.", flush=True)
    except Exception as e:
        import traceback
        print(f"❌ Error initializing runner: {e}", flush=True)
        print("\n=== FULL TRACEBACK ===", flush=True)
        traceback.print_exc()
        print("=" * 70, flush=True)
        return

    # --- 4. Run the Simulation ---
    t_final = corridor_cfg['params']['t_final']
    print(f"\n[PHASE 4] Running simulation for {t_final:.0f} seconds...", flush=True)
    print("=" * 70, flush=True)
    wall_start_sim = time.time()
    
    try:
        # The `run` method is now delegated to the NetworkSimulator
        results = runner.run(timeout=None)
        
        wall_end_sim = time.time()
        wall_elapsed = wall_end_sim - wall_start_sim
        
        print("=" * 70, flush=True)
        print("✅ Simulation finished.", flush=True)
        print(f"\n⏱️  TIMING:", flush=True)
        print(f"   Simulation time: {t_final:.0f} s", flush=True)
        print(f"   Wall clock time: {wall_elapsed:.1f} s", flush=True)
        print(f"   Time per sim second: {wall_elapsed/t_final:.3f} s/s", flush=True)
        print(f"   Speedup ratio: {t_final/wall_elapsed:.2f}x", flush=True)
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return

    # --- 5. Save Results ---
    print("\n[PHASE 5] Saving results...", flush=True)
    
    # Generate output filename based on corridor
    result_filename = f"{corridor_key}_simulation_results.pkl"
    
    # Adjust output path for Kaggle environment
    if os.path.exists('/kaggle/working/'):
        output_path = f'/kaggle/working/{result_filename}'
    else:
        output_dir = os.path.join(project_root, 'results')
        output_path = os.path.join(output_dir, result_filename)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'wb') as f:
            # The results object is a dictionary, not a legacy history object
            pickle.dump(results, f)
        print(f"✅ Results saved to {output_path}", flush=True)
        print("\nRun complete.", flush=True)
    except Exception as e:
        print(f"❌ Error saving results: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'=' * 70}", flush=True)
        print(f"=== FATAL ERROR IN MAIN ===", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
