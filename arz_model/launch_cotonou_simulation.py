"""
Main script to run a full Cotonou Vedoko corridor network simulation.

NOTE: This is a CUDA-accelerated implementation.
It MUST be executed using the `kaggle_runner` to ensure proper GPU environment setup.
Usage: python kaggle_runner/executor.py --target arz_model/launch_cotonou_simulation.py

⚠️ IMPORTANT: GPU/CUDA IMPLEMENTATION - USE KAGGLE RUNNER ⚠️
=============================================================
This script requires CUDA GPU support and should ALWAYS be executed via:

    python -m kaggle_runner.executor --script arz_model/launch_cotonou_simulation.py

Do NOT run this script directly on local machines without CUDA GPU.
The kaggle_runner automatically uploads to Kaggle and executes on P100 GPU.

=============================================================

This script demonstrates the complete automated workflow for the NEW Cotonou Corridor:
1. Uses ConfigFactory to automatically generate network configuration from Cotonou CSV topology
2. Builds the NetworkGrid from the generated configuration
3. Initializes the SimulationRunner with the NetworkGrid and config
4. Runs the simulation (delegated to NetworkSimulator on GPU)
5. Saves the results dictionary

This is a REUSABLE system - no manual segment-by-segment configuration needed!
"""
import os
import sys
import pickle
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the new ConfigFactory for automated configuration generation
from arz_model.config.config_factory import create_cotonou_vedoko_config
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner

def main():
    """Main execution function."""
    import sys
    wall_start = time.time()
    
    print("=" * 70, flush=True)
    print("=== DÉBUT DU SCRIPT launch_cotonou_simulation.py ===", flush=True)
    print("=" * 70, flush=True)
    sys.stdout.flush()
    
    print("=" * 70, flush=True)
    print("= COTONOU VÊDOKO CORRIDOR - FULL NETWORK SIMULATION =", flush=True)
    print("=" * 70, flush=True)

    # --- 1. Generate Configuration from CSV Using ConfigFactory ---
    print("\n[PHASE 1] Generating network configuration from topology CSV...", flush=True)
    print("             (Using automated ConfigFactory - NO manual configuration!)", flush=True)
    try:
        # The factory reads the CSV and automatically generates the complete config
        # This replaces all manual segment-by-segment configuration!
        
        config = create_cotonou_vedoko_config(
            # =================================================================
            # SCENARIO: "Triangle de la Mort - Rush Hour"
            # =================================================================
            # This scenario simulates the chaotic traffic at Vêdoko intersection.
            #
            # Initial state: HEAVY congestion (typical Cotonou traffic)
            # Perturbation: MASSIVE inflow from Stade Amitié (match day or rush hour)
            # Expected result: Gridlock formation and spillback
            # =================================================================
            
            # Initial conditions: HEAVY traffic
            default_density=80.0,   # veh/km - heavy traffic
            default_velocity=25.0,  # km/h - slow flow
            
            # Boundary conditions: MASSIVE inflow
            inflow_density=140.0,   # veh/km - jam density at entries
            inflow_velocity=10.0,   # km/h - crawling speed
            
            # Simulation parameters
            t_final=300.0,          # 5 minutes
            output_dt=5.0,          # Output every 5 seconds
            cells_per_100m=8        # Good resolution
        )
        print("✅ Network configuration generated successfully from CSV topology.", flush=True)
        print("   Scenario: Triangle de la Mort - Rush Hour", flush=True)
        print(f"   - Initial density: 80.0 veh/km (heavy traffic)", flush=True)
        print(f"   - Initial velocity: 25.0 km/h (slow flow)", flush=True)
        print(f"   - Inflow density: 140.0 veh/km (jam density)", flush=True)
        print(f"   - Inflow velocity: 10.0 km/h (crawling)", flush=True)
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
    print("\n[PHASE 4] Running simulation for 300 seconds...", flush=True)
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
        print(f"   Simulation time: 300.0 s", flush=True)
        print(f"   Wall clock time: {wall_elapsed:.1f} s", flush=True)
        print(f"   Time per sim second: {wall_elapsed/300.0:.3f} s/s", flush=True)
        print(f"   Speedup ratio: {300.0/wall_elapsed:.2f}x", flush=True)
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return

    # --- 5. Save Results ---
    print("\n[PHASE 5] Saving results...", flush=True)
    
    # Adjust output path for Kaggle environment
    if os.path.exists('/kaggle/working/'):
        # Save directly to the root output directory for the runner to find it
        output_path = '/kaggle/working/cotonou_simulation_results.pkl'
    else:
        output_dir = os.path.join(project_root, 'results')
        output_path = os.path.join(output_dir, 'cotonou_simulation_results.pkl')
        
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
    import sys
    try:
        print("\n" + "=" * 70, flush=True)
        print("=== SCRIPT MAIN STARTING ===", flush=True)
        print("=" * 70 + "\n", flush=True)
        sys.stdout.flush()
        main()
        print("\n" + "=" * 70, flush=True)
        print("=== SCRIPT MAIN COMPLETED ===", flush=True)
        print("=" * 70, flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"\n{'=' * 70}", flush=True)
        print(f"=== FATAL ERROR IN MAIN ===", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
