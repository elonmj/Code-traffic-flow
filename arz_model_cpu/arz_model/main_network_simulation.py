"""
Main script to run a full network simulation.

This script orchestrates the entire process:
1. Builds the network grid from a CSV file using the builder.
2. Initializes the SimulationRunner with the built network.
3. Runs the simulation.
4. Saves the results.
"""
import os
import sys
import pickle

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from arz_model.network.network_builder import build_network_from_csv
from arz_model.simulation.runner import SimulationRunner

def main():
    """Main execution function."""
    print("=====================================")
    print("= Full Network Simulation Execution =")
    print("=====================================")

    # --- 1. Build the Network ---
    print("\n[PHASE 1] Building network from CSV...")
    try:
        # Simplified to a direct function call as requested
        network_grid = build_network_from_csv(
            nodes_csv_path=os.path.join(project_root, 'arz_model', 'data', 'nodes.csv'),
            segments_csv_path=os.path.join(project_root, 'arz_model', 'data', 'segments.csv')
        )
        print("✅ Network built successfully.")
    except Exception as e:
        print(f"❌ Error building network: {e}")
        return

    # --- 2. Initialize the Simulation Runner ---
    print("\n[PHASE 2] Initializing simulation runner...")
    try:
        # The runner now takes the network_grid directly
        runner = SimulationRunner(network_grid=network_grid)
        print("✅ Simulation runner initialized.")
    except Exception as e:
        print(f"❌ Error initializing runner: {e}")
        return

    # --- 3. Run the Simulation ---
    print("\n[PHASE 3] Running simulation...")
    try:
        history = runner.run()
        print("✅ Simulation finished.")
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        return

    # --- 4. Save Results ---
    print("\n[PHASE 4] Saving results...")
    output_path = os.path.join(project_root, 'results', 'network_simulation_history.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure results directory exists
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(history, f)
        print(f"✅ Results saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

if __name__ == "__main__":
    main()
