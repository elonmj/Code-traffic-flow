"""
Main entry point for the new road network simulation system.

This script demonstrates the complete workflow:
1.  Parsing the CSV data into a validated `RoadNetwork` object.
2.  Building the simulation-specific network from the `RoadNetwork` model.
3.  (Future) Configuring and running the simulation.
4.  (Future) Analyzing and visualizing the results.

This modular approach ensures a clear separation of concerns, making the system
more maintainable and extensible.
"""
import sys
import os

# Add parent directory to path to enable absolute imports
project_root = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from arz_model.road_network.parser import parse_csv_to_road_network
from arz_model.road_network.builder import build_simulation_network
from arz_model.core.parameters import ModelParameters

def main():
    """
    Main execution function.
    """
    print("=====================================================")
    print("      Road Network Simulation Genesis")
    print("=====================================================")

    # --- 1. Data Loading and Parsing ---
    # The CSV file is expected to be in the same directory as this script
    # for simplicity. In a real application, this path would be configurable.
    try:
        # Construct the absolute path to the CSV file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file_path = os.path.join(current_dir, 'fichier_de_travail_corridor_utf8.csv')
        
        print(f"\n[PHASE 1] Parsing data from: {csv_file_path}")
        
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"The file was not found at the specified path: {csv_file_path}")

        road_network = parse_csv_to_road_network(csv_file_path)
        
        print(f"✅ Success! Parsed {len(road_network.nodes)} nodes and {len(road_network.links)} links.")
        print(f"   - Example Link: {road_network.links[0].name} (Length: {road_network.links[0].length_m:.2f}m)")
        print(f"   - Example Node ID: {list(road_network.nodes.keys())[0]}")

    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("   Please ensure the CSV file is in the same directory as this script.")
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred during parsing: {e}")
        return

    # --- 2. Simulation Building ---
    print("\n[PHASE 2] Building simulation environment")
    try:
        # Create default model parameters
        # We need to provide at least the ghost_cells parameter
        params = ModelParameters()
        params.ghost_cells = 3  # Standard for WENO5
        params.num_ghost_cells = 3
        
        # Build the simulation network
        simulation_network = build_simulation_network(
            road_network=road_network,
            default_dx=5.0
        )
        
        print(f"✅ Success! Built simulation network with:")
        print(f"   - {len(simulation_network.segments)} segments")
        print(f"   - {len(simulation_network.nodes)} nodes/junctions")
        
        # Initialize the network
        print("\n   Initializing network topology...")
        simulation_network.initialize()
        print("   ✅ Network initialized successfully")
        
    except Exception as e:
        print(f"❌ An unexpected error occurred during network building: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 3. Simulation Execution (Placeholder) ---
    print("\n[PHASE 3] Running simulation (Not yet implemented)")
    # This is where you would instantiate your `NetworkSimulator` and run it.
    # The configuration would be derived from the parsed data.
    print("   - The simulation will run using the adaptive timestepping we fixed.")

    # --- 4. Results Analysis (Placeholder) ---
    print("\n[PHASE 4] Analyzing results (Not yet implemented)")
    # Post-simulation analysis would go here.
    print("   - This will involve loading .npz files and generating plots, similar to `analyze_results.py`.")

    print("\n=====================================================")
    print("         Workflow Demonstration Complete")
    print("=====================================================")


if __name__ == "__main__":
    main()
