"""
Main script to run a network simulation from CSV data.

This script orchestrates the entire simulation pipeline for a road network
defined in a CSV file. It follows the correct architectural pattern:
1.  Parse the CSV file to create a validated `RoadNetwork` Pydantic model.
2.  Use the `build_simulation_network` factory to construct the simulation-ready
    `NetworkGrid` from the `RoadNetwork` model.
3.  Initialize and run the `SimulationRunner`.
4.  Save the complete simulation history to a file for post-processing.
"""
import os
import sys
import time
import pickle

from arz_model.road_network.parser import parse_csv_to_road_network
from arz_model.road_network.builder import build_simulation_network
from arz_model.simulation.runner import SimulationRunner
from arz_model.config.network_simulation_config import NetworkSimulationConfig
from arz_model.config.grid_config import GridConfig
from arz_model.config.physics_config import PhysicsConfig
from arz_model.config.ic_config import ICConfig, UniformEquilibriumIC
from arz_model.config.bc_config import BoundaryConditionsConfig, InflowBC, OutflowBC, BCState
from arz_model.config.time_config import TimeConfig
from arz_model.io import data_manager

# --- Configuration ---
# This should point to the location of your network data
# Using an absolute path is safest.
current_dir = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(current_dir, 'data', 'fichier_de_travail_corridor_utf8.csv')
OUTPUT_HISTORY_FILE = os.path.join(current_dir, 'results', 'network_simulation_history.pkl')
SIMULATION_TIME_SECONDS = 3600  # 1 hour of simulation

def main():
    """Main function to orchestrate the simulation."""
    print("--- Starting Network Simulation from CSV ---")

    # --- Create a complete simulation configuration ---
    # The `NetworkSimulationConfig` requires all network topology details.
    # These will be populated by the `build_simulation_network` function.
    # We pass the other config objects to the builder, which will then
    # assemble the final, complete `NetworkSimulationConfig`.
    
    # Define the components of the simulation that are NOT topology-dependent
    time_config = TimeConfig(t_final=SIMULATION_TIME_SECONDS, cfl_factor=0.8)
    physics_config = PhysicsConfig()
    ic_config = ICConfig(config=UniformEquilibriumIC(density=0.3))
    bc_config = BoundaryConditionsConfig(
        left=InflowBC(state=BCState(rho_m=0.1, w_m=50.0, rho_c=0.2, w_c=50.0)),
        right=OutflowBC(state=BCState(rho_m=0.1, w_m=50.0, rho_c=0.2, w_c=50.0))
    )
    grid_config = GridConfig()

    # --- 1. Parsing and Building the Network ---
    print(f"1. Parsing network data from: {CSV_FILE_PATH}")
    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ ERROR: CSV file not found at '{CSV_FILE_PATH}'. Please check the path.")
        sys.exit(1)
        
    start_time = time.time()
    road_network_model = parse_csv_to_road_network(CSV_FILE_PATH)
    print(f"   ...Parsing complete. Found {len(road_network_model.nodes)} nodes and {len(road_network_model.links)} links.")
    
    print("2. Building simulation grid from the network model...")
    # The builder function now takes the individual config components
    network_grid, simulation_config = build_simulation_network(
        road_network_model,
        time_config=time_config,
        physics_config=physics_config,
        ic_config=ic_config,
        bc_config=bc_config,
        grid_config=grid_config,
        default_dx=10.0
    )
    print("   ...Simulation grid built successfully.")
    build_time = time.time() - start_time
    print(f"   (Network parsing and building took {build_time:.2f} seconds)")

    # --- 2. Running the Simulation ---
    print(f"\n3. Initializing simulation runner for a duration of {SIMULATION_TIME_SECONDS} seconds.")
    # We pass the fully constructed grid to the runner.
    # The runner will use default parameters unless a config is specified,
    # which is suitable for this network-focused run.
    runner = SimulationRunner(
        simulation_config=simulation_config,
        network_grid=network_grid,
        quiet=False,
        device='cpu'  # Assuming CPU for now, can be parameterized
    )
    
    print("4. Starting simulation run...")
    start_time = time.time()
    # The runner executes the time-stepping loop
    times, states = runner.run()
    run_duration = time.time() - start_time
    print(f"   ...Simulation finished in {run_duration:.2f} seconds.")

    # --- 3. Saving Results for Post-Processing ---
    print(f"\n5. Saving simulation history to: {OUTPUT_HISTORY_FILE}")
    
    # Ensure the results directory exists
    os.makedirs(os.path.dirname(OUTPUT_HISTORY_FILE), exist_ok=True)
    
    # We need to save everything required for visualization:
    # - The history of states ('times' and 'states')
    # - The grid itself, which contains segment and node topology
    # - The model parameters for context
    history_data = {
        "times": times,
        "states": states,
        "grid": runner.network_grid,  # Use network_grid for consistency
        "simulation_config": runner.simulation_config
    }
    
    with open(OUTPUT_HISTORY_FILE, 'wb') as f:
        pickle.dump(history_data, f)
        
    print(f"   ...History saved successfully.")
    print("\n--- Simulation Complete ---")
    print(f"Next step: Run generate_video.py to create the visualization.")

if __name__ == "__main__":
    main()
