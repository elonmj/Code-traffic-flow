"""
Main script to run a full network simulation using the new Pydantic config system.

This script demonstrates the GPU-only architecture workflow:
1. Defines a network and simulation parameters using Pydantic models.
2. Builds the `NetworkGrid` from the configuration object.
3. Initializes the `SimulationRunner` with the `NetworkGrid` and config.
4. Runs the simulation (which is delegated to the `NetworkSimulator`).
5. Saves the results dictionary.
"""
import os
import sys
import pickle

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# New imports for Pydantic-based configuration and network building
from arz_model.config import (
    NetworkSimulationConfig,
    TimeConfig,
    PhysicsConfig,
    GridConfig,
    SegmentConfig,
    NodeConfig,
    InitialConditionsConfig,
    BoundaryConditionsConfig,
    InflowBC,
    OutflowBC,
    UniformIC
)
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner

def create_two_segment_corridor_config() -> NetworkSimulationConfig:
    """
    Creates a Pydantic configuration for a simple two-segment corridor
    with a single node connecting them.
    """
    print("   - Creating Pydantic config for a two-segment corridor...")
    
    # Define shared configurations
    time_config = TimeConfig(t_final=1800.0, output_dt=10.0)
    physics_config = PhysicsConfig(
        V_max_m=80.0, V_max_c=60.0,
        default_road_quality=8
    )
    grid_config = GridConfig(N=100, xmin=0.0, xmax=1000.0) # 1km road

    # Define Segments
    segment1_config = SegmentConfig(
        id="seg1",
        grid=grid_config,
        initial_conditions=UniformIC(rho_m=0.05, rho_c=0.08, w_m=15.0, w_c=12.0),
        boundary_conditions=BoundaryConditionsConfig(
            left=InflowBC(state=[0.05, 15.0, 0.08, 12.0]), # Inflow from outside
            right=OutflowBC() # This will be overridden by the node
        )
    )
    
    segment2_config = SegmentConfig(
        id="seg2",
        grid=grid_config,
        initial_conditions=UniformIC(rho_m=0.0, rho_c=0.0, w_m=0.0, w_c=0.0), # Initially empty
        boundary_conditions=BoundaryConditionsConfig(
            left=OutflowBC(), # This will be overridden by the node
            right=OutflowBC() # Outflow at the end of the corridor
        )
    )

    # Define Node connecting the segments
    node1_config = NodeConfig(
        id="node1",
        node_type="simple_merge", # or other types
        incoming_segments=["seg1"],
        outgoing_segments=["seg2"]
    )

    # Assemble the full network configuration
    network_config = NetworkSimulationConfig(
        time=time_config,
        physics=physics_config,
        segments=[segment1_config, segment2_config],
        nodes=[node1_config]
    )
    
    print("   - Pydantic config created.")
    return network_config

def main():
    """Main execution function."""
    print("======================================================")
    print("= Full Network Simulation Execution (GPU-Only/Pydantic) =")
    print("======================================================")

    # --- 1. Create the Simulation Configuration ---
    print("\n[PHASE 1] Defining simulation configuration...")
    try:
        config = create_two_segment_corridor_config()
        print("✅ Network configuration defined successfully.")
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        return

    # --- 2. Build the NetworkGrid from Configuration ---
    print("\n[PHASE 2] Building NetworkGrid from Pydantic config...")
    try:
        network_grid = NetworkGrid.from_config(config)
        print("✅ NetworkGrid built successfully.")
        print(f"   - Segments: {list(network_grid.segments.keys())}")
        print(f"   - Nodes: {list(network_grid.nodes.keys())}")
    except Exception as e:
        print(f"❌ Error building NetworkGrid: {e}")
        return

    # --- 3. Initialize the Simulation Runner ---
    print("\n[PHASE 3] Initializing simulation runner...")
    try:
        # The runner now requires both the grid and the config
        runner = SimulationRunner(network_grid=network_grid, simulation_config=config)
        print("✅ Simulation runner initialized.")
    except Exception as e:
        print(f"❌ Error initializing runner: {e}")
        return

    # --- 4. Run the Simulation ---
    print("\n[PHASE 4] Running simulation...")
    try:
        # The `run` method is now delegated to the NetworkSimulator
        results = runner.run()
        print("✅ Simulation finished.")
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        return

    # --- 5. Save Results ---
    print("\n[PHASE 5] Saving results...")
    output_path = os.path.join(project_root, 'results', 'network_simulation_results.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, 'wb') as f:
            # The results object is a dictionary, not a legacy history object
            pickle.dump(results, f)
        print(f"✅ Results saved to {output_path}")
        print("\nRun complete.")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
