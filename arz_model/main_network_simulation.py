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
    UniformIC,
    ICConfig  # Add ICConfig import
)
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner

def create_two_segment_corridor_config() -> NetworkSimulationConfig:
    """
    Creates a Pydantic configuration for a simple two-segment corridor
    with a single node connecting them.
    """
    print("   - Creating Pydantic config for a two-segment corridor...", flush=True)
    
    # Define shared configurations
    time_config = TimeConfig(t_final=1800.0, output_dt=10.0)
    physics_config = PhysicsConfig(
        v_max_m_kmh=100.0,  # Max speed motorcycles: 100 km/h
        v_max_c_kmh=120.0,  # Max speed cars: 120 km/h
        default_road_quality=1.0  # Perfect road quality (0-1 scale)
    )

    # Define Segments with x_min, x_max, N directly (no GridConfig wrapper)
    segment1_config = SegmentConfig(
        id="seg1",
        x_min=0.0,
        x_max=1000.0,
        N=100,
        start_node=None,  # No upstream node (boundary inflow)
        end_node="node1",  # Connects to node1
        initial_conditions=ICConfig(config=UniformIC(density=50.0, velocity=40.0)),  # 50 veh/km, 40 km/h
        boundary_conditions=BoundaryConditionsConfig(
            left=InflowBC(density=50.0, velocity=40.0),  # Inflow from outside: 50 veh/km, 40 km/h
            right=OutflowBC(density=20.0, velocity=50.0) # This will be overridden by the node
        )
    )
    
    segment2_config = SegmentConfig(
        id="seg2",
        x_min=0.0,
        x_max=1000.0,
        N=100,
        start_node="node1",  # Connects from node1
        end_node=None,  # No downstream node (boundary outflow)
        initial_conditions=ICConfig(config=UniformIC(density=20.0, velocity=50.0)),  # 20 veh/km, 50 km/h
        boundary_conditions=BoundaryConditionsConfig(
            left=OutflowBC(density=20.0, velocity=50.0), # This will be overridden by the node
            right=OutflowBC(density=20.0, velocity=50.0) # Outflow at the end of the corridor
        )
    )

    # Define Node connecting the segments
    node1_config = NodeConfig(
        id="node1",
        type="boundary",  # Use "type" not "node_type", value should be "boundary", "signalized", etc.
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
    
    print("   - Pydantic config created.", flush=True)
    return network_config

def main():
    """Main execution function."""
    import sys
    print("=" * 70, flush=True)
    print("=== DÉBUT DU SCRIPT main_network_simulation.py ===", flush=True)
    print("=" * 70, flush=True)
    sys.stdout.flush()
    
    print("======================================================", flush=True)
    print("= Full Network Simulation Execution (GPU-Only/Pydantic) =", flush=True)
    print("======================================================", flush=True)

    # --- 1. Create the Simulation Configuration ---
    print("\n[PHASE 1] Defining simulation configuration...", flush=True)
    try:
        config = create_two_segment_corridor_config()
        print("✅ Network configuration defined successfully.", flush=True)
    except Exception as e:
        print(f"❌ Error creating configuration: {e}", flush=True)
        return

    # --- 2. Build the NetworkGrid from Configuration ---
    print("\n[PHASE 2] Building NetworkGrid from Pydantic config...", flush=True)
    try:
        network_grid = NetworkGrid.from_config(config)
        print("✅ NetworkGrid built successfully.", flush=True)
        print(f"   - Segments: {list(network_grid.segments.keys())}", flush=True)
        print(f"   - Nodes: {list(network_grid.nodes.keys())}", flush=True)
    except Exception as e:
        print(f"❌ Error building NetworkGrid: {e}", flush=True)
        return

    # --- 3. Initialize the Simulation Runner ---
    print("\n[PHASE 3] Initializing simulation runner...", flush=True)
    try:
        # The runner now requires both the grid and the config
        runner = SimulationRunner(network_grid=network_grid, simulation_config=config, debug=True)
        print("✅ Simulation runner initialized.", flush=True)
    except Exception as e:
        print(f"❌ Error initializing runner: {e}", flush=True)
        return

    # --- 4. Run the Simulation ---
    print("\n[PHASE 4] Running simulation for 260 seconds...", flush=True)
    try:
        # The `run` method is now delegated to the NetworkSimulator
        results = runner.run(timeout=260)
        print("✅ Simulation finished.", flush=True)
    except Exception as e:
        print(f"❌ Error during simulation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return

    # --- 5. Save Results ---
    print("\n[PHASE 5] Saving results...", flush=True)
    output_path = os.path.join(project_root, 'results', 'network_simulation_results.pkl')
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
