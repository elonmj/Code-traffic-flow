import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arz_model.config import (
    NetworkSimulationConfig, TimeConfig, PhysicsConfig, GridConfig,
    SegmentConfig, NodeConfig, ICConfig, UniformIC,
    BoundaryConditionsConfig, InflowBC, OutflowBC
)
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner
from arz_model.core import physics
from arz_model.core.parameters import ModelParameters

def setup_simulation(scenario_name, t_final=30.0):
    # Create a simple 1-segment config
    seg_config = SegmentConfig(
        id="seg-0",
        x_min=0.0,
        x_max=1000.0,
        N=400, # 2.5m cells
        initial_conditions=ICConfig(config=UniformIC(density=20.0, velocity=50.0)), # Dummy
        boundary_conditions=BoundaryConditionsConfig(
            left=InflowBC(density=20.0, velocity=50.0),
            right=OutflowBC(density=20.0, velocity=50.0)
        ),
        start_node="node-0",
        end_node="node-1"
    )
    
    config = NetworkSimulationConfig(
        time=TimeConfig(t_final=t_final, output_dt=0.1),
        physics=PhysicsConfig(),
        grid=GridConfig(num_ghost_cells=3),
        segments=[seg_config],
        nodes=[
            NodeConfig(id="node-0", type="boundary", incoming_segments=[], outgoing_segments=["seg-0"]),
            NodeConfig(id="node-1", type="boundary", incoming_segments=["seg-0"], outgoing_segments=[])
        ]
    )
    
    network_grid = NetworkGrid.from_config(config)
    runner = SimulationRunner(network_grid=network_grid, simulation_config=config, quiet=True)
    return runner, config

def set_riemann_state(runner, rho_L, v_L, rho_R, v_R, split_frac=0.5):
    # rho_L/R: tuple (rho_m, rho_c) in veh/km
    # v_L/R: tuple (v_m, v_c) in km/h
    
    grid = runner.network_grid.segments['seg-0']['grid']
    params = runner.params
    
    # Convert to SI
    rho_m_L = rho_L[0] * physics.VEH_KM_TO_VEH_M
    rho_c_L = rho_L[1] * physics.VEH_KM_TO_VEH_M
    v_m_L = v_L[0] * physics.KMH_TO_MS
    v_c_L = v_L[1] * physics.KMH_TO_MS
    
    rho_m_R = rho_R[0] * physics.VEH_KM_TO_VEH_M
    rho_c_R = rho_R[1] * physics.VEH_KM_TO_VEH_M
    v_m_R = v_R[0] * physics.KMH_TO_MS
    v_c_R = v_R[1] * physics.KMH_TO_MS
    
    # Create arrays
    N = grid.N_total
    mid = int(N * split_frac)
    
    rho_m = np.zeros(N)
    rho_c = np.zeros(N)
    v_m = np.zeros(N)
    v_c = np.zeros(N)
    
    rho_m[:mid] = rho_m_L
    rho_m[mid:] = rho_m_R
    rho_c[:mid] = rho_c_L
    rho_c[mid:] = rho_c_R
    v_m[:mid] = v_m_L
    v_m[mid:] = v_m_R
    v_c[:mid] = v_c_L
    v_c[mid:] = v_c_R
    
    # Calculate Pressure
    p_m, p_c = physics.calculate_pressure(
        rho_m, rho_c, params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c
    )
    
    # Calculate w = v + P
    w_m = v_m + p_m
    w_c = v_c + p_c
    
    # Calculate y = rho * w
    y_m = rho_m * w_m
    y_c = rho_c * w_c
    
    # Set state
    U = np.zeros((4, N))
    U[0] = rho_m
    U[1] = y_m
    U[2] = rho_c
    U[3] = y_c
    
    # Update GPU state
    # Accessing the segment simulator directly
    # Note: runner.network_simulator is likely NetworkCouplingGPU
    # It has a .segments dictionary mapping ID to SegmentSimulatorGPU
    runner.network_simulator.segments['seg-0'].current_state = cuda.to_device(U)
    
    return U

def run_and_plot(runner, filename, title):
    print(f"Running simulation for {title}...")
    # Run
    runner.run()
    
    # Get final state
    final_state = runner.network_simulator.segments['seg-0'].current_state.copy_to_host()
    grid = runner.network_grid.segments['seg-0']['grid']
    params = runner.params
    
    # Extract physical cells
    start = grid.num_ghost_cells
    end = -grid.num_ghost_cells
    
    rho_m = final_state[0, start:end]
    y_m = final_state[1, start:end]
    rho_c = final_state[2, start:end]
    y_c = final_state[3, start:end]
    
    # Calculate velocities for plotting
    p_m, p_c = physics.calculate_pressure(
        rho_m, rho_c, params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c
    )
    
    # v = w - P = (y/rho) - P
    # Handle division by zero
    v_m = np.zeros_like(rho_m)
    mask_m = rho_m > params.epsilon
    v_m[mask_m] = (y_m[mask_m] / rho_m[mask_m]) - p_m[mask_m]
    
    v_c = np.zeros_like(rho_c)
    mask_c = rho_c > params.epsilon
    v_c[mask_c] = (y_c[mask_c] / rho_c[mask_c]) - p_c[mask_c]
    
    # Convert to plot units
    rho_m_plot = rho_m * physics.VEH_M_TO_VEH_KM
    rho_c_plot = rho_c * physics.VEH_M_TO_VEH_KM
    v_m_plot = v_m * physics.MS_TO_KMH
    v_c_plot = v_c * physics.MS_TO_KMH
    
    x = grid.cell_centers(include_ghost=False)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Density
    ax1.plot(x, rho_m_plot, label='Motos', color='red', linewidth=2)
    ax1.plot(x, rho_c_plot, label='Cars', color='blue', linestyle='--', linewidth=2)
    ax1.set_title(f'{title} - Density')
    ax1.set_ylabel('Density (veh/km)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Velocity
    ax2.plot(x, v_m_plot, label='Motos', color='red', linewidth=2)
    ax2.plot(x, v_c_plot, label='Cars', color='blue', linestyle='--', linewidth=2)
    ax2.set_title(f'{title} - Velocity')
    ax2.set_xlabel('Position (m)')
    ax2.set_ylabel('Velocity (km/h)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved plot to {filename}")

def main():
    # Output to 'results' directory so that kaggle_runner collects it
    output_dir = "results/reproduction"
    
    # Scenario 1: Simple Shock (Motos)
    # Left: Free flow (low rho, high v), Right: Congested (high rho, low v) -> Shock moves upstream
    print("\n--- Scenario 1: Simple Shock (Motos) ---")
    runner, config = setup_simulation("shock", t_final=15.0)
    set_riemann_state(
        runner,
        rho_L=(20, 0), v_L=(80, 0),  # Motos free, Cars empty
        rho_R=(150, 0), v_R=(10, 0)  # Motos congested, Cars empty
    )
    run_and_plot(runner, f"{output_dir}/fig_riemann_choc_simple.png", "Validation: Choc Simple (Motos)")
    
    # Scenario 2: Multi-class Interaction
    # Cars shock interacting with Motos
    print("\n--- Scenario 2: Multi-class Interaction ---")
    runner, config = setup_simulation("interaction", t_final=10.0)
    set_riemann_state(
        runner,
        rho_L=(20, 20), v_L=(80, 60),   # Both free
        rho_R=(20, 120), v_R=(80, 10)   # Motos free, Cars congested
    )
    run_and_plot(runner, f"{output_dir}/fig_riemann_interaction_multiclasse.png", "Validation: Interaction Multi-classes")

if __name__ == "__main__":
    main()
