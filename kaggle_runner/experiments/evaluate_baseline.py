import sys
import os
from pathlib import Path
import numpy as np
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from arz_model.config.config_factory import create_victoria_island_config
from arz_model.simulation.runner import SimulationRunner
from arz_model.network.network_grid import NetworkGrid
from Code_RL.src.config.rl_network_config import RLNetworkConfig

def evaluate_baseline(timesteps=1000, cycle_time=90.0, split=0.5, decision_interval=15.0):
    print(f"Starting Baseline Evaluation (Fixed Time Control)")
    print(f"Timesteps: {timesteps}, Cycle: {cycle_time}s, Split: {split}, Interval: {decision_interval}s")

    # 1. Setup
    config = create_victoria_island_config()
    network_grid = NetworkGrid.from_config(config)
    runner = SimulationRunner(network_grid=network_grid, simulation_config=config)
    rl_config = RLNetworkConfig(config)
    
    # Get signalized segments
    signalized_segments = rl_config.signalized_segment_ids
    print(f"Signalized segments: {len(signalized_segments)}")
    
    # Metrics
    total_densities = []
    
    start_time = time.time()
    
    # 2. Simulation Loop
    total_sim_time = timesteps * decision_interval
    print(f"Total simulation time: {total_sim_time}s")
    
    current_step = 0
    while runner.t < total_sim_time:
        t = runner.t
        
        # Fixed Time Logic
        # Simple 2-phase cycle
        cycle_pos = t % cycle_time
        if cycle_pos < (cycle_time * split):
            phase = 0 # Green NS
        else:
            phase = 1 # Green EW
            
        # Apply phase
        phase_updates = rl_config.get_phase_updates(phase)
        runner.set_boundary_phases_bulk(phase_updates, validate=False)
        
        # Step simulation
        # Run until next decision interval
        next_t = (current_step + 1) * decision_interval
        runner.run_until(next_t)
        
        # Calculate Metric (Density)
        # Replicating TrafficSignalEnvDirectV3._compute_reward logic
        current_total_density = 0.0
        for seg_id in signalized_segments:
            if seg_id in runner.network_grid.segments:
                seg = runner.network_grid.segments[seg_id]
                U = seg['U']
                grid = seg['grid']
                i_start = grid.num_ghost_cells
                i_end = grid.num_ghost_cells + grid.N_physical
                # Rho_m + Rho_c
                # U is (4, N)
                # U[0] is rho_m, U[2] is rho_c
                current_total_density += (U[0, i_start:i_end] + U[2, i_start:i_end]).mean()
        
        avg_density = current_total_density / len(signalized_segments) if signalized_segments else 0
        # Normalize
        rho_max = config.physics.rho_max
        avg_density_norm = avg_density / rho_max
        
        total_densities.append(float(avg_density_norm))
        
        current_step += 1
        if current_step % 100 == 0:
            print(f"Step {current_step}/{timesteps}, Time: {runner.t:.1f}s, Avg Density: {avg_density_norm:.4f}")
            
        if current_step >= timesteps:
            break

    end_time = time.time()
    duration = end_time - start_time
    
    mean_density = float(np.mean(total_densities))
    print(f"\nBaseline Evaluation Complete")
    print(f"Duration: {duration:.2f}s")
    print(f"Mean Normalized Density: {mean_density:.6f}")
    
    # Save results for Kaggle artifact retrieval
    import json
    results = {
        "mean_density": mean_density,
        "duration": duration,
        "timesteps": timesteps,
        "cycle_time": cycle_time,
        "split": split,
        "decision_interval": decision_interval,
        "density_history": total_densities
    }
    
    output_path = "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")
    
    return mean_density

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=1000)
    args = parser.parse_args()
    
    evaluate_baseline(timesteps=args.timesteps)
