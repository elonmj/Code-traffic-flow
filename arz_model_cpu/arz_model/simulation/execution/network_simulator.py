"""
Network Simulation Executor

This module provides the `NetworkSimulator` class, which is responsible for
orchestrating the time-stepping of a `NetworkGrid` object. It manages the
main simulation loop, calls the numerical schemes, and handles data logging.
"""

import numpy as np
import time
from tqdm import tqdm
from typing import Optional

from ...network.network_grid import NetworkGrid
from ...config.network_simulation_config import NetworkSimulationConfig
from ...numerics.cfl import cfl_condition

class NetworkSimulator:
    """
    Orchestrates the execution of a multi-segment network simulation.
    """

    def __init__(self, network: NetworkGrid, config: NetworkSimulationConfig, quiet: bool = False):
        """
        Initializes the network simulator.

        Args:
            network: The initialized NetworkGrid object.
            config: The simulation configuration object.
            quiet: Suppress progress bar and verbose output.
        """
        self.network = network
        self.config = config
        self.quiet = quiet
        self.t = 0.0
        self.time_step = 0
        
        # Data logging setup
        self.history = {
            'time': [],
            'segments': {seg_id: {'density': [], 'speed': []} for seg_id in self.network.segments.keys()}
        }

    def run(self, t_final: Optional[float] = None):
        """
        Runs the full simulation from t=0 to t=t_final.

        Args:
            t_final (float, optional): Overrides the simulation's final time.
        """
        sim_t_final = t_final if t_final is not None else self.config.t_final

        if not self.quiet:
            print(f"Starting network simulation from t=0 to t={sim_t_final}s...")

        # Use tqdm for progress bar if not in quiet mode
        # Calculate initial stable dt based on CFL condition for the whole network
        initial_stable_dt, _ = cfl_condition(self.network)
        
        time_steps = np.arange(0, sim_t_final, initial_stable_dt)
        pbar = tqdm(time_steps, desc="Simulating", disable=self.quiet)

        for self.t in pbar:
            # 1. Calculate network-wide CFL-stable dt
            stable_dt, cfl_limited_by = cfl_condition(self.network)
            
            # 2. Evolve the network by one time step
            self.network.step(dt=stable_dt, current_time=self.t)

            # 3. Log data at the specified output interval
            if self.time_step % max(1, int(self.config.output_dt / stable_dt)) == 0:
                self._log_state()

            # 4. Update progress bar description
            if not self.quiet:
                pbar.set_postfix({"Time": f"{self.t:.2f}s", "dt": f"{stable_dt:.4f}s"})
            
            self.time_step += 1

        if not self.quiet:
            print("Network simulation finished.")
            
        return self.history

    def _log_state(self):
        """
        Logs the current state of the network (density, speed) for each segment.
        """
        self.history['time'].append(self.t)
        for seg_id, segment_data in self.network.segments.items():
            U = segment_data['U']
            grid = segment_data['grid']
            
            # Extract physical data
            rho_c = U[0, grid.physical_cell_indices]
            rho_m = U[1, grid.physical_cell_indices]
            v_c = U[2, grid.physical_cell_indices]
            v_m = U[3, grid.physical_cell_indices]
            
            total_density = rho_c + rho_m
            
            # Avoid division by zero for speed calculation
            # Weighted average speed
            avg_speed = np.divide(
                rho_c * v_c + rho_m * v_m,
                total_density,
                out=np.zeros_like(total_density),
                where=total_density != 0
            )
            
            self.history['segments'][seg_id]['density'].append(total_density)
            self.history['segments'][seg_id]['speed'].append(avg_speed)
