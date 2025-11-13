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
from ...numerics.cfl import cfl_condition_gpu_native
from ...numerics.time_integration import strang_splitting_step_gpu_native
from ...numerics.gpu.memory_pool import GPUMemoryPool
from ...numerics.gpu.network_coupling_gpu import NetworkCouplingGPU


class NetworkSimulator:
    """
    Orchestrates the execution of a multi-segment network simulation on the GPU.
    """

    def __init__(self, network: NetworkGrid, config: NetworkSimulationConfig, quiet: bool = False):
        """
        Initializes the GPU-based network simulator.

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
        
        if not self.quiet:
            print("Initializing GPU Network Simulator...")

        # 1. Initialize GPU Memory Pool
        self.gpu_pool = self._initialize_gpu_pool()
        
        # 2. Initialize GPU-native Network Coupling
        self.network_coupling = self._initialize_gpu_coupling()

        # 3. Data logging setup
        self.history = {
            'time': [],
            'segments': {seg_id: {'density': [], 'speed': []} for seg_id in self.network.segments.keys()}
        }
        
        if not self.quiet:
            print("✅ GPU Network Simulator initialized.")

    def _initialize_gpu_pool(self) -> GPUMemoryPool:
        """Creates and initializes the GPUMemoryPool for the network."""
        if not self.quiet:
            print("  - Initializing GPU Memory Pool for network...")
            
        segment_ids = list(self.network.segments.keys())
        N_per_segment = {seg_id: segment['grid'].N for seg_id, segment in self.network.segments.items()}
        
        pool = GPUMemoryPool(
            segment_ids=segment_ids,
            N_per_segment=N_per_segment,
            ghost_cells=self.config.grid.ghost_cells
        )
        
        # Transfer initial states and road quality to the GPU
        for seg_id, segment_data in self.network.segments.items():
            U_cpu = segment_data['U']
            R_cpu = segment_data['grid'].road_quality
            pool.initialize_segment_state(seg_id, U_cpu, R_cpu)
            
        if not self.quiet:
            stats = pool.get_memory_stats()
            print(f"    - GPU Memory Pool created. Allocated: {stats['allocated_mb']:.2f} MB")
            
        return pool

    def _initialize_gpu_coupling(self) -> NetworkCouplingGPU:
        """Initializes the GPU-native network coupling manager."""
        if not self.quiet:
            print("  - Initializing GPU-native network coupling...")
            
        # The topology information needs to be passed to the coupling manager
        # This might involve creating a GPU-compatible representation of the network graph
        topology_info = {
            "nodes": self.network.nodes,
            "links": self.network.links,
            "segments": self.network.segments
        }
        
        coupling_manager = NetworkCouplingGPU(
            gpu_pool=self.gpu_pool,
            network_topology=topology_info
        )
        
        if not self.quiet:
            print("    - GPU Coupling Manager created.")
            
        return coupling_manager

    def run(self, t_final: Optional[float] = None):
        """
        Runs the full GPU-based simulation from t=0 to t=t_final.

        Args:
            t_final (float, optional): Overrides the simulation's final time.
        """
        sim_t_final = t_final if t_final is not None else self.config.t_final

        if not self.quiet:
            print(f"Starting GPU network simulation from t=0 to t={sim_t_final}s...")

        pbar = tqdm(total=sim_t_final, desc="Simulating on GPU", disable=self.quiet)

        while self.t < sim_t_final:
            # 1. Calculate network-wide CFL-stable dt on the GPU
            if self.config.time.dt:
                # Use a fixed time step if provided
                stable_dt = self.config.time.dt
            else:
                # Otherwise, calculate dt based on CFL condition
                stable_dt = cfl_condition_gpu_native(
                    gpu_pool=self.gpu_pool,
                    network_segments=self.network.segments,
                    cfl_factor=self.config.time.cfl_factor,
                    physics_params=self.config.physics
                )
            
            # Adjust last step to hit t_final exactly
            if self.t + stable_dt > sim_t_final:
                stable_dt = sim_t_final - self.t

            # If dt is somehow zero or negative, stop the simulation
            if stable_dt <= 0:
                if not self.quiet:
                    print(f"Stopping simulation: stable_dt is {stable_dt}.")
                break

            # 2. Evolve each segment on the GPU using Strang splitting
            for seg_id, segment_data in self.network.segments.items():
                d_U_in = self.gpu_pool.get_segment_state(seg_id)
                grid = segment_data['grid']
                
                # Perform one full time step for the segment
                d_U_out = strang_splitting_step_gpu_native(
                    d_U_n=d_U_in,
                    dt=stable_dt,
                    grid=grid,
                    params=self.config.physics,
                    gpu_pool=self.gpu_pool,
                    seg_id=seg_id,
                    current_time=self.t
                )
                
                # The output d_U_out is a new array; update the pool to point to it.
                self.gpu_pool.update_segment_state(seg_id, d_U_out)

            # 3. Apply network coupling on the GPU
            self.network_coupling.apply_coupling(self.config.physics)

            # 4. Log data (requires transferring data from GPU to CPU)
            if self.time_step % max(1, int(self.config.output_dt / stable_dt)) == 0:
                self._log_state()

            # 5. Update time and progress
            self.t += stable_dt
            self.time_step += 1
            pbar.update(stable_dt)
            pbar.set_postfix({"Time": f"{self.t:.2f}s", "dt": f"{stable_dt:.4f}s"})

        pbar.close()
        if not self.quiet:
            print("GPU network simulation finished.")
            
        return self.history

    def _log_state(self):
        """
        Logs the current state by transferring data from GPU to CPU.
        This is an expensive operation and should be done infrequently.
        """
        self.history['time'].append(self.t)
        for seg_id in self.network.segments.keys():
            # Checkpoint the segment state from GPU to CPU
            U_cpu = self.gpu_pool.checkpoint_to_cpu(seg_id)
            grid = self.network.segments[seg_id]['grid']
            
            # Extract physical data from the CPU copy
            rho_c = U_cpu[0, grid.physical_cell_indices]
            rho_m = U_cpu[1, grid.physical_cell_indices]
            v_c = U_cpu[2, grid.physical_cell_indices]
            v_m = U_cpu[3, grid.physical_cell_indices]
            
            total_density = rho_c + rho_m
            
            # Weighted average speed
            avg_speed = np.divide(
                rho_c * v_c + rho_m * v_m,
                total_density,
                out=np.zeros_like(total_density),
                where=total_density != 0
            )
            
            self.history['segments'][seg_id]['density'].append(total_density)
            self.history['segments'][seg_id]['speed'].append(avg_speed)

