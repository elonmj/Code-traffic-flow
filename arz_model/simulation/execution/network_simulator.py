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

    def __init__(self, network: NetworkGrid, config: NetworkSimulationConfig, quiet: bool = False, device: str = 'gpu', debug: bool = False):
        """
        Initializes the GPU-based network simulator.

        Args:
            network: The initialized NetworkGrid object.
            config: The simulation configuration object.
            quiet: Suppress progress bar and verbose output.
            device: The device to run the simulation on ('gpu' or 'cpu').
        """
        self.network = network
        self.config = config
        self.quiet = quiet
        self.device = device
        self.t = 0.0
        self.time_step = 0
        self.debug = debug
        
        self.params = config.physics
        
        if not self.quiet:
            print("Initializing GPU Network Simulator...")
            if self.debug:
                print("[DEBUG] GPU debug logging ENABLED")

        # 1. Apply initial conditions to the CPU-side state arrays
        self._apply_initial_conditions()

        # 2. Initialize GPU Memory Pool
        self.gpu_pool = self._initialize_gpu_pool()
        
        # 3. Initialize GPU-native Network Coupling
        self.network_coupling = self._initialize_gpu_coupling()

        if self.debug and self.device == 'gpu':
            self._debug_dump_state("Initial state (t=0)")

        # 4. Data logging setup
        self.history = {
            'time': [],
            'segments': {seg_id: {'density': [], 'speed': []} for seg_id in self.network.segments.keys()}
        }
        
        if not self.quiet:
            print("✅ GPU Network Simulator initialized.")

    def _apply_initial_conditions(self):
        """Applies initial conditions to the state arrays on the CPU before GPU transfer."""
        if not self.quiet:
            print("  - Applying initial conditions...")

        from ...core.physics import calculate_pressure
        
        print(f"[DEBUG_IC] Number of segments in config: {len(self.config.segments)}")
        
        # Iterate through segments and apply their ICs from the config
        for seg_config in self.config.segments:
            seg_id = seg_config.id
            print(f"[DEBUG_IC] Processing segment: {seg_id}")
            
            if seg_id not in self.network.segments:
                if not self.quiet:
                    print(f"    WARNING: Segment {seg_id} from config not found in network")
                continue
            
            segment_data = self.network.segments[seg_id]
            U = segment_data['U']  # This is the state array
            
            # Get IC configuration from the segment's config
            ic_config = seg_config.initial_conditions
            print(f"[DEBUG_IC]   ic_config: {ic_config}")
            print(f"[DEBUG_IC]   ic_config type: {type(ic_config)}")
            
            if ic_config is None:
                if not self.quiet:
                    print(f"    - Segment {seg_id} has no initial conditions defined. Skipping.")
                continue
            
            # The actual IC data is nested inside the 'config' attribute
            print(f"[DEBUG_IC]   Checking for 'config' attribute...")
            if hasattr(ic_config, 'config'):
                ic = ic_config.config
                print(f"[DEBUG_IC]   ic (from .config): {ic}")
                print(f"[DEBUG_IC]   ic type: {type(ic)}")
            else:
                ic = ic_config
                print(f"[DEBUG_IC]   ic (direct): {ic}")
                print(f"[DEBUG_IC]   ic type: {type(ic)}")
            
            # Apply based on IC type (currently handles UniformIC)
            print(f"[DEBUG_IC]   Checking for density and velocity attributes...")
            print(f"[DEBUG_IC]   hasattr(ic, 'density'): {hasattr(ic, 'density')}")
            print(f"[DEBUG_IC]   hasattr(ic, 'velocity'): {hasattr(ic, 'velocity')}")
            
            if hasattr(ic, 'density') and hasattr(ic, 'velocity'):
                # UniformIC case
                density = ic.density  # Total density in veh/km
                velocity = ic.velocity  # Velocity in km/h
                
                print(f"[DEBUG_IC]   density={density}, velocity={velocity}")
                
                # Convert to model units (veh/m, m/s)
                density_m = density / 1000.0  # veh/km -> veh/m
                velocity_ms = velocity / 3.6  # km/h -> m/s
                
                print(f"[DEBUG_IC]   density_m={density_m:.6f}, velocity_ms={velocity_ms:.6f}")
                
                # Split between motorcycles and cars (using alpha ratio)
                alpha = self.config.physics.alpha
                rho_m = alpha * density_m
                rho_c = (1.0 - alpha) * density_m
                
                print(f"[DEBUG_IC]   alpha={alpha}, rho_m={rho_m:.6f}, rho_c={rho_c:.6f}")
                
                # Set densities
                U[0, :] = rho_m  # Motorcycle density
                U[2, :] = rho_c  # Car density
                
                print(f"[DEBUG_IC]   After setting densities: U[0,5]={U[0,5]:.6f}, U[2,5]={U[2,5]:.6f}")
                
                # Calculate pressure using the correct parameter names from the Pydantic model
                p_m, p_c = calculate_pressure(
                    U[0, :], U[2, :],
                    self.config.physics.alpha,
                    self.config.physics.rho_jam,  # Corrected from rho_max
                    self.config.physics.epsilon,
                    self.config.physics.K_m,      # Corrected from k_m
                    self.config.physics.gamma_m,
                    self.config.physics.K_c,      # Corrected from k_c
                    self.config.physics.gamma_c
                )
                
                print(f"[DEBUG_IC]   Pressure calculated: p_m[5]={p_m[5]:.6f}, p_c[5]={p_c[5]:.6f}")
                
                # Set Lagrangian momentum w = v + p
                U[1, :] = velocity_ms + p_m  # Motorcycle momentum
                U[3, :] = velocity_ms + p_c  # Car momentum
                
                print(f"[DEBUG_IC]   After setting momentum: U[1,5]={U[1,5]:.6f}, U[3,5]={U[3,5]:.6f}")
                
                if not self.quiet:
                    print(f"    ✓ Applied IC to {seg_id}: ρ={density:.1f} veh/km, v={velocity:.1f} km/h")
            else:
                if not self.quiet:
                    print(f"    - Skipping IC for {seg_id}: unsupported IC type or missing attributes.")

    def _initialize_gpu_pool(self) -> Optional[GPUMemoryPool]:
        """Creates and initializes the GPUMemoryPool for the network."""
        if self.device == 'cpu':
            if not self.quiet:
                print("  - Running in CPU mode, skipping GPU Memory Pool initialization.")
            return None

        if not self.quiet:
            print("  - Initializing GPU Memory Pool for network...")
            
        segment_ids = list(self.network.segments.keys())
        N_per_segment = {seg_id: segment['grid'].N_physical for seg_id, segment in self.network.segments.items()}
        
        pool = GPUMemoryPool(
            segment_ids=segment_ids,
            N_per_segment=N_per_segment,
            ghost_cells=self.config.grid.num_ghost_cells
            # compute_capability defaults to (6,0) - sufficient for Kaggle P100
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

    def _initialize_gpu_coupling(self) -> Optional[NetworkCouplingGPU]:
        """Initializes the GPU-native network coupling manager."""
        if self.device == 'cpu':
            if not self.quiet:
                print("  - Running in CPU mode, skipping GPU-native network coupling initialization.")
            return None

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
            if self.device == 'cpu':
                raise NotImplementedError("CPU mode for NetworkSimulator.run() is not yet implemented.")

            # 1. Calculate network-wide CFL-stable dt on the GPU
            if self.config.time.dt:
                # Use a fixed time step if provided
                stable_dt = self.config.time.dt
            else:
                # Otherwise, calculate dt based on CFL condition
                stable_dt = cfl_condition_gpu_native(
                    gpu_pool=self.gpu_pool,
                    network=self.network,
                    params=self.config.physics,
                    cfl_max=self.config.time.cfl_factor
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
            if self.time_step % max(1, int(self.config.time.output_dt / stable_dt)) == 0:
                self._log_state()
                if self.debug:
                    self._debug_dump_state(f"State log at step {self.time_step} (t={self.t + stable_dt:.2f}s)")
            elif self.debug and self.time_step < 5:
                # Still capture early-step behavior even if not aligned with output_dt
                self._debug_dump_state(f"Early debug snapshot step {self.time_step}")

            # 5. Update time and progress
            self.t += stable_dt
            self.time_step += 1
            pbar.update(stable_dt)
            pbar.set_postfix({"Time": f"{self.t:.2f}s", "dt": f"{stable_dt:.4f}s"})

        pbar.close()
        if not self.quiet:
            print("GPU network simulation finished.")
        
        # Collect final states from GPU
        final_states = {}
        for seg_id in self.network.segments.keys():
            final_states[seg_id] = self.gpu_pool.checkpoint_to_cpu(seg_id)
        
        # Return results in the expected format
        return {
            'final_time': self.t,
            'total_steps': self.time_step,
            'final_states': final_states,
            'history': self.history  # Keep history for backward compatibility
        }

    def _debug_dump_state(self, label: str):
        """Prints min/max/mean stats for each segment's density and speed."""
        if not self.debug or self.device != 'gpu':
            return

        print(f"[DEBUG] {label}")
        for seg_id in self.network.segments.keys():
            U_cpu = self.gpu_pool.checkpoint_to_cpu(seg_id)
            grid = self.network.segments[seg_id]['grid']

            rho_c = U_cpu[0, grid.physical_cell_indices]
            rho_m = U_cpu[1, grid.physical_cell_indices]
            v_c = U_cpu[2, grid.physical_cell_indices]
            v_m = U_cpu[3, grid.physical_cell_indices]

            total_density = rho_c + rho_m
            avg_speed = np.divide(
                rho_c * v_c + rho_m * v_m,
                total_density,
                out=np.zeros_like(total_density),
                where=total_density != 0
            )

            density_min = np.min(total_density)
            density_max = np.max(total_density)
            density_mean = np.mean(total_density)

            speed_min = np.min(avg_speed)
            speed_max = np.max(avg_speed)
            speed_mean = np.mean(avg_speed)

            print(
                f"   Segment {seg_id}: density[min={density_min:.6f}, max={density_max:.6f}, mean={density_mean:.6f}] "
                f"speed[min={speed_min:.6f}, max={speed_max:.6f}, mean={speed_mean:.6f}]"
            )

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

