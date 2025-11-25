"""
Network Simulation Executor

This module provides the `NetworkSimulator` class, which is responsible for
orchestrating the time-stepping of a `NetworkGrid` object. It manages the
main simulation loop, calls the numerical schemes, and handles data logging.
"""

import numpy as np
import time
import os
from tqdm import tqdm
from typing import Optional

from ...network.network_grid import NetworkGrid
from ...config.network_simulation_config import NetworkSimulationConfig
from ...numerics.cfl import cfl_condition_gpu_native, compute_adaptive_cfl_with_history
from ...numerics.time_integration import strang_splitting_step_gpu_native
from ...numerics.gpu.memory_pool import GPUMemoryPool
from ...numerics.gpu.network_coupling_gpu import NetworkCouplingGPU
from ..logging import SimulationLogger, LogLevel, create_logger_from_flags

# Detect if running on Kaggle (disable TQDM to reduce log bloat)
IS_KAGGLE = os.path.exists('/kaggle/working') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


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
            debug: Enable detailed diagnostic logging.
        """
        self.network = network
        self.config = config
        self.quiet = quiet
        self.device = device
        self.t = 0.0
        self.time_step = 0
        self.debug = debug
        self.last_log_time = -np.inf # Initialize for time-based logging
        self.last_diagnostic_time = -np.inf
        self.dt_history = []
        
        # Initialize time-based checkpoint tracking (Phase 1)
        self.last_checkpoint_time = -self.config.time.output_dt  # Force checkpoint at t=0
        
        # Initialize CFL diagnostic throttling (Phase 3)
        self.cfl_warning_count = 0  # Counter for throttled warnings
        
        # Initialize structured logger (Phase 2)
        self.logger = create_logger_from_flags(quiet=quiet, debug=debug)
        
        self.params = config.physics
        
        self.logger.info("Initializing GPU Network Simulator...")
        if self.debug:
            self.logger.debug("GPU debug logging ENABLED")

        # 1. Initialize GPU Memory Pool
        self.gpu_pool = self._initialize_gpu_pool()
        
        # 2. Initialize GPU-native Network Coupling
        self.network_coupling = self._initialize_gpu_coupling()

        if self.debug and self.device == 'gpu':
            self._debug_dump_state("Initial state (t=0)")

        # 3. Data logging setup
        self.history = {
            'time': [],
            'segments': {
                seg_id: {
                    'density': [], 'speed': [],
                    'rho_c': [], 'rho_m': [], 'v_c': [], 'v_m': []
                } for seg_id in self.network.segments.keys()
            }
        }

        # Initialize traffic lights logging
        self.history['traffic_lights'] = {}
        for node_id, node in self.network.nodes.items():
            # Check if node has traffic lights (handle both attribute existence and None value)
            if getattr(node, 'traffic_lights', None) is not None:
                self.history['traffic_lights'][node_id] = {'green_segments': []}
        
        self.logger.info("✅ GPU Network Simulator initialized.")

    # Note: Initial conditions are now applied during NetworkGrid construction
    # in NetworkGrid.add_segment_from_config(), not here. This ensures the
    # correct state is set before GPU memory pool initialization.

    def _initialize_gpu_pool(self) -> Optional[GPUMemoryPool]:
        """Creates and initializes the GPUMemoryPool for the network."""
        if self.device == 'cpu':
            self.logger.debug("Running in CPU mode, skipping GPU Memory Pool initialization.")
            return None

        self.logger.info("  - Initializing GPU Memory Pool for network...")
            
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
            U_cpu = np.ascontiguousarray(segment_data['U'])
            
            # Get road quality: if not set on the grid, use default from physics config
            road_quality = segment_data['grid'].road_quality
            if road_quality is None:
                # Create uniform array with default quality
                N_phys = segment_data['grid'].N_physical
                road_quality = np.full(N_phys, self.config.physics.default_road_quality, dtype=np.float64)
            
            R_cpu = np.ascontiguousarray(road_quality)
            pool.initialize_segment_state(seg_id, U_cpu, R_cpu)
        
        stats = pool.get_memory_stats()
        self.logger.info(f"✅ GPUMemoryPool initialized:")
        self.logger.info(f"  - Segments: {len(segment_ids)}")
        self.logger.info(f"  - Total cells: {sum(N_per_segment.values())}")
        self.logger.debug(f"  - Ghost cells: {self.config.grid.num_ghost_cells}")
        self.logger.debug(f"  - Compute Capability: {pool.compute_capability}")
        self.logger.debug(f"  - CUDA streams: {'Enabled' if pool.enable_streams else 'Disabled'}")
        self.logger.debug(f"  - GPU memory allocated: {stats['allocated_mb']:.2f} MB")
            
        return pool

    def _initialize_gpu_coupling(self) -> Optional[NetworkCouplingGPU]:
        """Initializes the GPU-native network coupling manager."""
        if self.device == 'cpu':
            self.logger.debug("Running in CPU mode, skipping GPU-native network coupling initialization.")
            return None

        self.logger.info("  - Initializing GPU-native network coupling...")
            
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
        
        self.logger.debug("GPU Coupling Manager created.")
            
        return coupling_manager

    def run(self, t_final: Optional[float] = None, timeout: Optional[float] = None):
        """
        Runs the full GPU-based simulation from t=0 to t=t_final.

        Args:
            t_final (float, optional): Overrides the simulation's final time.
            timeout (float, optional): Maximum wall-clock time in seconds before stopping.
        """
        sim_t_final = t_final if t_final is not None else self.config.t_final

        self.logger.info(f"Starting GPU network simulation from t=0 to t={sim_t_final}s...")

        # On Kaggle, TQDM creates excessive log output - disable it and use periodic logging instead
        use_tqdm = not IS_KAGGLE and not self.quiet
        pbar = tqdm(total=sim_t_final, desc="Simulating on GPU", disable=not use_tqdm)
        
        start_time = time.time()
        last_progress_log = 0.0  # Track when we last logged progress
        progress_log_interval = 60.0  # Log progress every 60 simulated seconds

        while self.t < sim_t_final:
            # Check for timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                self.logger.warning(f"Simulation stopped after {timeout} seconds (timeout).")
                break
                
            if self.device == 'cpu':
                raise NotImplementedError("CPU mode for NetworkSimulator.run() is not yet implemented.")

            # 1. Calculate network-wide CFL-stable dt on the GPU
            if self.config.time.dt:
                # Use a fixed time step if provided
                stable_dt = self.config.time.dt
                diagnostics = None
            else:
                # Otherwise, calculate dt based on CFL condition
                # Enable diagnostics if dt was small in previous step or if debug mode
                enable_diag = (self.time_step > 0 and hasattr(self, '_last_dt') and self._last_dt < 0.05) or self.debug
                
                # Compute adaptive CFL factor based on recent history
                adaptive_cfl_factor = compute_adaptive_cfl_with_history(
                    self.dt_history, 
                    base_cfl=self.config.time.cfl_factor,
                    n_window=10,
                    threshold=self.config.time.dt_collapse_threshold
                )

                # Log if CFL is being reduced
                if adaptive_cfl_factor < self.config.time.cfl_factor:
                    self.logger.warning(
                        f"Adaptive CFL reduction: {self.config.time.cfl_factor:.2f} → {adaptive_cfl_factor:.2f} "
                        f"due to persistent dt collapse"
                    )

                # PHASE GPU BATCHING: Use batched CFL calculation
                from ..numerics.cfl import cfl_condition_gpu_batched
                
                # Get dx from first segment (all segments have same dx in Victoria Island)
                first_seg = next(iter(self.network.segments.values()))
                dx = first_seg['grid'].dx
                
                stable_dt = cfl_condition_gpu_batched(
                    gpu_pool=self.gpu_pool,
                    dx=dx,
                    params=self.config.physics,
                    cfl_max=adaptive_cfl_factor
                )
                diagnostics = None  # Batched version doesn't provide diagnostics yet
            
            # Store for next iteration
            self._last_dt = stable_dt

            # Check for dt collapse - fail fast instead of slowing to crawl
            if stable_dt < self.config.time.dt_min:
                raise RuntimeError(
                    f"NUMERICAL INSTABILITY DETECTED: CFL dt collapsed to {stable_dt:.6f}s "
                    f"which is below dt_min={self.config.time.dt_min}s. "
                    f"\n\nThis indicates extreme eigenvalues in the system. "
                    f"\nLimiting segment: {diagnostics.get('limiting_segment', 'unknown') if diagnostics else 'unknown'}"
                    f"\n\nPossible causes:"
                    f"\n  1. Density approaching rho_max causing pressure gradient explosion"
                    f"\n  2. Unphysical velocities at network junctions"
                    f"\n  3. Insufficient spatial resolution (dx too large)"
                    f"\n\nRecommended actions:"
                    f"\n  - Review CFL diagnostic logs above for problematic segments"
                    f"\n  - Reduce dx or increase rho_max if physically justified"
                    f"\n  - Check initial/boundary conditions for extreme values"
                )

            # Log warning if dt is collapsing but still above minimum
            if stable_dt < self.config.time.dt_collapse_threshold:
                self.logger.warning(
                    f"dt approaching collapse: {stable_dt:.6f}s < threshold={self.config.time.dt_collapse_threshold}s"
                )

            # Clamp dt to configured bounds (after checking for collapse)
            stable_dt = max(self.config.time.dt_min, min(stable_dt, self.config.time.dt_max))

            # Track dt history for adaptive CFL
            self.dt_history.append(stable_dt)
            if len(self.dt_history) > 20:  # Keep last 20 values only
                self.dt_history.pop(0)
            
            # Log CFL diagnostics if dt is collapsing (Phase 3: throttled)
            if diagnostics is not None and stable_dt < 0.05:
                self._log_cfl_diagnostics_throttled(diagnostics, stable_dt)
            
            # Adjust last step to hit t_final exactly
            if self.t + stable_dt > sim_t_final:
                stable_dt = sim_t_final - self.t

            # If dt is somehow zero or negative, stop the simulation
            if stable_dt <= 0:
                self.logger.warning(f"Stopping simulation: stable_dt is {stable_dt}.")
                break

            # 2. Evolve ALL segments on GPU using batched kernel
            # PHASE GPU BATCHING: Single kernel launch replaces per-segment loop
            from ...numerics.time_integration import batched_strang_splitting_step_gpu_native
            
            # Get dx from first segment (all segments have same dx in Victoria Island)
            first_seg = next(iter(self.network.segments.values()))
            dx = first_seg['grid'].dx
            
            # Launch batched kernel for all 70 segments in parallel
            batched_strang_splitting_step_gpu_native(
                gpu_pool=self.gpu_pool,
                dt=stable_dt,
                dx=dx,
                params=self.config.physics,
                current_time=self.t
            )

            # 3. Apply network coupling on the GPU
            self.network_coupling.apply_coupling(self.config.physics)

            # DEBUGGING BLOCK: Stop and dump state if dt collapses
            if self.debug and stable_dt < 0.01 and self.time_step > 10:
                self.logger.debug(f"dt collapsed to {stable_dt:.6f} at t={self.t:.4f}s. Dumping state.")
                self._debug_dump_state("State at dt collapse")
                # You might want to save the full state to a file here for later analysis
                # For now, we just stop the simulation to inspect the log.
                break

            # 4. Log data at regular time intervals (Phase 1: time-based checkpointing)
            if self.t - self.last_checkpoint_time >= self.config.time.output_dt:
                self._log_state()
                self.last_checkpoint_time = self.t
                if self.debug:
                    self._debug_dump_state(f"State log at step {self.time_step} (t={self.t:.2f}s)")
            elif self.debug and self.time_step < 5:
                # Still capture early-step behavior even if not aligned with output_dt
                self._debug_dump_state(f"Early debug snapshot step {self.time_step}")

            # 5. Update time and progress
            self.t += stable_dt
            self.time_step += 1
            pbar.update(stable_dt)
            pbar.set_postfix({"Time": f"{self.t:.2f}s", "dt": f"{stable_dt:.4f}s", "step": self.time_step})
            
            # On Kaggle: periodic progress logging instead of TQDM spam
            if IS_KAGGLE and (self.t - last_progress_log >= progress_log_interval):
                elapsed_wall = time.time() - start_time
                progress_pct = (self.t / sim_t_final) * 100
                remaining_sim = sim_t_final - self.t
                sim_rate = self.t / elapsed_wall if elapsed_wall > 0 else 0
                eta_wall = remaining_sim / sim_rate if sim_rate > 0 else 0
                
                self.logger.info(
                    f"Progress: {progress_pct:.1f}% | "
                    f"t={self.t:.1f}/{sim_t_final:.0f}s | "
                    f"Step={self.time_step} | "
                    f"dt={stable_dt:.4f}s | "
                    f"Wall: {elapsed_wall:.0f}s | "
                    f"ETA: {eta_wall/60:.1f}min"
                )
                last_progress_log = self.t

        pbar.close()
        self.logger.info("GPU network simulation finished.")
        
        # Save final state if not just checkpointed (Phase 1: final checkpoint)
        if self.t > self.last_checkpoint_time + 1e-9:  # Small epsilon to avoid floating point issues
            self._log_state()
        
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

    def reset(self):
        """Resets the simulation state to t=0 and restores initial conditions."""
        self.t = 0.0
        self.time_step = 0
        self.last_log_time = -np.inf
        self.last_diagnostic_time = -np.inf
        self.last_checkpoint_time = -self.config.time.output_dt
        self.cfl_warning_count = 0
        self.dt_history = []
        
        # Reset GPU data (zeroes out arrays)
        if self.gpu_pool:
            self.gpu_pool.reset_data()
            
            # Restore Initial Conditions from NetworkGrid
            for seg_id, segment_data in self.network.segments.items():
                if 'U_initial' in segment_data:
                    U_cpu = np.ascontiguousarray(segment_data['U_initial'])
                    
                    # Get road quality
                    road_quality = segment_data['grid'].road_quality
                    if road_quality is None:
                        N_phys = segment_data['grid'].N_physical
                        road_quality = np.full(N_phys, self.config.physics.default_road_quality, dtype=np.float64)
                    R_cpu = np.ascontiguousarray(road_quality)
                    
                    self.gpu_pool.initialize_segment_state(seg_id, U_cpu, R_cpu)
                else:
                    self.logger.warning(f"Segment {seg_id} has no U_initial, resetting to zero.")
        
        self.logger.info("✅ NetworkSimulator reset to t=0.")

    def sync_state_to_cpu(self):
        """Downloads current GPU state to NetworkGrid segments on CPU."""
        if not self.gpu_pool:
            return
            
        for seg_id, segment_data in self.network.segments.items():
            d_U = self.gpu_pool.get_segment_state(seg_id)
            # Fix for stride mismatch: copy to new host array then assign
            # d_U is a slice of a larger array, so it has different strides than a contiguous segment array
            host_U = d_U.copy_to_host()
            segment_data['U'][:] = host_U

    def _debug_dump_state(self, label: str):
        """Prints min/max/mean stats for each segment's density and speed (DEBUG mode only)."""
        if not self.debug or self.device != 'gpu':
            return

        self.logger.debug(f"{label}")
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

            self.logger.debug(
                f"   Segment {seg_id}: density[min={density_min:.6f}, max={density_max:.6f}, mean={density_mean:.6f}] "
                f"speed[min={speed_min:.6f}, max={speed_max:.6f}, mean={speed_mean:.6f}]"
            )

    def _log_cfl_diagnostics_throttled(self, diagnostics, stable_dt):
        """
        Logs THROTTLED CFL diagnostics when dt becomes small (Phase 3 optimization).
        
        Instead of printing ALL 70 segments, this:
        - Shows the first CFL warning
        - Shows every 10th warning after that
        - Only displays the TOP 5 worst segments instead of all segments
        
        Args:
            diagnostics: Dictionary containing CFL diagnostic information
            stable_dt: The calculated stable time step
        """
        self.cfl_warning_count += 1
        
        # Throttle: only log first warning, then every 50th (reduced log volume for long runs)
        if self.cfl_warning_count > 1 and self.cfl_warning_count % 50 != 0:
            return  # Skip this warning
        
        # Header
        self.logger.section(
            f"CFL DIAGNOSTIC #{self.cfl_warning_count}: dt={stable_dt:.6e} at t={self.t:.4f}s (step {self.time_step})",
            char="=",
            width=70
        )
        
        self.logger.warning(f"Global max_ratio (λ/dx): {diagnostics['global_max_ratio']:.6e}")
        self.logger.info(f"CFL factor: {self.config.time.cfl_factor}")
        
        # Sort segments by max_ratio to find the worst offenders
        sorted_segments = sorted(
            diagnostics['segments'].items(),
            key=lambda item: item[1]['max_ratio'],
            reverse=True
        )
        
        # Show only TOP 5 worst segments (not all 70!)
        top_n = 5
        self.logger.subsection(f"🔴 TOP {top_n} LIMITING SEGMENTS (out of {len(sorted_segments)})", char="-")
        
        for rank, (seg_id, seg_diag) in enumerate(sorted_segments[:top_n], start=1):
            self.logger.warning(f"#{rank} Segment {seg_id}:")
            self.logger.info(f"   max_ratio (λ/dx): {seg_diag['max_ratio']:.6e}")
            self.logger.debug(f"   dx: {seg_diag['dx']:.6f} m")
            self.logger.debug(f"   dt_seg: {seg_diag['dt_seg']:.6e} s")
            
            # Only show full density/velocity details in DEBUG mode
            if self.logger.level <= LogLevel.DEBUG:
                self.logger.debug(f"   ρ_m: min={seg_diag['rho_m']['min']:.6e}, max={seg_diag['rho_m']['max']:.6e}, mean={seg_diag['rho_m']['mean']:.6e}")
                self.logger.debug(f"   ρ_c: min={seg_diag['rho_c']['min']:.6e}, max={seg_diag['rho_c']['max']:.6e}, mean={seg_diag['rho_c']['mean']:.6e}")
                self.logger.debug(f"   w_m: min={seg_diag['w_m']['min']:.6e}, max={seg_diag['w_m']['max']:.6e}, mean={seg_diag['w_m']['mean']:.6e}")
                self.logger.debug(f"   w_c: min={seg_diag['w_c']['min']:.6e}, max={seg_diag['w_c']['max']:.6e}, mean={seg_diag['w_c']['mean']:.6e}")
        
        # Summary of remaining segments
        if len(sorted_segments) > top_n:
            remaining_count = len(sorted_segments) - top_n
            self.logger.info(f"   ... and {remaining_count} other segments with lower max_ratio")
        
        self.logger.info(f"{'='*70}\n")

    def _log_state(self):
        """
        Logs the current state by transferring data from GPU to CPU.
        This is an expensive operation and should be done infrequently.
        
        PHASE GPU BATCHING: Now uses batched checkpoint when available for better performance.
        """
        self.history['time'].append(self.t)
        for seg_id in self.network.segments.keys():
            # Try batched checkpoint first (Phase GPU Batching)
            try:
                U_cpu = self.gpu_pool.checkpoint_to_cpu_batched(seg_id)
            except (AttributeError, ValueError, KeyError):
                # Fallback to legacy checkpoint if batched not available
                U_cpu = self.gpu_pool.checkpoint_to_cpu(seg_id)
            
            grid = self.network.segments[seg_id]['grid']
            
            # Extract physical data from the CPU copy
            # State vector U = [rho_moto, v_moto, rho_car, v_car]
            rho_m = U_cpu[0, grid.physical_cell_indices]  # Motos density
            v_m = U_cpu[1, grid.physical_cell_indices]    # Motos velocity
            rho_c = U_cpu[2, grid.physical_cell_indices]  # Cars density
            v_c = U_cpu[3, grid.physical_cell_indices]    # Cars velocity
            
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
            
            # Log individual class data
            self.history['segments'][seg_id]['rho_c'].append(rho_c)
            self.history['segments'][seg_id]['rho_m'].append(rho_m)
            self.history['segments'][seg_id]['v_c'].append(v_c)
            self.history['segments'][seg_id]['v_m'].append(v_m)

        # Log traffic lights state
        for node_id, node_log in self.history['traffic_lights'].items():
            node = self.network.nodes[node_id]
            if getattr(node, 'traffic_lights', None) is not None:
                # Get list of currently green segments
                green_segments = node.traffic_lights.get_current_green_segments(self.t)
                node_log['green_segments'].append(green_segments)

    def _log_diagnostics(self):
        """Logs diagnostic information about the simulation run."""
        if not self.dt_history:
            return
        
        dt_array = np.array(self.dt_history)
        mean_dt = np.mean(dt_array)
        min_dt = np.min(dt_array)
        max_dt = np.max(dt_array)
        
        self.logger.subsection(f"Simulation Diagnostics (at t={self.t:.2f}s)", char="-")
        self.logger.info(f"  - Timestep (dt) Stats: Min={min_dt:.6f}s, Max={max_dt:.6f}s, Mean={mean_dt:.6f}s")
        self.logger.info(f"  - Total Steps: {self.time_step}")
        
        # Clear history for next diagnostic interval
        self.dt_history.clear()

