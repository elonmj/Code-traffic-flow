"""
TrafficSignalEnvDirectV3 - The definitive, modern, and CORRECT Gymnasium Environment

This version is built from the ground up to be compatible with the latest
arz_model architecture, including:
- 100% Pydantic configuration
- Direct in-process GPU memory access via NetworkGrid
- Correct usage of the modern SimulationRunner API
- Correct usage of Grid1D and PhysicsConfig APIs

This file supersedes all previous versions (V1, V2) and should be the
standard for all future RL training.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List
import os
import sys

# Add arz_model to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from arz_model.config import NetworkSimulationConfig
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner
from Code_RL.src.config.rl_network_config import RLNetworkConfig


class TrafficSignalEnvDirectV3(gym.Env):
    """
    Modern Gymnasium environment for traffic signal control, compatible with the
    latest arz_model architecture.
    
    MDP Specification:
    - State: [ρ_m_norm, v_m_norm, ρ_c_norm, v_c_norm] × N_segments + phase_onehot
    - Action: 0 = maintain phase, 1 = switch phase
    - Reward: -α·congestion + μ·throughput - κ·phase_change
    - Decision interval: Δt_dec (typically 15s)
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        simulation_config: NetworkSimulationConfig,
        decision_interval: float = 15.0,
        observation_segment_ids: Optional[List[str]] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        quiet: bool = True
    ):
        """
        Initialize RL environment with Pydantic configuration.
        
        Args:
            simulation_config: NetworkSimulationConfig from config factory.
            decision_interval: Time between RL decisions in seconds.
            observation_segment_ids: Segment IDs to observe.
            reward_weights: Reward function weights dict.
            quiet: Suppress output.
        """
        super().__init__()
        
        self.simulation_config = simulation_config
        self.decision_interval = decision_interval
        self.quiet = quiet
        
        # Extract observation segments
        if observation_segment_ids is None:
            self.observation_segment_ids = self.simulation_config.rl_metadata.get('observation_segment_ids', [seg.id for seg in self.simulation_config.segments[:6]])
        else:
            self.observation_segment_ids = observation_segment_ids
        
        # Reward weights
        self.reward_weights = reward_weights or {'alpha': 1.0, 'kappa': 0.1, 'mu': 0.5}
        
        # Domain Randomization logging control
        self._dr_call_count = 0  # Track set_inflow_conditions calls
        
        # Traffic signal state
        self.current_phase = 0
        self.n_phases = 2
        
        # RL config helper
        self.rl_config_helper = RLNetworkConfig(simulation_config)
        
        # Gymnasium spaces
        self.action_space = spaces.Discrete(2)
        obs_dim = 4 * len(self.observation_segment_ids) + self.n_phases
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        # Simulator components
        self.network_grid: Optional[NetworkGrid] = None
        self.runner: Optional[SimulationRunner] = None
        self._initialize_simulator()
        
        # Episode tracking
        self.episode_step = 0
        self.total_reward = 0.0

    def _initialize_simulator(self):
        """Initializes the arz_model simulator."""
        self.network_grid = NetworkGrid.from_config(self.simulation_config)
        self.runner = SimulationRunner(
            network_grid=self.network_grid,
            simulation_config=self.simulation_config,
            quiet=self.quiet,
            device='gpu'
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.runner is None:
            self._initialize_simulator()
        else:
            # Reuse existing simulator
            self.runner.reset()
            
        self.current_phase = 0
        self.episode_step = 0
        self.total_reward = 0.0
        obs = self._get_observation()
        info = {'time': self.runner.network_simulator.t, 'phase': self.current_phase, 'episode_step': 0}
        return obs, info

    def step(self, action: int):
        if action == 1:
            self.current_phase = 1 - self.current_phase
            self._apply_phase_to_network(self.current_phase)
        
        # Use the simulator's time as the source of truth
        t_start = self.runner.network_simulator.t
        t_end = t_start + self.decision_interval
        
        self.runner.run_until(t_end)
        
        obs = self._get_observation()
        reward = self._compute_reward(action)
        
        self.episode_step += 1
        self.total_reward += reward
        
        terminated = self.runner.network_simulator.t >= self.simulation_config.time.t_final
        truncated = False
        
        info = {
            'time': self.runner.network_simulator.t,
            'phase': self.current_phase,
            'episode_step': self.episode_step,
            'total_reward': self.total_reward
        }
        
        # Add metrics if available
        if hasattr(self, 'last_metrics'):
            info.update(self.last_metrics)
            
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        # Sync GPU state to CPU first!
        self.runner.sync_state_to_cpu()
        
        obs_list = []
        for seg_id in self.observation_segment_ids:
            seg = self.network_grid.segments[seg_id]
            U = seg['U']
            grid = seg['grid']
            
            i_start = grid.num_ghost_cells
            i_end = grid.num_ghost_cells + grid.N_physical
            
            rho_m, w_m, rho_c, w_c = U[0, i_start:i_end].mean(), U[1, i_start:i_end].mean(), U[2, i_start:i_end].mean(), U[3, i_start:i_end].mean()
            
            phys = self.simulation_config.physics
            rho_max_m = phys.rho_max * phys.alpha
            rho_max_c = phys.rho_max * (1.0 - phys.alpha)
            
            obs_list.extend([
                np.clip(rho_m / rho_max_m, 0, 1),
                np.clip(w_m / phys.v_max_m_ms, 0, 1),
                np.clip(rho_c / rho_max_c, 0, 1),
                np.clip(w_c / phys.v_max_c_ms, 0, 1)
            ])
        
        phase_onehot = [1.0, 0.0] if self.current_phase == 0 else [0.0, 1.0]
        obs_list.extend(phase_onehot)
        
        return np.array(obs_list, dtype=np.float32)

    def _compute_reward(self, action: int) -> float:
        total_density = 0.0
        total_throughput = 0.0
        
        phys = self.simulation_config.physics
        rho_max = phys.rho_max
        
        for seg_id in self.observation_segment_ids:
            seg = self.network_grid.segments[seg_id]
            U = seg['U']
            grid = seg['grid']
            i_start = grid.num_ghost_cells
            i_end = grid.num_ghost_cells + grid.N_physical
            
            # Density Calculation
            total_density += (U[0, i_start:i_end] + U[2, i_start:i_end]).mean()
            
            # Throughput Calculation (Flux at the end of the segment)
            # We use the last physical cell to estimate outflow
            idx = i_end - 1
            rho_m = max(U[0, idx], 0.0)
            w_m = U[1, idx]
            rho_c = max(U[2, idx], 0.0)
            w_c = U[3, idx]
            
            # Calculate Pressure
            rho_eff_m = rho_m + phys.alpha * rho_c
            rho_total = rho_m + rho_c
            
            norm_rho_eff_m = max(rho_eff_m / rho_max, 0.0)
            norm_rho_total = max(rho_total / rho_max, 0.0)
            
            p_m = phys.k_m * (norm_rho_eff_m ** phys.gamma_m)
            p_c = phys.k_c * (norm_rho_total ** phys.gamma_c)
            
            # Calculate Velocity: v = w - p
            v_m = w_m - p_m
            v_c = w_c - p_c
            
            # Flux Q = rho * v
            flux = rho_m * v_m + rho_c * v_c
            total_throughput += max(flux, 0.0) # Only positive flux counts
        
        avg_density_norm = (total_density / len(self.observation_segment_ids)) / rho_max
        
        # Normalize throughput (approximate max flux is rho_max * v_max / 4)
        # Let's just scale it reasonably. Max flux ~ 0.2 * 30 ~ 6.0 veh/s?
        # We'll use a small weight or normalize it.
        # Let's assume max flux per lane is roughly rho_max * v_free / 4 (LWR max flux)
        # rho_max ~ 0.2 veh/m, v_free ~ 30 m/s -> max flux ~ 1.5 veh/s per lane?
        # Let's just use the raw value and weight it.
        
        congestion_penalty = self.reward_weights['alpha'] * avg_density_norm
        phase_change_penalty = self.reward_weights['kappa'] if action == 1 else 0.0
        throughput_reward = self.reward_weights.get('mu', 0.0) * total_throughput
        
        # Store metrics for info
        self.last_metrics = {
            'avg_density': float(avg_density_norm),
            'throughput': float(total_throughput),
            'congestion_penalty': float(congestion_penalty),
            'throughput_reward': float(throughput_reward)
        }
        
        return float(-congestion_penalty + throughput_reward - phase_change_penalty)

    def _apply_phase_to_network(self, phase: int):
        phase_updates = self.rl_config_helper.get_phase_updates(phase)
        try:
            self.runner.set_boundary_phases_bulk(phase_updates=phase_updates, validate=False)
            
            # Update TrafficLightControllers for logging
            # We map RL phases to manual traffic light states for visualization
            # Phase 0 (green_NS) -> Assume Green for incoming segments
            # Phase 1 (green_EW) -> Assume Red for incoming segments (simplified)
            
            for node_id, node in self.network_grid.nodes.items():
                if getattr(node, 'traffic_lights', None) is not None:
                    if phase == 0:
                        # Phase 0: Assume Green
                        green_segments = node.incoming_segments
                    else:
                        # Phase 1: Assume Red
                        green_segments = []
                    
                    node.traffic_lights.set_manual_phase(green_segments)
                    
        except Exception as e:
            if not self.quiet:
                print(f"Warning: Failed to apply phase {phase} to network: {e}")

    def set_inflow_conditions(self, density: float, velocity: float) -> None:
        """
        Update inflow boundary conditions for all entry segments.
        
        This enables TRUE Domain Randomization by allowing the inflow parameters
        to be changed at reset time WITHOUT recreating the environment.
        
        ARCHITECTURE NOTE (Critical Fix 2025-11-28):
        ============================================
        The GPU-batched architecture does NOT dynamically apply boundary conditions
        during simulation. Ghost cells use simple reflection/extrapolation, and
        the network_coupling only handles internal junctions.
        
        Therefore, to implement TRUE Domain Randomization, we must:
        1. Modify U_initial (stored initial state) for entry segments
        2. These modified states will be restored on reset() and uploaded to GPU
        
        MUST be called BEFORE reset() for changes to take effect on the next episode.
        
        Args:
            density: Inflow density in veh/km (will be converted to veh/m)
            velocity: Inflow velocity in km/h (will be converted to m/s)
            
        Example:
            >>> env.set_inflow_conditions(density=200.0, velocity=40.0)
            >>> obs, info = env.reset()  # New episode uses updated inflow
        """
        if self.network_grid is None:
            if not self.quiet:
                print("Warning: set_inflow_conditions called before environment initialized")
            return
            
        phys = self.simulation_config.physics
        
        # Convert units: veh/km → veh/m, km/h → m/s
        rho_total = density / 1000.0  # veh/km → veh/m
        v_ms = velocity / 3.6  # km/h → m/s
        
        # Split density by vehicle class using alpha (motorcycle fraction)
        alpha = phys.alpha
        rho_m = rho_total * alpha           # Motorcycle density
        rho_c = rho_total * (1.0 - alpha)   # Car density
        
        # Calculate pressure for Lagrangian momentum w = v + p
        # At the densities we're using, pressure is non-negligible
        from arz_model.core.physics import calculate_pressure
        import numpy as np
        
        # Create temporary arrays for pressure calculation
        rho_m_arr = np.array([rho_m])
        rho_c_arr = np.array([rho_c])
        p_m, p_c = calculate_pressure(
            rho_m_arr, rho_c_arr,
            phys.alpha, phys.rho_max, phys.epsilon,
            phys.k_m, phys.gamma_m, phys.k_c, phys.gamma_c
        )
        
        # Lagrangian momentum: w = v + p
        w_m = v_ms + p_m[0]
        w_c = v_ms + p_c[0]
        
        # =====================================================================
        # TRUE DOMAIN RANDOMIZATION: Modify U_initial for entry segments
        # =====================================================================
        # Find entry nodes (boundary nodes with outgoing but no incoming segments)
        entry_segments = []
        for node_id, node in self.network_grid.nodes.items():
            if node.node_type == 'boundary':
                incoming = node.incoming_segments or []
                outgoing = node.outgoing_segments or []
                # Entry node: has outgoing segments but no incoming
                if len(outgoing) > 0 and len(incoming) == 0:
                    entry_segments.extend(outgoing)
        
        # Update U_initial for each entry segment
        segments_updated = 0
        
        for seg_id in entry_segments:
            segment = self.network_grid.segments[seg_id]
            U_initial = segment.get('U_initial')
            grid = segment.get('grid')
            
            if U_initial is None or grid is None:
                continue
                
            # Get ghost cell count
            n_ghost = grid.num_ghost_cells
            
            # Update ghost cells (left boundary) AND first few physical cells
            # This ensures the inflow state is properly propagated
            cells_to_update = n_ghost + 3  # Ghost cells + 3 physical cells
            
            U_initial[0, :cells_to_update] = rho_m  # Motorcycle density
            U_initial[1, :cells_to_update] = w_m    # Motorcycle Lagrangian momentum
            U_initial[2, :cells_to_update] = rho_c  # Car density
            U_initial[3, :cells_to_update] = w_c    # Car Lagrangian momentum
            
            segments_updated += 1
        
        # Track call count (silent - no logging during training)
        self._dr_call_count += 1

    def render(self):
        pass

    def close(self):
        pass
