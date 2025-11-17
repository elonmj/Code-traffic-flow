"""
TrafficSignalEnvDirect - Direct In-Process Gymnasium Environment

⚠️ DEPRECATED: This environment (V1) is deprecated as of 2025-11-17.
Please migrate to TrafficSignalEnvDirectV2 for:
- 100-200x performance improvement (0.2-0.6ms vs 50-100ms step latency)
- Pydantic-based type-safe configuration
- Direct GPU memory access (no HTTP overhead)

Migration Guide: See Code_RL/docs/MIGRATION_GUIDE.md

---

This environment implements direct coupling with ARZ simulator following
the industry-standard MuJoCo pattern (no HTTP/server overhead).

Performance: ~0.2-0.6ms per step (100-200x faster than server-based coupling)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List
import os
import sys

# Add arz_model to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from arz_model.simulation.runner import SimulationRunner
from arz_model.core.parameters import VEH_KM_TO_VEH_M


class TrafficSignalEnvDirect(gym.Env):
    """
    Gymnasium environment for traffic signal control using direct ARZ simulator coupling.
    
    This environment follows the MuJoCo direct integration pattern:
    - Simulator instantiated in __init__() (no server/client architecture)
    - Direct method calls to simulator (no network serialization)
    - In-process memory access to state arrays (no IPC overhead)
    
    MDP Specification (from Chapter 6):
    - Decision interval: Δt_dec = 10s
    - Observation: [ρ_m/ρ_max, v_m/v_free, ρ_c/ρ_max, v_c/v_free, phase_onehot]
    - Action: 0 = maintain phase, 1 = switch phase
    - Reward: R_congestion + R_stabilite + R_fluidite
    
    Attributes:
        runner (SimulationRunner): Direct ARZ simulator instance
        observation_space (spaces.Box): Normalized continuous observation space
        action_space (spaces.Discrete): Discrete action space {0, 1}
    """
    
    metadata = {'render_modes': []}
    
    def __init__(self,
                 simulation_config = None,  # Pydantic SimulationConfig or NetworkSimulationConfig
                 decision_interval: float = 15.0,  # Changed from 10.0 (Bug #27 validation, 4x improvement)
                 observation_segments: Dict[str, list] = None,
                 normalization_params: Dict[str, float] = None,
                 reward_weights: Dict[str, float] = None,
                 episode_max_time: float = 3600.0,
                 quiet: bool = True,
                 device: str = 'cpu'):
        """
        Initialize the traffic signal environment with direct simulator coupling.
        
        Args:
            simulation_config: Pydantic SimulationConfig or NetworkSimulationConfig object (REQUIRED)
            decision_interval: Time between agent decisions in seconds (default: 15.0)
                              Justification: Bug #27 investigation showed 4x improvement (593 → 2361 episode reward)
                              Literature: Chu et al. (2020) found 15s provides 'best balance' for urban TSC
                              Physical: 15s ≈ 0.18 × τ_propagation (transient-rich regime, captures dynamics)
            observation_segments: Dict with 'upstream' and 'downstream' segment indices
                                 Example: {'upstream': [8, 9, 10], 'downstream': [11, 12, 13]}
            normalization_params: Dict with 'rho_max', 'v_free' for observation normalization
                                 Example: {'rho_max': 0.2, 'v_free': 15.0}
            reward_weights: Dict with 'alpha', 'kappa', 'mu' for reward calculation
                           Example: {'alpha': 1.0, 'kappa': 0.1, 'mu': 0.5}
            episode_max_time: Maximum simulation time per episode in seconds
            quiet: Suppress simulator output
            device: 'cpu' or 'gpu' for simulation backend
            
        Note:
            YAML configuration is no longer supported. Use Pydantic config objects.
            For migration guide, see Code_RL/docs/MIGRATION_GUIDE.md
        """
        super().__init__()
        
        # Validate configuration inputs
        if simulation_config is None:
            raise ValueError(
                "simulation_config (Pydantic) is required. "
                "YAML configuration mode has been removed. "
                "See migration guide for details."
            )
        
        # Store configuration
        self.simulation_config = simulation_config
        self.decision_interval = decision_interval
        self.episode_max_time = episode_max_time
        self.quiet = quiet
        self.device = device
        
        # Default observation segments (can be customized per scenario)
        if observation_segments is None:
            # Default: 3 upstream and 3 downstream segments around middle
            observation_segments = {
                'upstream': [8, 9, 10],
                'downstream': [11, 12, 13]
            }
        self.observation_segments = observation_segments
        self.n_segments = len(observation_segments['upstream']) + len(observation_segments['downstream'])
        
        # Normalization parameters (from calibration)
        # Separated by vehicle class (Chapter 6, Section 6.2.1)
        # NOTE: In Pydantic mode, extract from simulation_config.physics
        if normalization_params is None:
            # Try to extract from Pydantic simulation_config if available
            if simulation_config is not None and hasattr(simulation_config, 'physics'):
                physics = simulation_config.physics
                normalization_params = {
                    'rho_max_motorcycles': physics.rho_max * physics.alpha * 1000.0,  # veh/km
                    'rho_max_cars': physics.rho_max * (1.0 - physics.alpha) * 1000.0,  # veh/km
                    'v_free_motorcycles': physics.V0_m * 3.6,  # m/s → km/h
                    'v_free_cars': physics.V0_c * 3.6          # m/s → km/h
                }
                if not quiet:
                    print(f"[NORMALIZATION] Using Pydantic config V0 speeds: v_m={physics.V0_m*3.6:.1f} km/h, v_c={physics.V0_c*3.6:.1f} km/h")
        
        # Fallback to default if still None
        if normalization_params is None:
            normalization_params = {
                'rho_max_motorcycles': 300.0,  # veh/km (West African context)
                'rho_max_cars': 150.0,         # veh/km
                'v_free_motorcycles': 40.0,    # km/h (urban free flow)
                'v_free_cars': 50.0            # km/h
            }
        
        # Convert to SI units (veh/m, m/s) and store per-class values
        self.rho_max_m = normalization_params.get('rho_max_motorcycles', 300.0) / 1000.0  # veh/m
        self.rho_max_c = normalization_params.get('rho_max_cars', 150.0) / 1000.0         # veh/m
        self.v_free_m = normalization_params.get('v_free_motorcycles', 40.0) / 3.6        # m/s
        self.v_free_c = normalization_params.get('v_free_cars', 50.0) / 3.6               # m/s
        
        # Reward weights (from Chapter 6)
        if reward_weights is None:
            reward_weights = {
                'alpha': 1.0,   # Congestion penalty weight
                'kappa': 0.1,   # Phase change penalty weight
                'mu': 0.5       # Outflow reward weight
            }
        self.alpha = reward_weights['alpha']
        self.kappa = reward_weights['kappa']
        self.mu = reward_weights['mu']
        
        # Traffic signal phases (simple 2-phase for single intersection)
        self.n_phases = 2
        self.current_phase = 0
        
        # Define action space: {0: maintain, 1: switch}
        self.action_space = spaces.Discrete(2)
        
        # Define observation space
        # Observation: [ρ_m_norm, v_m_norm, ρ_c_norm, v_c_norm] × n_segments + phase_onehot
        obs_dim = 4 * self.n_segments + self.n_phases
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize simulator (DIRECT INSTANTIATION - MuJoCo pattern)
        self.runner = None
        self._initialize_simulator()
        
        # Episode tracking
        self.episode_step = 0
        self.total_reward = 0.0
        self.previous_observation = None
        
        if not self.quiet:
            print(f"TrafficSignalEnvDirect initialized:")
            print(f"  Decision interval: {self.decision_interval}s")
            print(f"  Observation segments: {self.n_segments}")
            print(f"  Observation space: {self.observation_space.shape}")
            print(f"  Action space: {self.action_space.n}")
            print(f"  Device: {self.device}")
    
    def _initialize_simulator(self):
        """
        Initialize simulator (Grid1D or NetworkGrid).
        
        Detects configuration type and creates appropriate simulator through SimulationRunner:
        - SimulationConfig (grid=GridConfig) → SimulationRunner → Grid1D
        - NetworkSimulationConfig (segments/nodes/links) → SimulationRunner → NetworkGrid
        
        This enables seamless scaling from single-segment (20m) to multi-segment (400m-15km).
        """
        if self.simulation_config is None:
            raise ValueError(
                "YAML configuration mode is deprecated. "
                "Please provide simulation_config (Pydantic) instead. "
                "See migration guide for details."
            )
        
        # Pydantic config mode
        # SimulationRunner handles BOTH SimulationConfig and NetworkSimulationConfig
        if not self.quiet:
            from arz_model.config.network_simulation_config import NetworkSimulationConfig
            if isinstance(self.simulation_config, NetworkSimulationConfig):
                print("🌐 Initializing NetworkGrid simulation (multi-segment)")
                print(f"   Segments: {len(self.simulation_config.segments)}")
                print(f"   Nodes: {len(self.simulation_config.nodes)}")
            else:
                print("📏 Initializing Grid1D simulation (single segment)")
        
        # Create SimulationRunner (handles both types!)
        self.runner = SimulationRunner(
            config=self.simulation_config,
            quiet=self.quiet,
            device=self.device
        )
        
        # Initialize simulation
        self.runner.initialize_simulation()
        
        if not self.quiet:
            print(f"✅ Simulation initialized")
            print(f"   Decision interval: {self.decision_interval}s")
        
        # Store grid/network reference for observation extraction
        if hasattr(self.runner, 'grid'):
            # Single segment mode
            self.grid = self.runner.grid
        elif hasattr(self.runner, 'network'):
            # Network mode
            self.network = self.runner.network
            if not self.quiet:
                print(f"   Network segments available for observation:")
                for seg_id in list(self.network.segments.keys())[:5]:
                    print(f"      - {seg_id}")
                if len(self.network.segments) > 5:
                    print(f"      ... and {len(self.network.segments) - 5} more")

    @property
    def _current_time(self) -> float:
        """Get current simulation time - uses SimulationRunner unified API."""
        return self.runner.current_time

    def _extract_network_observations(self, segment_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract observations from NetworkGrid segments.
        
        Args:
            segment_ids: List of segment IDs or indices to observe
            
        Returns:
            Dictionary with keys: 'rho_m', 'v_m', 'rho_c', 'v_c' (averaged values per segment)
        """
        n_segments = len(segment_ids)
        rho_m = np.zeros(n_segments, dtype=np.float32)
        v_m = np.zeros(n_segments, dtype=np.float32)
        rho_c = np.zeros(n_segments, dtype=np.float32)
        v_c = np.zeros(n_segments, dtype=np.float32)
        
        # NetworkGrid uses segment IDs like 'seg_0', 'seg_1'
        # But observation_segments may contain indices (for Grid1D compatibility)
        # Convert indices to segment IDs if needed
        network_seg_ids = []
        for seg in segment_ids:
            if isinstance(seg, (int, np.integer)):
                # It's an index - map to segment ID
                # Assume segments are named 'seg_0', 'seg_1', etc.
                network_seg_ids.append(f'seg_{seg % len(self.runner.network.segments)}')
            else:
                network_seg_ids.append(seg)
        
        for i, seg_id in enumerate(network_seg_ids):
            if seg_id in self.runner.network.segments:
                segment = self.runner.network.segments[seg_id]
                # segment is a dict: {'grid': Grid1D, 'U': np.ndarray, ...}
                U = segment['U']  # State array (4, N_total)
                
                # Average over physical cells (exclude ghost cells at indices 0,1 and -2,-1)
                rho_m[i] = np.mean(U[0, 2:-2])  # Motorcycles density
                rho_c[i] = np.mean(U[2, 2:-2])  # Cars density
                
                # For ARZ model: v = w - p (Lagrangian variables)
                # Must calculate pressure to get physical velocity
                from arz_model.core.physics import calculate_pressure, calculate_physical_velocity
                
                # Extract momentum averages
                w_m_avg = np.mean(U[1, 2:-2])
                w_c_avg = np.mean(U[3, 2:-2])
                
                # Calculate pressure at average densities
                p_m_val, p_c_val = calculate_pressure(
                    np.array([rho_m[i]]), np.array([rho_c[i]]),
                    self.runner.params.alpha, self.runner.params.rho_jam, self.runner.params.epsilon,
                    self.runner.params.K_m, self.runner.params.gamma_m,
                    self.runner.params.K_c, self.runner.params.gamma_c
                )
                
                # Physical velocity: v = w - p
                v_m_calc, v_c_calc = calculate_physical_velocity(
                    np.array([w_m_avg]), np.array([w_c_avg]),
                    p_m_val, p_c_val
                )
                
                v_m[i] = float(v_m_calc[0])
                v_c[i] = float(v_c_calc[0])
        
        return {
            'rho_m': rho_m,
            'v_m': v_m,
            'rho_c': rho_c,
            'v_c': v_c
        }

    def _set_signal_state(self, node_id: str, phase_id: int) -> None:
        """
        Set traffic signal state - uses SimulationRunner unified API.
        
        Args:
            node_id: Node identifier (e.g., 'left' for Grid1D, 'node_1' for NetworkGrid)
            phase_id: Phase identifier (0=red, 1=green)
        """
        # Check if NetworkGrid mode
        if hasattr(self.runner, 'is_network_mode') and self.runner.is_network_mode:
            # NetworkGridSimulator API: set_signal(signal_plan: Dict)
            if phase_id == 0:
                green_times = [5.0, 25.0]  # [main, cross] - RED phase
            else:
                green_times = [25.0, 5.0]  # [main, cross] - GREEN phase
            
            signal_plan = {
                'node_id': node_id,
                'phase_id': phase_id,
                'green_times': green_times,
                'yellow_time': 3.0,
                'all_red_time': 2.0
            }
            self.runner.set_signal(signal_plan)
        else:
            # Grid1D API: set_traffic_signal_state()
            self.runner.set_traffic_signal_state(node_id, phase_id)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            observation: Initial normalized observation vector
            info: Dictionary with auxiliary information
        """
        # CRITICAL: Call super().reset() to properly seed the environment
        super().reset(seed=seed)
        
        # ✅ LOGGING: Episode summary (if not first reset)
        if hasattr(self, 'episode_step') and self.episode_step > 0 and not self.quiet:
            print(f"\n{'='*80}")
            print(f"[EPISODE END] Steps: {self.episode_step} | "
                  f"Duration: {self._current_time:.1f}s | "
                  f"Total Reward: {self.total_reward:.2f} | "
                  f"Avg Reward/Step: {self.total_reward/max(1, self.episode_step):.3f}")
            print(f"{'='*80}\n")
        
        # Reinitialize simulator to t=0
        self._initialize_simulator()
        
        # Reset episode tracking
        self.episode_step = 0
        self.total_reward = 0.0
        self.current_phase = 0
        self.previous_observation = None
        
        # Reset queue tracking for reward calculation
        if hasattr(self, 'previous_queue_length'):
            delattr(self, 'previous_queue_length')
        
        # Set initial traffic signal state
        # For Grid1D: node_id='left', For NetworkGrid: node_id='node_1' (first controlled node)
        if hasattr(self, 'network') and self.network is not None:
            # NetworkGrid mode: use first controlled node
            controlled_nodes = self.simulation_config.controlled_nodes
            if controlled_nodes:
                node_id = controlled_nodes[0]
            else:
                node_id = 'node_1'  # Default fallback
        else:
            # Grid1D mode: use 'left' boundary
            node_id = 'left'
        
        self._set_signal_state(node_id, self.current_phase)
        
        # Build initial observation
        observation = self._build_observation()
        self.previous_observation = observation.copy()
        
        # Info dict
        info = {
            'episode_step': self.episode_step,
            'simulation_time': self._current_time,
            'current_phase': self.current_phase
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep of the environment.
        
        Args:
            action: 0 = maintain current phase, 1 = switch to next phase
            
        Returns:
            observation: New observation after action
            reward: Reward for this step
            terminated: Whether episode ended (goal reached)
            truncated: Whether episode ended (time limit)
            info: Auxiliary information dict
        """
        # Store previous state for reward calculation
        prev_phase = self.current_phase
        
        # ✅ BUG #7 FIX: Interpret action as desired phase directly
        # Action 0 = RED phase, Action 1 = GREEN phase
        # This fixes semantic mismatch with BaselineController:
        #   - BaselineController returns: 1.0 (wants GREEN) or 0.0 (wants RED)
        #   - Previous env logic: 1 (toggle), 0 (maintain) → phase desynchronization
        #   - New logic: Action directly specifies desired phase → perfect alignment
        # Evidence from kernel czlc: Phase stayed GREEN when should be RED, causing drainage
        
        # ✅ BUG #37 FIX: Properly discretize continuous actions from RL agents
        # Problem: int(action) truncates all 0 ≤ action < 1 to phase 0
        #   - RL agents output continuous actions like 0.3, 0.7, 0.95
        #   - int(0.3) = 0, int(0.7) = 0, int(0.95) = 0 (phase stays RED)
        #   - Phase locked at RED → queue builds → delta_queue constant → reward = 0.0 ALWAYS
        #   - RL agent can't learn with zero reward signal
        # Solution: Use round() to discretize fairly around 0.5 threshold
        #   - round(0.3) = 0 (RED)
        #   - round(0.7) = 1 (GREEN)
        #   - round(0.95) = 1 (GREEN)
        #   - Phase transitions properly → queue changes → reward varies → learning possible
        self.current_phase = round(float(action))
        
        # ✅ BUG #6 FIX PRESERVED: ALWAYS synchronize BC with current phase
        # This ensures boundary conditions match controller intent
        # For Grid1D: node_id='left', For NetworkGrid: node_id='node_1' (first controlled node)
        if hasattr(self, 'network') and self.network is not None:
            # NetworkGrid mode: use first controlled node
            controlled_nodes = self.simulation_config.controlled_nodes
            if controlled_nodes:
                node_id = controlled_nodes[0]
            else:
                node_id = 'node_1'  # Default fallback
        else:
            # Grid1D mode: use 'left' boundary
            node_id = 'left'
        
        self._set_signal_state(node_id, self.current_phase)
        
        # Advance simulation by decision_interval
        # This executes multiple internal simulation steps (Δt_sim << Δt_dec)
        from arz_model.network.network_simulator import NetworkGridSimulator
        if isinstance(self.runner, NetworkGridSimulator):
            # NetworkGridSimulator: step(dt, repeat_k)
            # Calculate how many timesteps needed
            n_steps = int(self.decision_interval / self.runner.dt_sim)
            self.runner.step(dt=self.runner.dt_sim, repeat_k=n_steps)
        else:
            # SimulationRunner: run(t_final, output_dt)
            target_time = self._current_time + self.decision_interval
            self.runner.run(t_final=target_time, output_dt=self.decision_interval)
        
        # Build new observation
        observation = self._build_observation()
        
        # Calculate reward
        reward = self._calculate_reward(observation, action, prev_phase)
        
        # Update tracking
        self.episode_step += 1
        self.total_reward += reward
        self.previous_observation = observation.copy()
        
        # ✅ LOGGING: Step progress (every 10 steps for Bug #20 debugging)
        if self.episode_step % 10 == 0 and not self.quiet:
            print(f"[STEP {self.episode_step:>4d}] t={self._current_time:>7.1f}s | "
                  f"phase={self.current_phase} | reward={reward:>7.2f} | "
                  f"total_reward={self.total_reward:>8.2f}", flush=True)
        
        # Check termination conditions
        terminated = False  # No explicit goal state
        truncated = self._current_time >= self.episode_max_time
        
        if not self.quiet and truncated:
            print(f"DEBUG: Truncation - runner.t={self._current_time}, episode_max_time={self.episode_max_time}")
        
        # Build info dict
        info = {
            'episode_step': self.episode_step,
            'simulation_time': self._current_time,
            'current_phase': self.current_phase,
            'phase_changed': (action == 1),
            'total_reward': self.total_reward
        }
        
        return observation, reward, terminated, truncated, info
    
    def _build_observation(self) -> np.ndarray:
        """
        Build normalized observation vector from simulator state.
        
        Observation structure (Chapter 6):
            [ρ_m/ρ_max, v_m/v_free, ρ_c/ρ_max, v_c/v_free] × n_segments + phase_onehot
        
        Returns:
            Normalized observation vector (float32)
        """
        # Collect segment indices
        all_segments = (
            self.observation_segments['upstream'] +
            self.observation_segments['downstream']
        )
        
        # Extract raw observations from simulator (unified API)
        if hasattr(self.runner, 'is_network_mode') and self.runner.is_network_mode:
            # NetworkGrid mode: extract from network segments
            print(f"[DEBUG] _build_observation: Using NetworkGrid mode, all_segments={all_segments}")
            raw_obs = self._extract_network_observations(all_segments)
            print(f"[DEBUG]   raw_obs: rho_m={raw_obs['rho_m']}, v_m={raw_obs['v_m']}")
        else:
            # Grid1D mode: use SimulationRunner API
            print(f"[DEBUG] _build_observation: Using Grid1D mode")
            raw_obs = self.runner.get_segment_observations(all_segments)
        
        # Normalize densities and velocities (class-specific, Chapter 6)
        rho_m_norm = raw_obs['rho_m'] / self.rho_max_m
        v_m_norm = raw_obs['v_m'] / self.v_free_m
        rho_c_norm = raw_obs['rho_c'] / self.rho_max_c
        v_c_norm = raw_obs['v_c'] / self.v_free_c
        
        # Clip to [0, 1] range (handle edge cases)
        rho_m_norm = np.clip(rho_m_norm, 0.0, 1.0)
        v_m_norm = np.clip(v_m_norm, 0.0, 1.0)
        rho_c_norm = np.clip(rho_c_norm, 0.0, 1.0)
        v_c_norm = np.clip(v_c_norm, 0.0, 1.0)
        
        # Interleave: [ρ_m, v_m, ρ_c, v_c] for each segment
        traffic_obs = np.empty(4 * self.n_segments, dtype=np.float32)
        for i in range(self.n_segments):
            traffic_obs[4*i + 0] = rho_m_norm[i]
            traffic_obs[4*i + 1] = v_m_norm[i]
            traffic_obs[4*i + 2] = rho_c_norm[i]
            traffic_obs[4*i + 3] = v_c_norm[i]
        
        # Add phase one-hot encoding
        phase_onehot = np.zeros(self.n_phases, dtype=np.float32)
        phase_onehot[self.current_phase] = 1.0
        
        # Concatenate
        observation = np.concatenate([traffic_obs, phase_onehot])
        
        return observation
    
    def _calculate_reward(self, observation: np.ndarray, action: int, prev_phase: int) -> float:
        """
        Queue-based reward following Cai & Wei (2024).
        
        Reward = -(queue_length_t+1 - queue_length_t) - penalty_phase_change
        
        This replaces the density-based reward that caused RL to learn RED constant
        strategy (minimizing density) instead of optimal GREEN cycling (maximizing throughput).
        
        Args:
            observation: State vector normalized
            action: Control action (0=maintain, 1=switch)
            prev_phase: Previous phase (not used)
        
        Returns:
            reward: Scalar reward value
        
        References:
            Cai & Wei (2024). Deep reinforcement learning for traffic signal control.
            Scientific Reports 14:14116. Nature Portfolio.
        """
        n_segments = self.n_segments
        
        # Get dx (cell size) - different for Grid1D vs NetworkGrid
        if hasattr(self.runner, 'is_network_mode') and self.runner.is_network_mode:
            # NetworkGrid: use first segment's grid dx
            first_seg_dict = list(self.runner.network_simulator.network.segments.values())[0]
            dx = first_seg_dict['grid'].dx
        else:
            # Grid1D: use runner's grid dx
            dx = self.runner.grid.dx
        
        # Extract and denormalize densities and velocities
        densities_m = observation[0::4][:n_segments] * self.rho_max_m
        velocities_m = observation[1::4][:n_segments] * self.v_free_m
        densities_c = observation[2::4][:n_segments] * self.rho_max_c
        velocities_c = observation[3::4][:n_segments] * self.v_free_c
        
        # 🔍 BUG #35 DIAGNOSTIC: Log queue calculation details
        if not hasattr(self, '_queue_log_count'):
            self._queue_log_count = 0
        if self._queue_log_count < 5 or self.episode_step % 10 == 0:  # First 5 steps + every 10th
            print(f"[QUEUE_DIAGNOSTIC] ===== Step {self.episode_step} t={self._current_time:.1f}s =====")
            print(f"[QUEUE_DIAGNOSTIC] Observation shape: {observation.shape}, n_segments: {n_segments}")
            print(f"[QUEUE_DIAGNOSTIC] Normalized obs[0:4]: {observation[:4]} (rho_m, v_m, rho_c, v_c)")
            print(f"[QUEUE_DIAGNOSTIC] Denorm factors: rho_max_m={self.rho_max_m:.3f}, v_free_m={self.v_free_m:.2f}")
            print(f"[QUEUE_DIAGNOSTIC] velocities_m (m/s): {velocities_m}")
            print(f"[QUEUE_DIAGNOSTIC] velocities_c (m/s): {velocities_c}")
            print(f"[QUEUE_DIAGNOSTIC] densities_m (veh/m): {densities_m}")
            print(f"[QUEUE_DIAGNOSTIC] densities_c (veh/m): {densities_c}")
            self._queue_log_count += 1
        
        # Define queue threshold: vehicles slower than expected speed are queued
        # ✅ CRITICAL FIX (2025-10-24): Use ABSOLUTE threshold based on congestion physics
        #
        # Previous approach: QUEUE_SPEED_THRESHOLD = avg_velocity * 0.85
        # Problem: If ALL vehicles move at V0 (8.889 m/s), threshold = 7.56 m/s
        #          → NO vehicles slower than 85% of uniform flow → queue_length = 0 always!
        #
        # New approach: Use PHYSICAL congestion threshold
        #   - Free flow: v ≈ V0 (8.889 m/s for Lagos)
        #   - Congestion: v < 0.5 * V0 (< 4.4 m/s indicates traffic slowdown)
        #   - Severe queue: v < 0.3 * V0 (< 2.7 m/s indicates stopped/creeping traffic)
        #
        # Why 50% of V0? Physics basis:
        #   - Ve = V_creeping + (V0 - V_creeping) * (1 - rho/rho_jam)
        #   - At 50% jam density: Ve ≈ 0.5 * V0 (transition to congested flow)
        #   - Below this, vehicles experience significant queueing
        QUEUE_SPEED_THRESHOLD = self.v_free_m * 0.5  # Congested if slower than 50% of free speed
        # For Lagos: 8.889 * 0.5 = 4.44 m/s threshold (16 km/h)
        # This captures realistic congestion (speeds < 16 km/h in 32 km/h zone)
        
        # Count queued vehicles (density where v < threshold)
        queued_m = densities_m[velocities_m < QUEUE_SPEED_THRESHOLD]
        queued_c = densities_c[velocities_c < QUEUE_SPEED_THRESHOLD]
        
        # 🔍 BUG #35 DIAGNOSTIC: Log threshold check results
        if self._queue_log_count <= 5 or self.episode_step % 10 == 0:
            print(f"[QUEUE_DIAGNOSTIC] Threshold: {QUEUE_SPEED_THRESHOLD:.2f} m/s ({QUEUE_SPEED_THRESHOLD*3.6:.1f} km/h)")
            print(f"[QUEUE_DIAGNOSTIC] v_free_m={self.v_free_m:.2f} m/s → Threshold=0.5*v_free_m")
            print(f"[QUEUE_DIAGNOSTIC] velocities_m: {velocities_m}")
            print(f"[QUEUE_DIAGNOSTIC] Below threshold (m): {velocities_m < QUEUE_SPEED_THRESHOLD}")
            print(f"[QUEUE_DIAGNOSTIC] Below threshold (c): {velocities_c < QUEUE_SPEED_THRESHOLD}")
            print(f"[QUEUE_DIAGNOSTIC] queued_m densities: {queued_m} (sum={np.sum(queued_m):.4f})")
            print(f"[QUEUE_DIAGNOSTIC] queued_c densities: {queued_c} (sum={np.sum(queued_c):.4f})")
        
        # Total queue length (vehicles in congestion)
        current_queue_length = np.sum(queued_m) + np.sum(queued_c)
        current_queue_length *= dx  # Convert to total vehicles
        
        # 🔍 BUG #35 DIAGNOSTIC: Log final queue length
        if self._queue_log_count <= 5 or self.episode_step % 10 == 0:
            print(f"[QUEUE_DIAGNOSTIC] dx={dx:.2f}m, queue_length={current_queue_length:.2f} vehicles")
        
        # Get previous queue length
        if not hasattr(self, 'previous_queue_length'):
            self.previous_queue_length = current_queue_length
            delta_queue = 0.0
        else:
            delta_queue = current_queue_length - self.previous_queue_length
            self.previous_queue_length = current_queue_length
        
        # Reward component 1: Queue change (PRIMARY)
        # Negative change = queue reduction = positive reward
        # ✅ BUG #29 FIX: Amplify queue signal (was 10.0, now 50.0)
        # Problem: Queue barely changes (delta_queue ≈ 0), reward always ≈ 0
        # With 10.0 multiplier, even 0.1 vehicle change = 1.0 reward (too weak)
        # With 50.0 multiplier, 0.1 vehicle change = 5.0 reward (more meaningful)
        R_queue = -delta_queue * 50.0  # Increased from 10.0 to amplify learning signal
        
        # Reward component 2: Phase change penalty (SECONDARY)
        # ✅ BUG #28 FIX: Correctly detect actual phase changes
        # ✅ BUG #29 FIX: Reduce penalty from 0.1 to 0.01
        # Problem: With constant queue (R_queue≈0), penalty dominates
        #   - Change phase: reward = 0 - 0.1 = -0.1 (ALWAYS NEGATIVE)
        #   - Stay same: reward = 0 - 0 = 0.0 (ALWAYS BETTER)
        # Agent learns: "Never change phase to avoid penalty" → stuck at one action
        # Solution: Make penalty 10x smaller so queue changes can dominate
        phase_changed = (self.current_phase != prev_phase)
        R_stability = -0.01 if phase_changed else 0.0  # Reduced from -0.1
        
        # Reward component 3: Action diversity bonus (NEW)
        # ✅ BUG #29 FIX: Encourage exploration
        # Problem: Agent stuck at one action (100% RED or 100% GREEN)
        # Solution: Small bonus for using different actions recently
        if not hasattr(self, 'action_history'):
            self.action_history = []
        self.action_history.append(self.current_phase)
        if len(self.action_history) > 10:
            self.action_history.pop(0)
        
        # Give bonus if agent used both actions in last 5 steps
        if len(self.action_history) >= 5:
            recent_actions = self.action_history[-5:]
            action_diversity = len(set(recent_actions))
            R_diversity = 0.02 if action_diversity > 1 else 0.0
        else:
            R_diversity = 0.0
        
        # Total reward
        reward = R_queue + R_stability + R_diversity
        
        # ✅ MICROSCOPIC DEBUG LOGGING (Bug #30 validation)
        # Log every reward computation with full state details for analysis
        if not hasattr(self, 'reward_log_count'):
            self.reward_log_count = 0
        self.reward_log_count += 1
        
        # Create detailed log entry with searchable patterns
        log_entry = (
            f"[REWARD_MICROSCOPE] step={self.reward_log_count} "
            f"t={self._current_time:.1f}s "
            f"phase={self.current_phase} "
            f"prev_phase={prev_phase} "
            f"phase_changed={phase_changed} "
            f"| QUEUE: current={current_queue_length:.2f} "
            f"prev={self.previous_queue_length:.2f} "
            f"delta={delta_queue:.4f} "
            f"R_queue={R_queue:.4f} "
            f"| PENALTY: R_stability={R_stability:.4f} "
            f"| DIVERSITY: actions={self.action_history[-5:] if len(self.action_history) >= 5 else self.action_history} "
            f"diversity_count={len(set(recent_actions)) if len(self.action_history) >= 5 else 0} "
            f"R_diversity={R_diversity:.4f} "
            f"| TOTAL: reward={reward:.4f}"
        )
        print(log_entry, flush=True)
        
        return float(reward)
    
    def render(self):
        """Rendering not implemented (optional for training)."""
        pass
    
    def close(self):
        """Clean up resources."""
        if self.runner is not None:
            self.runner = None
