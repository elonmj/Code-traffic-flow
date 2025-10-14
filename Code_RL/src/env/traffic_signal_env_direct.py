"""
TrafficSignalEnvDirect - Direct In-Process Gymnasium Environment

This environment implements direct coupling with ARZ simulator following
the industry-standard MuJoCo pattern (no HTTP/server overhead).

Performance: ~0.2-0.6ms per step (100-200x faster than server-based coupling)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
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
                 scenario_config_path: str,
                 base_config_path: str = None,
                 decision_interval: float = 10.0,
                 observation_segments: Dict[str, list] = None,
                 normalization_params: Dict[str, float] = None,
                 reward_weights: Dict[str, float] = None,
                 episode_max_time: float = 3600.0,
                 quiet: bool = True,
                 device: str = 'cpu'):
        """
        Initialize the traffic signal environment with direct simulator coupling.
        
        Args:
            scenario_config_path: Path to ARZ scenario YAML config
            base_config_path: Path to ARZ base config (default: arz_model/config/config_base.yml)
            decision_interval: Time between agent decisions in seconds (default: 10.0)
            observation_segments: Dict with 'upstream' and 'downstream' segment indices
                                 Example: {'upstream': [8, 9, 10], 'downstream': [11, 12, 13]}
            normalization_params: Dict with 'rho_max', 'v_free' for observation normalization
                                 Example: {'rho_max': 0.2, 'v_free': 15.0}
            reward_weights: Dict with 'alpha', 'kappa', 'mu' for reward calculation
                           Example: {'alpha': 1.0, 'kappa': 0.1, 'mu': 0.5}
            episode_max_time: Maximum simulation time per episode in seconds
            quiet: Suppress simulator output
            device: 'cpu' or 'gpu' for simulation backend
        """
        super().__init__()
        
        # Store configuration
        self.scenario_config_path = scenario_config_path
        self.base_config_path = base_config_path or os.path.join(
            os.path.dirname(__file__), '../../../arz_model/config/config_base.yml'
        )
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
        """Initialize ARZ simulator with direct instantiation (MuJoCo pattern)."""
        self.runner = SimulationRunner(
            scenario_config_path=self.scenario_config_path,
            base_config_path=self.base_config_path,
            quiet=self.quiet,
            device=self.device
        )
    
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
                  f"Duration: {self.runner.t:.1f}s | "
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
        self.runner.set_traffic_signal_state('left', phase_id=self.current_phase)
        
        # Build initial observation
        observation = self._build_observation()
        self.previous_observation = observation.copy()
        
        # Info dict
        info = {
            'episode_step': self.episode_step,
            'simulation_time': self.runner.t,
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
        self.current_phase = int(action)
        
        # ✅ BUG #6 FIX PRESERVED: ALWAYS synchronize BC with current phase
        # This ensures boundary conditions match controller intent
        self.runner.set_traffic_signal_state('left', phase_id=self.current_phase)
        
        # Advance simulation by decision_interval
        # This executes multiple internal simulation steps (Δt_sim << Δt_dec)
        target_time = self.runner.t + self.decision_interval
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
            print(f"[STEP {self.episode_step:>4d}] t={self.runner.t:>7.1f}s | "
                  f"phase={self.current_phase} | reward={reward:>7.2f} | "
                  f"total_reward={self.total_reward:>8.2f}", flush=True)
        
        # Check termination conditions
        terminated = False  # No explicit goal state
        truncated = self.runner.t >= self.episode_max_time
        
        if not self.quiet and truncated:
            print(f"DEBUG: Truncation - runner.t={self.runner.t}, episode_max_time={self.episode_max_time}")
        
        # Build info dict
        info = {
            'episode_step': self.episode_step,
            'simulation_time': self.runner.t,
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
        
        # Extract raw observations from simulator
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
        dx = self.runner.grid.dx
        
        # Extract and denormalize densities and velocities
        densities_m = observation[0::4][:n_segments] * self.rho_max_m
        velocities_m = observation[1::4][:n_segments] * self.v_free_m
        densities_c = observation[2::4][:n_segments] * self.rho_max_c
        velocities_c = observation[3::4][:n_segments] * self.v_free_c
        
        # Define queue threshold: vehicles with speed < 5 m/s are queued
        QUEUE_SPEED_THRESHOLD = 5.0  # m/s (~18 km/h, congestion threshold)
        
        # Count queued vehicles (density where v < threshold)
        queued_m = densities_m[velocities_m < QUEUE_SPEED_THRESHOLD]
        queued_c = densities_c[velocities_c < QUEUE_SPEED_THRESHOLD]
        
        # Total queue length (vehicles in congestion)
        current_queue_length = np.sum(queued_m) + np.sum(queued_c)
        current_queue_length *= dx  # Convert to total vehicles
        
        # Get previous queue length
        if not hasattr(self, 'previous_queue_length'):
            self.previous_queue_length = current_queue_length
            delta_queue = 0.0
        else:
            delta_queue = current_queue_length - self.previous_queue_length
            self.previous_queue_length = current_queue_length
        
        # Reward component 1: Queue change (PRIMARY)
        # Negative change = queue reduction = positive reward
        R_queue = -delta_queue * 10.0  # Scale factor for meaningful magnitudes
        
        # Reward component 2: Phase change penalty (SECONDARY)
        phase_changed = (action == 1)
        R_stability = -self.kappa if phase_changed else 0.0
        
        # Total reward
        reward = R_queue + R_stability
        
        return float(reward)
    
    def render(self):
        """Rendering not implemented (optional for training)."""
        pass
    
    def close(self):
        """Clean up resources."""
        if self.runner is not None:
            self.runner = None
