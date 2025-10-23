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
            scenario_config_path: Path to ARZ scenario YAML config
            base_config_path: Path to ARZ base config (default: arz_model/config/config_base.yml)
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
        
        # 🔍 BUG #35 DIAGNOSTIC: Log queue calculation details
        if not hasattr(self, '_queue_log_count'):
            self._queue_log_count = 0
        if self._queue_log_count < 5 or self.episode_step % 10 == 0:  # First 5 steps + every 10th
            print(f"[QUEUE_DIAGNOSTIC] ===== Step {self.episode_step} t={self.runner.t:.1f}s =====")
            print(f"[QUEUE_DIAGNOSTIC] Observation shape: {observation.shape}, n_segments: {n_segments}")
            print(f"[QUEUE_DIAGNOSTIC] Normalized obs[0:4]: {observation[:4]} (rho_m, v_m, rho_c, v_c)")
            print(f"[QUEUE_DIAGNOSTIC] Denorm factors: rho_max_m={self.rho_max_m:.3f}, v_free_m={self.v_free_m:.2f}")
            print(f"[QUEUE_DIAGNOSTIC] velocities_m (m/s): {velocities_m}")
            print(f"[QUEUE_DIAGNOSTIC] velocities_c (m/s): {velocities_c}")
            print(f"[QUEUE_DIAGNOSTIC] densities_m (veh/m): {densities_m}")
            print(f"[QUEUE_DIAGNOSTIC] densities_c (veh/m): {densities_c}")
            self._queue_log_count += 1
        
        # Define queue threshold: vehicles slower than fleet average are queued
        # ✅ BUG #31 FIX: Compute ADAPTIVE threshold based on actual observed velocities
        # Problem: Static threshold (5.0 m/s or 70% v_free) didn't work because:
        #   - Scenario may not have congestion at all
        #   - Static values miss relative congestion (e.g., 9m/s is fast normally but slow if avg is 11m/s)
        # Solution: Use fleet-wide average velocity as dynamic threshold
        #   - Queued = vehicles with v < avg(velocities_in_network)
        #   - This captures RELATIVE congestion regardless of scenario intensity
        avg_velocity = (np.mean(velocities_m) + np.mean(velocities_c)) / 2.0
        # Safety: if average is very low (unexpected), fall back to 80% of average
        QUEUE_SPEED_THRESHOLD = avg_velocity * 0.85  # Queued if slower than 85% of average
        
        # Count queued vehicles (density where v < threshold)
        queued_m = densities_m[velocities_m < QUEUE_SPEED_THRESHOLD]
        queued_c = densities_c[velocities_c < QUEUE_SPEED_THRESHOLD]
        
        # 🔍 BUG #35 DIAGNOSTIC: Log threshold check results
        if self._queue_log_count <= 5 or self.episode_step % 10 == 0:
            print(f"[QUEUE_DIAGNOSTIC] Threshold: {QUEUE_SPEED_THRESHOLD} m/s")
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
            f"t={self.runner.t:.1f}s "
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
