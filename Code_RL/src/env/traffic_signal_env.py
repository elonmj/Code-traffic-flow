"""
Traffic Signal RL Environment

Gymnasium-compatible environment for training RL agents to control
traffic signals using an external ARZ simulator.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

from endpoint.client import ARZEndpointClient, SimulationState
from signals.controller import SignalController

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    dt_decision: float = 10.0          # RL decision interval
    episode_length: int = 3600         # Episode length in seconds
    max_steps: int = 360               # Max steps per episode
    
    # Normalization parameters
    rho_max_motorcycles: float = 300.0
    rho_max_cars: float = 150.0
    v_free_motorcycles: float = 40.0
    v_free_cars: float = 50.0
    queue_max: float = 400.0
    phase_time_max: float = 120.0
    
    # Reward weights
    w_wait_time: float = 1.0
    w_queue_length: float = 0.5
    w_stops: float = 0.3
    w_switch_penalty: float = 0.1
    w_throughput: float = 0.8
    reward_clip: Tuple[float, float] = (-10.0, 10.0)
    stop_speed_threshold: float = 5.0  # km/h
    
    # Observation settings
    ewma_alpha: float = 0.1
    include_phase_timing: bool = True
    include_queues: bool = True


class TrafficSignalEnv(gym.Env):
    """
    RL Environment for traffic signal control
    
    Action Space: Discrete(2)
        0: Maintain current phase
        1: Switch to next phase
        
    Observation Space: Box (normalized float32)
        For each branch: [rho_m/rho_max, v_m/v_free, rho_c/rho_max, v_c/v_free, queue/queue_max]
        Plus: [phase_one_hot..., phase_time_normalized]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        endpoint_client: ARZEndpointClient,
        signal_controller: SignalController, 
        config: EnvironmentConfig,
        branch_ids: List[str],
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.endpoint_client = endpoint_client
        self.signal_controller = signal_controller
        self.config = config
        self.branch_ids = branch_ids
        self.render_mode = render_mode
        
        # Timing
        self.k_steps = int(config.dt_decision / 0.5)  # Assuming 0.5s ARZ timestep
        
        # Action space: 0=maintain, 1=switch
        self.action_space = spaces.Discrete(2)
        
        # Observation space
        self._setup_observation_space()
        
        # Episode tracking
        self.current_step = 0
        self.current_time = 0.0
        self.episode_rewards = []
        self.episode_kpis = []
        
        # State tracking
        self.last_state = None
        self.last_observation = None
        self.phase_switches = 0
        
        logger.info(f"Initialized TrafficSignalEnv with {len(branch_ids)} branches")
    
    def _setup_observation_space(self):
        """Setup observation space dimensions"""
        # Per branch: rho_m, v_m, rho_c, v_c, (queue_len if enabled)
        obs_per_branch = 5 if self.config.include_queues else 4
        
        # Total branches
        branch_obs_size = len(self.branch_ids) * obs_per_branch
        
        # Phase information
        num_phases = len(self.signal_controller.phases)
        phase_obs_size = num_phases  # one-hot encoding
        
        # Phase timing (if enabled)
        timing_obs_size = 1 if self.config.include_phase_timing else 0
        
        total_obs_size = branch_obs_size + phase_obs_size + timing_obs_size
        
        # All observations normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0, 
            shape=(total_obs_size,),
            dtype=np.float32
        )
        
        logger.info(f"Observation space: {total_obs_size} dimensions "
                   f"({branch_obs_size} branch + {phase_obs_size} phase + {timing_obs_size} timing)")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset ARZ simulator
        scenario = options.get("scenario") if options else None
        initial_state, timestamp = self.endpoint_client.reset(scenario=scenario, seed=seed)
        
        # Reset signal controller
        self.signal_controller.reset(timestamp)
        
        # Reset episode tracking
        self.current_step = 0
        self.current_time = timestamp
        self.episode_rewards = []
        self.episode_kpis = []
        self.phase_switches = 0
        
        # Set initial signal plan
        signal_plan = self.signal_controller.get_signal_plan()
        self.endpoint_client.set_signal(signal_plan)
        
        # Build initial observation
        self.last_state = initial_state
        observation = self._build_observation(initial_state, timestamp)
        self.last_observation = observation
        
        info = {
            "timestamp": timestamp,
            "phase_id": self.signal_controller.current_phase_id,
            "step": self.current_step
        }
        
        logger.info(f"Environment reset at t={timestamp:.1f}")
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step"""
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Update signal controller
        controller_info = self.signal_controller.update(self.current_time)
        
        # Process RL action through safety layer
        action_taken = self._process_action(action, self.current_time)
        
        # Step ARZ simulator
        new_state, new_time = self.endpoint_client.step(
            dt=self.config.dt_decision / self.k_steps, 
            repeat_k=self.k_steps
        )
        
        # Update environment state
        self.current_time = new_time
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(self.last_state, new_state, action, action_taken)
        self.episode_rewards.append(reward)
        
        # Build observation
        observation = self._build_observation(new_state, new_time)
        
        # Check termination
        terminated = self.current_step >= self.config.max_steps
        truncated = False  # Could add other truncation conditions
        
        # Calculate KPIs
        kpis = self._calculate_kpis(new_state)
        self.episode_kpis.append(kpis)
        
        # Build info dict
        info = {
            "timestamp": new_time,
            "phase_id": self.signal_controller.current_phase_id,
            "step": self.current_step,
            "action_requested": action,
            "action_taken": action_taken,
            "phase_switches": self.phase_switches,
            "controller_info": controller_info,
            "kpis": kpis,
            "reward_components": self._get_reward_components(self.last_state, new_state, action, action_taken)
        }
        
        # Update state
        self.last_state = new_state
        self.last_observation = observation
        
        return observation, reward, terminated, truncated, info
    
    def _process_action(self, action: int, timestamp: float) -> int:
        """Process RL action through safety layer"""
        if action == 1:  # Switch request
            if self.signal_controller.request_phase_switch(timestamp):
                self.phase_switches += 1
                return 1
            else:
                return 0  # Switch denied by safety layer
        else:
            return 0  # Maintain phase
    
    def _build_observation(self, state: SimulationState, timestamp: float) -> np.ndarray:
        """Build normalized observation vector"""
        obs_components = []
        
        # Branch observations
        for branch_id in self.branch_ids:
            if branch_id in state.branches:
                branch_data = state.branches[branch_id]
                
                # Normalize densities and velocities
                rho_m_norm = min(branch_data["rho_m"] / self.config.rho_max_motorcycles, 1.0)
                v_m_norm = branch_data["v_m"] / self.config.v_free_motorcycles
                rho_c_norm = min(branch_data["rho_c"] / self.config.rho_max_cars, 1.0)
                v_c_norm = branch_data["v_c"] / self.config.v_free_cars
                
                obs_components.extend([rho_m_norm, v_m_norm, rho_c_norm, v_c_norm])
                
                # Add queue length if enabled
                if self.config.include_queues:
                    queue_norm = min(branch_data["queue_len"] / self.config.queue_max, 1.0)
                    obs_components.append(queue_norm)
            else:
                # Missing branch data - use zeros
                size = 5 if self.config.include_queues else 4
                obs_components.extend([0.0] * size)
        
        # Phase one-hot encoding
        num_phases = len(self.signal_controller.phases)
        phase_onehot = [0.0] * num_phases
        if self.signal_controller.current_phase_id < num_phases:
            phase_onehot[self.signal_controller.current_phase_id] = 1.0
        obs_components.extend(phase_onehot)
        
        # Phase timing
        if self.config.include_phase_timing:
            phase_duration = self.signal_controller.get_phase_duration(timestamp)
            phase_time_norm = min(phase_duration / self.config.phase_time_max, 1.0)
            obs_components.append(phase_time_norm)
        
        # Apply EWMA smoothing if enabled
        observation = np.array(obs_components, dtype=np.float32)
        if self.last_observation is not None and self.config.ewma_alpha < 1.0:
            alpha = self.config.ewma_alpha
            observation = alpha * observation + (1 - alpha) * self.last_observation
        
        return observation
    
    def _calculate_reward(
        self, 
        old_state: SimulationState, 
        new_state: SimulationState,
        action_requested: int,
        action_taken: int
    ) -> float:
        """Calculate reward based on traffic performance"""
        
        # Wait time penalty (estimated from queue lengths and velocities)
        wait_time_penalty = 0.0
        queue_penalty = 0.0
        throughput_reward = 0.0
        stops_penalty = 0.0
        
        for branch_id in self.branch_ids:
            if branch_id in new_state.branches:
                branch_data = new_state.branches[branch_id]
                
                # Queue length penalty
                queue_penalty += branch_data["queue_len"]
                
                # Throughput reward (flow)
                throughput_reward += branch_data["flow"]
                
                # Stops penalty (low velocity)
                if branch_data["v_m"] < self.config.stop_speed_threshold:
                    stops_penalty += branch_data["rho_m"]
                if branch_data["v_c"] < self.config.stop_speed_threshold:
                    stops_penalty += branch_data["rho_c"]
                
                # Wait time estimation (queue / flow rate)
                flow_rate = max(branch_data["flow"], 0.1)  # Avoid division by zero
                wait_time_penalty += branch_data["queue_len"] / flow_rate
        
        # Switch penalty (only if action was actually taken)
        switch_penalty = 1.0 if (action_requested == 1 and action_taken == 1) else 0.0
        
        # Combine components
        reward = (
            -self.config.w_wait_time * wait_time_penalty
            - self.config.w_queue_length * queue_penalty 
            - self.config.w_stops * stops_penalty
            - self.config.w_switch_penalty * switch_penalty
            + self.config.w_throughput * throughput_reward
        )
        
        # Normalize and clip
        reward = reward / 1000.0  # Scale down
        reward = np.clip(reward, self.config.reward_clip[0], self.config.reward_clip[1])
        
        return float(reward)
    
    def _get_reward_components(
        self,
        old_state: SimulationState,
        new_state: SimulationState, 
        action_requested: int,
        action_taken: int
    ) -> Dict[str, float]:
        """Get detailed reward breakdown for analysis"""
        components = {
            "wait_time": 0.0,
            "queue_length": 0.0,
            "stops": 0.0,
            "switch_penalty": 0.0,
            "throughput": 0.0
        }
        
        for branch_id in self.branch_ids:
            if branch_id in new_state.branches:
                branch_data = new_state.branches[branch_id]
                components["queue_length"] += branch_data["queue_len"]
                components["throughput"] += branch_data["flow"]
                
                if branch_data["v_m"] < self.config.stop_speed_threshold:
                    components["stops"] += branch_data["rho_m"]
                if branch_data["v_c"] < self.config.stop_speed_threshold:
                    components["stops"] += branch_data["rho_c"]
                
                flow_rate = max(branch_data["flow"], 0.1)
                components["wait_time"] += branch_data["queue_len"] / flow_rate
        
        if action_requested == 1 and action_taken == 1:
            components["switch_penalty"] = 1.0
        
        return components
    
    def _calculate_kpis(self, state: SimulationState) -> Dict[str, float]:
        """Calculate Key Performance Indicators"""
        total_queue = 0.0
        total_throughput = 0.0
        avg_speed_m = 0.0
        avg_speed_c = 0.0
        total_vehicles = 0.0
        
        for branch_id in self.branch_ids:
            if branch_id in state.branches:
                branch_data = state.branches[branch_id]
                total_queue += branch_data["queue_len"]
                total_throughput += branch_data["flow"]
                
                # Weight speeds by density
                avg_speed_m += branch_data["v_m"] * branch_data["rho_m"]
                avg_speed_c += branch_data["v_c"] * branch_data["rho_c"]
                total_vehicles += branch_data["rho_m"] + branch_data["rho_c"]
        
        # Avoid division by zero
        if total_vehicles > 0:
            avg_speed_overall = (avg_speed_m + avg_speed_c) / total_vehicles
        else:
            avg_speed_overall = 0.0
        
        return {
            "total_queue_length": total_queue,
            "total_throughput": total_throughput,
            "avg_speed": avg_speed_overall,
            "phase_switches": self.phase_switches
        }
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics for completed episode"""
        if not self.episode_rewards:
            return {}
        
        # Aggregate KPIs
        avg_kpis = {}
        if self.episode_kpis:
            for key in self.episode_kpis[0].keys():
                avg_kpis[f"avg_{key}"] = np.mean([kpi[key] for kpi in self.episode_kpis])
                avg_kpis[f"max_{key}"] = np.max([kpi[key] for kpi in self.episode_kpis])
        
        return {
            "episode_length": self.current_step,
            "total_reward": sum(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards),
            "phase_switches": self.phase_switches,
            **avg_kpis
        }
    
    def render(self):
        """Render environment (placeholder)"""
        if self.render_mode == "human":
            print(f"Step {self.current_step}, Phase {self.signal_controller.current_phase_id}, "
                  f"Time {self.current_time:.1f}s")
        
        return None
    
    def close(self):
        """Clean up resources"""
        pass
