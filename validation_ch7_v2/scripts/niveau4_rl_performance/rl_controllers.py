"""
RL Controllers - Pure domain logic for traffic signal control

Implements:
- BaselineController: Fixed-time 60s GREEN / 60s RED
- RLController: Wrapper for Stable-Baselines3 models (DQN/PPO)

CLEAN ARCHITECTURE: Domain logic only (no infrastructure).
"""

import numpy as np
from typing import Any, Dict

class BaselineController:
    """Fixed-time traffic signal controller (Baseline)."""
    
    def __init__(self, scenario_type: str, control_interval: float = 15.0):
        self.scenario_type = scenario_type
        self.control_interval = control_interval
        self.green_duration = 60.0
        self.red_duration = 60.0
        self.cycle_duration = 120.0
        self.time_step = 0
    
    def step(self, observation: Dict[str, Any]) -> int:
        cycle_position = self.time_step % self.cycle_duration
        action = 1 if cycle_position < self.green_duration else 0
        self.time_step += self.control_interval
        return action
    
    def serialize_state(self) -> Dict[str, Any]:
        return {"time_step": self.time_step}
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        self.time_step = state.get("time_step", 0)
    
    def reset(self) -> None:
        self.time_step = 0


class RLController:
    """RL Agent Controller - Wrapper for Stable-Baselines3 models."""
    
    def __init__(self, model: Any):
        self.model = model
    
    def step(self, observation: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)
    
    def reset(self) -> None:
        pass
