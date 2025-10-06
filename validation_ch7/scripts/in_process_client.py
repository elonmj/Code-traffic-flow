#!/usr/bin/env python3
"""
In-Process ARZ Client for Standalone Validation

This module provides an adapter that implements the ARZ Endpoint Client interface.
Instead of communicating with a remote server, it encapsulates a real
`SimulationRunner` from `arz_model` to run simulations locally within
the same process. This allows for autonomous, fast, and deterministic
validation of RL environments that expect an endpoint client, while using the
actual physics of the ARZ model.
"""

import numpy as np
from pathlib import Path
import sys

# Add project root to path to allow imports from arz_model
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from arz_model.simulation.runner import SimulationRunner


class InProcessARZClient:
    """
    An adapter client that simulates the ARZ endpoint by running a local
    `SimulationRunner` instance. It conforms to the interface expected by
    the RL environment.
    """

    def __init__(self, scenario_config_path: str):
        """
        Initializes the in-process client.

        Args:
            scenario_config_path (str): Path to the YAML scenario file for the simulation.
        """
        self.scenario_config_path = scenario_config_path
        self.simulation_runner = None
        self.state_history = []
        self.t = 0.0
        self._initialize_simulator()

    def _initialize_simulator(self):
        """Initializes or re-initializes the internal SimulationRunner."""
        base_config_path = str(project_root / "config" / "config_base.yml")
        try:
            self.simulation_runner = SimulationRunner(
                scenario_config_path=self.scenario_config_path,
                base_config_path=base_config_path,
                quiet=True,
                device='cpu'  # Use CPU for deterministic validation
            )
            self.t = 0.0
            self.state_history = [self.simulation_runner.U.copy()]
            print("  [InProcessClient] Internal SimulationRunner initialized.")
        except Exception as e:
            print(f"  [InProcessClient ERROR] Failed to initialize SimulationRunner: {e}")
            self.simulation_runner = None

    def reset(self):
        """Resets the internal simulator to its initial state."""
        print("  [InProcessClient] Resetting internal simulator.")
        self._initialize_simulator()
        return self.get_observation()

    def step(self, action: int):
        """
        Performs a simulation step by directly driving the internal runner.

        The `action` from the RL agent is used to control a parameter
        of the internal simulator (e.g., a traffic light phase). The simulator
        is then run for a fixed duration.

        Args:
            action (int): The action from the RL agent.

        Returns:
            A tuple (observation, reward, done, info).
        """
        if not self.simulation_runner:
            raise RuntimeError("InProcessClient's internal simulator is not initialized.")

        # --- Apply Action ---
        # Example: action 0 = red light (low Vmax), action 1 = green light (high Vmax)
        # This simulates the effect of a traffic light on the road segment.
        base_vmax = 25.0  # m/s
        vmax_multiplier = 0.1 if action == 0 else 1.0
        self.simulation_runner.params.Vmax_c['default'] = base_vmax * vmax_multiplier
        self.simulation_runner.params.Vmax_m['default'] = base_vmax * vmax_multiplier

        # --- Run Simulation for a fixed duration (e.g., control interval) ---
        step_duration = 60.0  # Simulate 60 seconds of traffic per agent step
        t_end = self.t + step_duration
        while self.simulation_runner.t < t_end:
            self.simulation_runner.run_step()
        
        self.t = self.simulation_runner.t
        self.state_history.append(self.simulation_runner.U.copy())

        # --- Calculate Reward and Observation ---
        observation = self.get_observation()
        # Reward is higher for higher average speed (less congestion)
        reward = np.mean(self.simulation_runner.U[1, :]) + np.mean(self.simulation_runner.U[3, :])
        done = self.t >= 3600.0  # End after 1 hour

        return observation, reward, done, {}

    def get_observation(self):
        """Extracts an observation vector from the current simulator state."""
        if not self.simulation_runner:
            return np.zeros(3) # Default observation
        
        U = self.simulation_runner.U
        avg_density = np.mean(U[0, :] + U[2, :])
        avg_momentum = np.mean(U[1, :] + U[3, :])
        avg_speed = avg_momentum / (avg_density + 1e-8)
        return np.array([avg_density, avg_speed, 0.0]) # obs: density, speed, queue_length (mock)

    def close(self):
        """Cleans up resources."""
        self.simulation_runner = None
        print("  [InProcessClient] Closed.")