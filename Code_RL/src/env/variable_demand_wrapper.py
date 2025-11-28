"""
Variable Demand Wrapper for Traffic Signal Environment

Implements Domain Randomization by varying inflow_density between episodes.
This creates scenarios where RL can outperform Fixed-Time baselines.

Scientific Rationale (Webster's Formula):
- For stationary demand q, optimal cycle time C* minimizes delay d
- d = f(λ) is convex where λ = g/C (green ratio)
- Fixed-Time tuned to (λ*, C*) is globally optimal for constant q
- RL cannot beat this theoretical optimum; it can only match it
- BUT: When q(t) varies, Fixed-Time fails → RL excels through adaptivity
"""
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional

from arz_model.config import create_victoria_island_config
from Code_RL.src.env.traffic_signal_env_direct_v3 import TrafficSignalEnvDirectV3


class VariableDemandEnv(gym.Env):
    """
    Environment with Domain Randomization for inflow density.
    
    At each episode reset, samples a new inflow_density from U(ρ_min, ρ_max).
    This breaks stationarity and creates scenarios where RL can outperform
    Fixed-Time baselines that are calibrated for average demand.
    
    Mathematical Justification:
    - Under constant demand: FT-90s ≈ optimal (Webster formula)
    - Under variable demand: FT-90s is suboptimal at extremes
      * Low demand (ρ < 100): FT wastes green time
      * High demand (ρ > 200): FT creates residual queues
    - RL learns π(s) that adapts to observed congestion state
    
    Args:
        density_range: (min, max) for ρ_in ~ U(min, max) in veh/km
        velocity_range: (min, max) for v_in ~ U(min, max) in km/h
        default_density: Initial network density (veh/km)
        t_final: Episode duration (seconds)
        decision_interval: Time between RL decisions (seconds)
        reward_weights: {'alpha': congestion, 'kappa': switching, 'mu': throughput}
        seed: Random seed for reproducibility
        quiet: Suppress output
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        density_range: Tuple[float, float] = (80.0, 280.0),
        velocity_range: Tuple[float, float] = (30.0, 50.0),
        default_density: float = 120.0,
        t_final: float = 450.0,
        decision_interval: float = 15.0,
        reward_weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        quiet: bool = True
    ):
        super().__init__()
        
        self.density_range = density_range
        self.velocity_range = velocity_range
        self.default_density = default_density
        self.t_final = t_final
        self.decision_interval = decision_interval
        self.reward_weights = reward_weights or {'alpha': 5.0, 'kappa': 0.0, 'mu': 0.1}
        self.quiet = quiet
        
        # Domain Randomization logging (can be enabled separately from quiet)
        self.log_dr = False  # Set to True to see DR logs during training
        self._episode_count = 0  # Track episode count for periodic logging
        
        self._rng = np.random.default_rng(seed)
        self._current_inflow_density = None
        self._current_inflow_velocity = None
        
        # Create initial environment to get spaces
        initial_density = float(np.mean(density_range))
        initial_velocity = float(np.mean(velocity_range))
        self._inner_env = self._create_env_with_demand(initial_density, initial_velocity)
        
        # Copy spaces from inner environment
        self.action_space = self._inner_env.action_space
        self.observation_space = self._inner_env.observation_space
        
    def _create_env_with_demand(
        self, 
        inflow_density: float, 
        inflow_velocity: float
    ) -> TrafficSignalEnvDirectV3:
        """Create environment with specific demand parameters."""
        config = create_victoria_island_config(
            t_final=self.t_final,
            output_dt=self.decision_interval,
            cells_per_100m=4,
            default_density=self.default_density,
            inflow_density=inflow_density,
            inflow_velocity=inflow_velocity,
            use_cache=False  # Important: don't cache variable configs
        )
        config.rl_metadata = {
            'observation_segment_ids': [s.id for s in config.segments],
            'decision_interval': self.decision_interval
        }
        
        return TrafficSignalEnvDirectV3(
            simulation_config=config,
            decision_interval=self.decision_interval,
            reward_weights=self.reward_weights,
            quiet=self.quiet
        )
    
    def reset(self, seed=None, options=None):
        """Reset with randomized demand parameters (Domain Randomization).
        
        TRUE Domain Randomization: At each episode reset, samples new inflow
        parameters and applies them to the simulator BEFORE resetting state.
        
        Args:
            seed: Optional random seed
            options: Optional dict that can override demand parameters:
                - 'inflow_density': Override sampled density (veh/km)
                - 'inflow_velocity': Override sampled velocity (km/h)
        """
        # Parse options for external control of demand
        if options is None:
            options = {}
        
        # Sample new demand parameters OR use provided values
        if 'inflow_density' in options:
            self._current_inflow_density = float(options['inflow_density'])
        else:
            self._current_inflow_density = float(self._rng.uniform(*self.density_range))
            
        if 'inflow_velocity' in options:
            self._current_inflow_velocity = float(options['inflow_velocity'])
        else:
            self._current_inflow_velocity = float(self._rng.uniform(*self.velocity_range))
        
        # Ensure inner environment exists
        if self._inner_env is None:
            self._inner_env = self._create_env_with_demand(
                inflow_density=self._current_inflow_density,
                inflow_velocity=self._current_inflow_velocity
            )
            if self._inner_env is None:
                raise RuntimeError("Failed to create environment: _create_env_with_demand returned None")
        
        # ✅ CRITICAL: Update BC params BEFORE reset (TRUE Domain Randomization)
        # This modifies the inflow boundary conditions that will be applied
        # during simulation, without recreating the environment
        self._inner_env.set_inflow_conditions(
            density=self._current_inflow_density,
            velocity=self._current_inflow_velocity
        )
        
        # Reset the simulator (restores initial state but uses NEW BC params)
        try:
            obs, info = self._inner_env.reset(seed=seed, options=options)
        except Exception as e:
            # If reset fails catastrophically, try recreating environment
            if not self.quiet:
                print(f"⚠️ Reset failed, recreating environment: {e}")
            self._inner_env = self._create_env_with_demand(
                inflow_density=self._current_inflow_density,
                inflow_velocity=self._current_inflow_velocity
            )
            self._inner_env.set_inflow_conditions(
                density=self._current_inflow_density,
                velocity=self._current_inflow_velocity
            )
            obs, info = self._inner_env.reset(seed=seed, options=options)
        
        # Add demand info to info dict for logging/monitoring
        info['inflow_density'] = self._current_inflow_density
        info['inflow_velocity'] = self._current_inflow_velocity
        
        # Track episode count
        self._episode_count += 1
        
        # Log Domain Randomization (every 10 episodes or if log_dr is True)
        should_log = self.log_dr or (self._episode_count <= 5) or (self._episode_count % 50 == 0)
        if should_log:
            print(f"🎲 DR Episode {self._episode_count}: ρ={self._current_inflow_density:.0f} veh/km, v={self._current_inflow_velocity:.0f} km/h")
        
        if not self.quiet:
            print(f"📊 Episode demand: ρ={self._current_inflow_density:.1f} veh/km, v={self._current_inflow_velocity:.1f} km/h")
        
        return obs, info
    
    def step(self, action):
        """Forward step to inner environment."""
        if self._inner_env is None:
            raise RuntimeError("Cannot call step() before reset(): inner environment is None")
        return self._inner_env.step(action)
    
    def render(self):
        """Forward render to inner environment."""
        if self._inner_env is None:
            return None
        return self._inner_env.render()
    
    def close(self):
        """Close inner environment."""
        if self._inner_env is not None:
            self._inner_env.close()
    
    @property
    def current_demand(self) -> Dict[str, float]:
        """Get current episode's demand parameters."""
        return {
            'inflow_density': self._current_inflow_density or 0.0,
            'inflow_velocity': self._current_inflow_velocity or 0.0
        }
