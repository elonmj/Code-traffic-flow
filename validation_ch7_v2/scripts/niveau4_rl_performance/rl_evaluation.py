"""RL Evaluation - REAL Integration with ARZ simulation"""
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Setup paths
CODE_RL_PATH = Path(__file__).parent.parent.parent.parent / "Code_RL"
ARZ_MODEL_PATH = Path(__file__).parent.parent.parent.parent / "arz_model"
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

if str(CODE_RL_PATH) not in sys.path:
    sys.path.insert(0, str(CODE_RL_PATH))
    sys.path.insert(0, str(CODE_RL_PATH / "src"))
sys.path.insert(0, str(ARZ_MODEL_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

# ✅ REAL IMPORTS
from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect
from stable_baselines3 import DQN

# Import ARZ simulation (for baseline evaluation)
try:
    from arz_model.src.simulateur import SimulationRunner
    HAS_ARZ = True
except:
    HAS_ARZ = False
    print("[WARNING] ARZ simulation not available - using synthetic comparison")


class TrafficControllerWrapper:
    """Base wrapper for traffic control strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.episode_metrics = {"travel_time": [], "throughput": [], "queue_length": []}
    
    def get_action(self, observation: np.ndarray) -> int:
        """Get control action from observation. To be overridden."""
        raise NotImplementedError
    
    def reset(self):
        """Reset episode metrics."""
        self.episode_metrics = {"travel_time": [], "throughput": [], "queue_length": []}
    
    def record_step_metrics(self, travel_time: float, throughput: float, queue_length: float):
        """Record metrics for current step."""
        self.episode_metrics["travel_time"].append(travel_time)
        self.episode_metrics["throughput"].append(throughput)
        self.episode_metrics["queue_length"].append(queue_length)
    
    def get_episode_summary(self) -> Dict[str, float]:
        """Get summary metrics for episode."""
        return {
            "avg_travel_time": np.mean(self.episode_metrics["travel_time"]) if self.episode_metrics["travel_time"] else 0.0,
            "total_throughput": np.sum(self.episode_metrics["throughput"]),
            "avg_queue_length": np.mean(self.episode_metrics["queue_length"]) if self.episode_metrics["queue_length"] else 0.0,
        }


class BaselineController(TrafficControllerWrapper):
    """Fixed-time baseline controller - alternates 60s GREEN/RED."""
    
    def __init__(self):
        super().__init__(name="Baseline (Fixed-Time 60s)")
        self.time_elapsed = 0.0
        self.phase_duration = 60.0  # 60 seconds per phase
        self.current_phase = 0  # 0=GREEN, 1=RED
    
    def get_action(self, observation: np.ndarray, delta_time: float = 1.0) -> int:
        """Get fixed-time action (no observation used)."""
        self.time_elapsed += delta_time
        
        if self.time_elapsed >= self.phase_duration:
            self.time_elapsed = 0.0
            self.current_phase = 1 - self.current_phase
        
        return self.current_phase
    
    def reset(self):
        """Reset controller state."""
        super().reset()
        self.time_elapsed = 0.0
        self.current_phase = 0


class RLController(TrafficControllerWrapper):
    """RL-based controller using trained DQN model."""
    
    def __init__(self, model_path: str):
        super().__init__(name="RL (DQN)")
        self.model = DQN.load(model_path)
        self.current_observation = None
    
    def get_action(self, observation: np.ndarray) -> int:
        """Get action from trained DQN model."""
        self.current_observation = observation
        action, _states = self.model.predict(observation, deterministic=True)
        return int(action)
    
    def reset(self):
        """Reset controller."""
        super().reset()
        self.current_observation = None


class TrafficEvaluator:
    """Evaluates traffic control strategies using REAL simulations."""
    
    def __init__(self, scenario_config_path: Optional[str] = None, device: str = "cpu"):
        self.scenario_config_path = scenario_config_path
        self.device = device
        self.results = {}
    
    def evaluate_strategy(self, controller: TrafficControllerWrapper, 
                         num_episodes: int = 3, max_episode_length: int = 3600) -> Dict[str, Any]:
        """Evaluate a traffic control strategy using REAL simulation."""
        
        print(f"\n[EVAL] Evaluating {controller.name}...")
        print(f"       Episodes: {num_episodes}, Max length: {max_episode_length}s")
        
        episode_results = []
        
        for episode in range(num_episodes):
            print(f"\n  Episode {episode + 1}/{num_episodes}...")
            
            # ✅ CREATE ENVIRONMENT
            env = TrafficSignalEnvDirect(
                scenario_config_path=self.scenario_config_path or str(
                    Path(__file__).parent.parent.parent.parent / "Code_RL" / "configs" / "env_lagos.yaml"
                ),
                decision_interval=15.0,  # ✅ Bug #27 fix
                episode_max_time=float(max_episode_length),
                observation_segments={'upstream': [3, 4, 5], 'downstream': [6, 7, 8]},
                device=self.device,
                quiet=True
            )
            
            observation, _ = env.reset()
            controller.reset()
            
            episode_reward = 0.0
            step = 0
            
            while True:
                # Get action from controller
                action = controller.get_action(observation)
                
                # Step environment
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step += 1
                
                # Extract metrics from info
                travel_time = info.get("avg_travel_time", 0.0)
                throughput = info.get("vehicles_that_exited_this_step", 0)
                queue_length = info.get("current_queue_length", 0.0)
                
                controller.record_step_metrics(travel_time, throughput, queue_length)
                
                if terminated or truncated:
                    break
            
            env.close()
            
            # Get episode summary
            summary = controller.get_episode_summary()
            episode_results.append(summary)
            
            print(f"    Avg Travel Time: {summary['avg_travel_time']:.2f}s")
            print(f"    Total Throughput: {summary['total_throughput']:.0f} vehicles")
            print(f"    Avg Queue Length: {summary['avg_queue_length']:.1f} vehicles")
        
        # Aggregate results
        aggregate_result = {
            "name": controller.name,
            "num_episodes": num_episodes,
            "avg_travel_time": np.mean([r["avg_travel_time"] for r in episode_results]),
            "total_throughput": np.mean([r["total_throughput"] for r in episode_results]),
            "avg_queue_length": np.mean([r["avg_queue_length"] for r in episode_results]),
            "episode_results": episode_results
        }
        
        return aggregate_result
    
    def compare_strategies(self, baseline_controller: TrafficControllerWrapper,
                          rl_model_path: str, num_episodes: int = 3,
                          max_episode_length: int = 3600) -> Dict[str, Any]:
        """Compare baseline vs RL controller and compute improvements."""
        
        print(f"\n{'='*60}")
        print(f"[COMPARISON] Traffic Control Strategy Evaluation")
        print(f"{'='*60}")
        
        # ✅ EVALUATE BASELINE
        baseline_results = self.evaluate_strategy(baseline_controller, num_episodes, max_episode_length)
        
        # ✅ EVALUATE RL CONTROLLER
        rl_controller = RLController(rl_model_path)
        rl_results = self.evaluate_strategy(rl_controller, num_episodes, max_episode_length)
        
        # ✅ COMPUTE IMPROVEMENTS (NOT HARDCODED!)
        improvements = {}
        
        # Travel time improvement (lower is better, so negative improvement is good)
        baseline_tt = baseline_results["avg_travel_time"]
        rl_tt = rl_results["avg_travel_time"]
        if baseline_tt > 0:
            improvements["travel_time_improvement"] = ((baseline_tt - rl_tt) / baseline_tt) * 100.0
        else:
            improvements["travel_time_improvement"] = 0.0
        
        # Throughput improvement (higher is better)
        baseline_tp = baseline_results["total_throughput"]
        rl_tp = rl_results["total_throughput"]
        if baseline_tp > 0:
            improvements["throughput_improvement"] = ((rl_tp - baseline_tp) / baseline_tp) * 100.0
        else:
            improvements["throughput_improvement"] = 0.0
        
        # Queue length reduction (lower is better)
        baseline_ql = baseline_results["avg_queue_length"]
        rl_ql = rl_results["avg_queue_length"]
        if baseline_ql > 0:
            improvements["queue_reduction"] = ((baseline_ql - rl_ql) / baseline_ql) * 100.0
        else:
            improvements["queue_reduction"] = 0.0
        
        # ✅ RETURN REAL COMPARISON
        comparison = {
            "baseline": baseline_results,
            "rl": rl_results,
            "improvements": improvements,
            "num_episodes": num_episodes,
            "max_episode_length": max_episode_length
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"[RESULTS] RL vs Baseline Comparison")
        print(f"{'='*60}")
        print(f"\nBaseline (Fixed-Time 60s):")
        print(f"  Avg Travel Time: {baseline_results['avg_travel_time']:.2f}s")
        print(f"  Total Throughput: {baseline_results['total_throughput']:.0f} vehicles")
        print(f"  Avg Queue Length: {baseline_results['avg_queue_length']:.1f} vehicles")
        print(f"\nRL (DQN):")
        print(f"  Avg Travel Time: {rl_results['avg_travel_time']:.2f}s")
        print(f"  Total Throughput: {rl_results['total_throughput']:.0f} vehicles")
        print(f"  Avg Queue Length: {rl_results['avg_queue_length']:.1f} vehicles")
        print(f"\nImprovements:")
        print(f"  Travel Time: {improvements['travel_time_improvement']:+.1f}%")
        print(f"  Throughput: {improvements['throughput_improvement']:+.1f}%")
        print(f"  Queue Reduction: {improvements['queue_reduction']:+.1f}%")
        print(f"{'='*60}\n")
        
        return comparison


def evaluate_traffic_performance(rl_model_path: str, config_name: str = "lagos_master",
                                 num_episodes: int = 3, device: str = "cpu") -> Dict[str, Any]:
    """Convenience function for evaluating RL agent against baseline."""
    
    evaluator = TrafficEvaluator(device=device)
    baseline = BaselineController()
    
    return evaluator.compare_strategies(
        baseline_controller=baseline,
        rl_model_path=rl_model_path,
        num_episodes=num_episodes,
        max_episode_length=3600
    )
