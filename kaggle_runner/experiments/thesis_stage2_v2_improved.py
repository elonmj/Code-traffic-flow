"""
Thesis Stage 2 v2: RL Training with Better Reward Weights
=========================================================

Key changes:
1. Increased congestion penalty (alpha: 1.0 -> 5.0)
2. Reduced throughput reward (mu: 0.5 -> 0.1) 
3. Higher phase change penalty (kappa: 0.1 -> 0.3)

This makes the RL problem more challenging and differentiates
good signal timing from bad timing.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("THESIS STAGE 2 v2: RL TRAINING WITH IMPROVED REWARD")
print("=" * 60)
print(f"Project root: {project_root}")

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from Code_RL.src.env.traffic_signal_env_direct_v3 import TrafficSignalEnvDirectV3
from arz_model.config import create_victoria_island_config


def create_env(scenario='congested_v2', quiet=True):
    """
    Create environment with improved reward weights.
    
    v2 reward weights prioritize congestion reduction over raw throughput.
    """
    if scenario == 'congested_v2':
        # HIGH CONGESTION with IMPROVED REWARD WEIGHTS
        arz_config = create_victoria_island_config(
            t_final=450.0,
            output_dt=15.0,
            cells_per_100m=4,
            default_density=80.0,   # HIGH: 80 veh/km 
            inflow_density=100.0,   # HIGH inflow
            use_cache=False
        )
        arz_config.rl_metadata = {
            'observation_segment_ids': [s.id for s in arz_config.segments],
            'decision_interval': 15.0,
        }
        
        class SimpleConfig:
            def __init__(self, arz_config):
                self.arz_simulation_config = arz_config
                self.rl_env_params = {
                    'dt_decision': 15.0,
                    'observation_segment_ids': None,
                    # IMPROVED REWARD WEIGHTS:
                    # - Higher congestion penalty (alpha) to punish density
                    # - Lower throughput reward (mu) to reduce its dominance
                    # - Higher switch penalty (kappa) to avoid random switching
                    'reward_weights': {
                        'alpha': 5.0,   # Was 1.0 - Now congestion matters MORE
                        'kappa': 0.3,   # Was 0.1 - Discourage unnecessary switching
                        'mu': 0.1       # Was 0.5 - Throughput is less dominant
                    }
                }
        
        rl_config = SimpleConfig(arz_config)
        
    elif scenario == 'original':
        # Original reward weights for comparison
        arz_config = create_victoria_island_config(
            t_final=450.0,
            output_dt=15.0,
            cells_per_100m=4,
            default_density=80.0,
            inflow_density=100.0,
            use_cache=False
        )
        arz_config.rl_metadata = {
            'observation_segment_ids': [s.id for s in arz_config.segments],
            'decision_interval': 15.0,
        }
        
        class SimpleConfig:
            def __init__(self, arz_config):
                self.arz_simulation_config = arz_config
                self.rl_env_params = {
                    'dt_decision': 15.0,
                    'observation_segment_ids': None,
                    'reward_weights': {'alpha': 1.0, 'kappa': 0.1, 'mu': 0.5}
                }
        
        rl_config = SimpleConfig(arz_config)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    env = TrafficSignalEnvDirectV3(
        simulation_config=rl_config.arz_simulation_config,
        decision_interval=rl_config.rl_env_params.get('dt_decision', 15.0),
        observation_segment_ids=rl_config.rl_env_params.get('observation_segment_ids'),
        reward_weights=rl_config.rl_env_params.get('reward_weights'),
        quiet=quiet
    )
    return env


def evaluate_policy(env, policy_type='fixed_time', model=None, n_episodes=5, 
                   fixed_time_interval=30.0):
    """
    Evaluate a policy over multiple episodes.
    """
    metrics = {'rewards': [], 'densities': [], 'throughputs': []}
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = truncated = False
        ep_reward = 0.0
        ep_densities = []
        ep_throughputs = []
        
        time_since_switch = 0.0
        current_action = 0
        
        while not (done or truncated):
            if policy_type == 'model':
                action, _ = model.predict(obs, deterministic=True)
            else:
                time_since_switch += env.decision_interval
                if time_since_switch >= fixed_time_interval:
                    action = 1
                    time_since_switch = 0.0
                else:
                    action = 0
            
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            
            if 'avg_density' in info:
                ep_densities.append(info['avg_density'])
            if 'throughput' in info:
                ep_throughputs.append(info['throughput'])
                
        metrics['rewards'].append(ep_reward)
        metrics['densities'].append(np.mean(ep_densities) if ep_densities else 0.0)
        metrics['throughputs'].append(np.sum(ep_throughputs) if ep_throughputs else 0.0)
        
        print(f"  Episode {ep+1}: Reward={ep_reward:.2f}, Density={metrics['densities'][-1]:.4f}")
        
    return {
        'mean_reward': float(np.mean(metrics['rewards'])),
        'std_reward': float(np.std(metrics['rewards'])),
        'mean_density': float(np.mean(metrics['densities'])),
        'mean_throughput': float(np.mean(metrics['throughputs'])),
        'all_rewards': [float(r) for r in metrics['rewards']]
    }


def main():
    parser = argparse.ArgumentParser(description='Thesis Stage 2 v2: RL Training')
    parser.add_argument('--timesteps', type=int, default=100000, help='Training timesteps')
    parser.add_argument('--episodes', type=int, default=5, help='Evaluation episodes')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif os.path.exists('/kaggle/working'):
        output_dir = Path('/kaggle/working/thesis_stage2_v2')
    else:
        output_dir = project_root / 'results' / 'thesis_stage2_v2'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =====================================================
    # PART 1: Baseline with NEW reward weights
    # =====================================================
    print("\n" + "=" * 60)
    print("PART 1: BASELINE (IMPROVED REWARD WEIGHTS)")
    print("=" * 60)
    print("Reward weights: alpha=5.0, kappa=0.3, mu=0.1")
    
    env = create_env(scenario='congested_v2', quiet=True)
    baseline_results = evaluate_policy(
        env, 
        policy_type='fixed_time', 
        n_episodes=args.episodes,
        fixed_time_interval=30.0
    )
    env.close()
    
    print(f"\nBaseline: {json.dumps(baseline_results, indent=2)}")
    
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    
    # =====================================================
    # PART 2: DQN Training
    # =====================================================
    print("\n" + "=" * 60)
    print("PART 2: DQN TRAINING")
    print("=" * 60)
    
    env = create_env(scenario='congested_v2', quiet=True)
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1
    )
    
    print(f"Training for {args.timesteps} timesteps...")
    start_time = time.time()
    model.learn(total_timesteps=args.timesteps)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed/60:.1f} min")
    
    model_path = output_dir / "dqn_improved_final"
    model.save(model_path)
    
    # =====================================================
    # PART 3: RL Evaluation
    # =====================================================
    print("\n" + "=" * 60)
    print("PART 3: RL EVALUATION")
    print("=" * 60)
    
    rl_results = evaluate_policy(
        env,
        model=model,
        policy_type='model',
        n_episodes=args.episodes
    )
    env.close()
    
    print(f"\nRL Results: {json.dumps(rl_results, indent=2)}")
    
    with open(output_dir / "rl_results.json", "w") as f:
        json.dump(rl_results, f, indent=2)
    
    # =====================================================
    # PART 4: Comparison
    # =====================================================
    improvement = {
        "reward": (rl_results['mean_reward'] - baseline_results['mean_reward']) / abs(baseline_results['mean_reward']) * 100,
        "density": (rl_results['mean_density'] - baseline_results['mean_density']) / baseline_results['mean_density'] * 100,
        "throughput": (rl_results['mean_throughput'] - baseline_results['mean_throughput']) / baseline_results['mean_throughput'] * 100
    }
    
    summary = {
        "baseline": baseline_results,
        "rl_dqn": rl_results,
        "improvement": improvement,
        "config": {
            "reward_weights": {"alpha": 5.0, "kappa": 0.3, "mu": 0.1},
            "timesteps": args.timesteps,
            "eval_episodes": args.episodes
        }
    }
    
    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Baseline reward: {baseline_results['mean_reward']:.2f} ± {baseline_results['std_reward']:.2f}")
    print(f"RL reward:       {rl_results['mean_reward']:.2f} ± {rl_results['std_reward']:.2f}")
    print(f"Improvement:     {improvement['reward']:+.2f}%")
    print(f"\nDensity change:  {improvement['density']:+.2f}%")
    print(f"Throughput change: {improvement['throughput']:+.2f}%")


if __name__ == "__main__":
    main()
