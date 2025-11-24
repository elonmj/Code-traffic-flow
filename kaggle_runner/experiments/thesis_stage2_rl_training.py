"""
Thesis Stage 2: RL Training & Baseline Evaluation

This script generates results for Section 8 of the thesis:
- Part 2A: Baseline Evaluation (Fixed-Time Control)
- Part 2B: RL Agent Training (DQN)
- Part 2C: RL Agent Evaluation & Comparison

Usage:
    # Via Kaggle executor
    python kaggle_runner/executor.py --target kaggle_runner/experiments/thesis_stage2_rl_training.py --timeout 7200
    
    # Local test
    python kaggle_runner/experiments/thesis_stage2_rl_training.py --timesteps 1000 --episodes 2
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 80)
print("THESIS STAGE 2: RL TRAINING & EVALUATION")
print("=" * 80)

from Code_RL.src.utils.config import RLConfigBuilder
from Code_RL.src.env.traffic_signal_env_direct_v3 import TrafficSignalEnvDirectV3


def create_env(scenario='victoria_island', quiet=True):
    """Create the environment"""
    rl_config = RLConfigBuilder.for_training(
        scenario=scenario,
        episode_length=None,  # Use default
        cells_per_100m=4
    )
    
    env = TrafficSignalEnvDirectV3(
        simulation_config=rl_config.arz_simulation_config,
        decision_interval=rl_config.rl_env_params.get('dt_decision', 15.0),
        observation_segment_ids=rl_config.rl_env_params.get('observation_segment_ids'),
        reward_weights=rl_config.rl_env_params.get('reward_weights'),
        quiet=quiet
    )
    return env


def evaluate_policy(env, model=None, policy_type='model', n_episodes=5, fixed_time_interval=30.0):
    """
    Evaluate a policy (Model or Fixed-Time)
    
    Args:
        env: The environment
        model: The RL model (if policy_type='model')
        policy_type: 'model' or 'fixed_time'
        n_episodes: Number of episodes
        fixed_time_interval: Interval for fixed-time switching (seconds)
    """
    print(f"\nEvaluating {policy_type} policy over {n_episodes} episodes...")
    
    metrics = {
        'rewards': [],
        'densities': [],
        'throughputs': [],
        'waiting_times': [] # If available
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        ep_reward = 0
        ep_densities = []
        ep_throughputs = []
        
        # For fixed time
        time_since_switch = 0.0
        current_action = 0
        
        while not (done or truncated):
            if policy_type == 'model':
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Fixed Time Logic
                # Toggle every fixed_time_interval
                # env.decision_interval is the step size
                time_since_switch += env.decision_interval
                if time_since_switch >= fixed_time_interval:
                    action = 1 # Switch
                    time_since_switch = 0.0
                else:
                    action = 0 # Keep
            
            obs, reward, done, truncated, info = env.step(action)
            
            ep_reward += reward
            
            if 'avg_density' in info:
                ep_densities.append(info['avg_density'])
            if 'throughput' in info:
                ep_throughputs.append(info['throughput'])
                
        metrics['rewards'].append(ep_reward)
        metrics['densities'].append(np.mean(ep_densities) if ep_densities else 0.0)
        metrics['throughputs'].append(np.sum(ep_throughputs) if ep_throughputs else 0.0)
        
        print(f"  Episode {ep+1}: Reward={ep_reward:.2f}, Avg Density={metrics['densities'][-1]:.4f}")
        
    results = {
        'mean_reward': float(np.mean(metrics['rewards'])),
        'std_reward': float(np.std(metrics['rewards'])),
        'mean_density': float(np.mean(metrics['densities'])),
        'mean_throughput': float(np.mean(metrics['throughputs']))
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Thesis Stage 2: RL Training')
    parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps (default: 50K)')
    parser.add_argument('--episodes', type=int, default=5, help='Evaluation episodes (default: 5)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif os.path.exists('/kaggle/working'):
        output_dir = Path('/kaggle/working/thesis_stage2')
    else:
        output_dir = project_root / 'results' / 'thesis_stage2'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Baseline Evaluation
    print("\n" + "=" * 60)
    print("PART 2A: BASELINE EVALUATION (FIXED TIME)")
    print("=" * 60)
    print(f"Running {args.episodes} episodes with fixed-time 30s...")
    
    env = create_env(quiet=True)
    baseline_results = evaluate_policy(
        env, 
        policy_type='fixed_time', 
        n_episodes=args.episodes,
        fixed_time_interval=30.0 # Standard 30s green/red
    )
    env.close()
    
    print(f"\nBaseline Results: {json.dumps(baseline_results, indent=2)}")
    
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
        
    # 2. RL Training (DQN)
    print("\n" + "=" * 60)
    print("PART 2B: RL TRAINING (DQN)")
    print("=" * 60)
    print(f"Training DQN for {args.timesteps} timesteps...")
    
    # Re-create env wrapped for training
    env = Monitor(create_env(quiet=True), str(output_dir))
    
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=min(1000, args.timesteps // 10),  # Adaptive
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        tensorboard_log=None  # Disable tensorboard to save time
    )
    
    print(f"Starting training at {time.strftime('%H:%M:%S')}...")
    start_time = time.time()
    model.learn(total_timesteps=args.timesteps)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    model_path = output_dir / "dqn_victoria_island_final"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # 3. RL Evaluation
    print("\n" + "=" * 60)
    print("PART 2C: RL EVALUATION")
    print("=" * 60)
    print(f"Evaluating trained DQN over {args.episodes} episodes...")
    
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
        
    # 4. Comparison Summary
    summary = {
        "baseline": baseline_results,
        "rl_dqn": rl_results,
        "improvement": {
            "reward": (rl_results['mean_reward'] - baseline_results['mean_reward']) / abs(baseline_results['mean_reward']) * 100,
            "density": (rl_results['mean_density'] - baseline_results['mean_density']) / baseline_results['mean_density'] * 100,
            "throughput": (rl_results['mean_throughput'] - baseline_results['mean_throughput']) / baseline_results['mean_throughput'] * 100
        }
    }
    
    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 80)
    print("THESIS STAGE 2: COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
