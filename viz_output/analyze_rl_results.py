import matplotlib.pyplot as plt
import re
import numpy as np
import os

def parse_log(log_path):
    # Training data
    train_timesteps = []
    train_rewards = []
    
    # Eval data
    eval_timesteps = []
    eval_rewards = []
    
    current_ep_rew = None
    
    with open(log_path, 'r') as f:
        for line in f:
            # Training metrics
            if "ep_rew_mean" in line:
                match = re.search(r"\|\s+ep_rew_mean\s+\|\s+([-\d.]+)\s+\|", line)
                if match:
                    current_ep_rew = float(match.group(1))
            
            if "total_timesteps" in line:
                match = re.search(r"\|\s+total_timesteps\s+\|\s+(\d+)\s+\|", line)
                if match:
                    step = int(match.group(1))
                    # If we have a recent ep_rew_mean, add it
                    if current_ep_rew is not None:
                        train_timesteps.append(step)
                        train_rewards.append(current_ep_rew)
            
            # Eval metrics
            # Log format: "Eval num_timesteps=1000, episode_reward=-6.00 +/- 0.00"
            if "Eval num_timesteps=" in line:
                match = re.search(r"Eval num_timesteps=(\d+), episode_reward=([-\d.]+)", line)
                if match:
                    eval_timesteps.append(int(match.group(1)))
                    eval_rewards.append(float(match.group(2)))
                    
    return train_timesteps, train_rewards, eval_timesteps, eval_rewards

def main():
    log_path = "kaggle/results/rl-training-runner/test_log.txt"
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    train_steps, train_rews, eval_steps, eval_rews = parse_log(log_path)
    
    # Baseline estimation
    # Density 0.1 -> -3.0 (for 30 steps)
    # Switch cost -> -1.0 (for 10 switches)
    baseline_reward = -4.0
    
    plt.figure(figsize=(12, 7))
    
    # Plot Training
    plt.plot(train_steps, train_rews, label='Training Mean Reward (Moving Avg)', color='blue', alpha=0.6)
    
    # Plot Eval
    if eval_steps:
        plt.plot(eval_steps, eval_rews, 'ro-', label='Evaluation Reward (Deterministic)', linewidth=2)
    
    # Plot Baseline
    plt.axhline(y=baseline_reward, color='green', linestyle='--', label='Baseline (Fixed Time) Estimate: -4.0')
    
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('RL Agent Performance: Training vs Evaluation vs Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = "viz_output/rl_training_progress.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    if eval_rews:
        print(f"Final Eval Reward: {eval_rews[-1]}")
    print(f"Baseline Reward: {baseline_reward}")
    if eval_rews:
        print(f"Improvement vs Baseline: {eval_rews[-1] - baseline_reward:.2f}")

if __name__ == "__main__":
    main()
