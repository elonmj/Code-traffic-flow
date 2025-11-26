"""Quick analysis of RL training results"""
import json
import pandas as pd
import numpy as np

# Load monitor.csv
df = pd.read_csv('kaggle/results/generic-test-runner-kernel/thesis_stage2/monitor.csv', skiprows=1)

print("=" * 60)
print("TRAINING DATA ANALYSIS")
print("=" * 60)
print(f"Total training episodes: {len(df)}")
print(f"Reward mean: {df['r'].mean():.2f}")
print(f"Reward std: {df['r'].std():.2f}")
print(f"Reward min: {df['r'].min():.2f}")
print(f"Reward max: {df['r'].max():.2f}")

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

# Load evaluation results
with open('kaggle/results/generic-test-runner-kernel/thesis_stage2/baseline_results.json') as f:
    baseline = json.load(f)
    
with open('kaggle/results/generic-test-runner-kernel/thesis_stage2/rl_results.json') as f:
    rl = json.load(f)

print("\nBaseline (Fixed-Time):")
print(f"  Mean reward: {baseline['mean_reward']:.2f}")
print(f"  Std reward: {baseline['std_reward']:.2f}")

print("\nRL Agent (DQN):")
print(f"  Mean reward: {rl['mean_reward']:.2f}")
print(f"  Std reward: {rl['std_reward']:.2f}")

print("\n" + "=" * 60)
print("PROBLEM ANALYSIS")
print("=" * 60)

if rl['std_reward'] == 0.0:
    print("⚠️ RL std_reward = 0.0 is HIGHLY SUSPICIOUS!")
    print("   This means ALL evaluation episodes had EXACTLY the same reward.")
    print("   Possible causes:")
    print("   1. Only 1 evaluation episode was run")
    print("   2. Deterministic policy + identical initial state = same trajectory")
    print("   3. Bug in reward calculation")

# Check if rewards are close to training rewards
training_mean = df['r'].mean()
rl_eval_mean = rl['mean_reward']

print(f"\n📊 Training mean reward: {training_mean:.2f}")
print(f"📊 RL eval mean reward: {rl_eval_mean:.2f}")

# The baseline is MUCH higher than training average - this is suspicious
print(f"\n📊 Baseline is {(baseline['mean_reward'] - training_mean) / training_mean * 100:+.1f}% vs training mean")
print(f"📊 RL eval is {(rl_eval_mean - training_mean) / training_mean * 100:+.1f}% vs training mean")

# Check training progression
print("\n" + "=" * 60)
print("TRAINING PROGRESSION (first vs last 100 episodes)")
print("=" * 60)
first_100 = df['r'].head(100).mean()
last_100 = df['r'].tail(100).mean()
print(f"First 100 episodes mean: {first_100:.2f}")
print(f"Last 100 episodes mean: {last_100:.2f}")
print(f"Improvement: {(last_100 - first_100) / first_100 * 100:+.1f}%")
