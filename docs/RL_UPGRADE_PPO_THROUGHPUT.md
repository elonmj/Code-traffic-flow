# RL System Upgrade: PPO + Throughput Optimization

## Overview
To achieve "better" results (>25% improvement), we have upgraded the Reinforcement Learning system from **DQN** to **PPO (Proximal Policy Optimization)** and enhanced the reward function to explicitly target **Traffic Throughput**.

## Key Changes

### 1. Algorithm: PPO vs DQN
- **Why PPO?**
  - PPO (Proximal Policy Optimization) is the state-of-the-art for continuous and complex control tasks.
  - It handles stochastic environments (like traffic) better than DQN.
  - It avoids the "overestimation bias" of Q-learning.
  - **Expected Gain**: More stable learning and higher final performance.

### 2. Reward Function: Throughput Maximization
- **Old Reward**: $R = -\text{Density} - \text{SwitchCost}$
  - *Problem*: The agent could minimize density by blocking traffic upstream (creating a ghost jam outside the observation window).
- **New Reward**: $R = -\text{Density} + \mu \cdot \text{Throughput} - \text{SwitchCost}$
  - **Throughput ($Q$)**: Calculated physically using the ARZ fundamental diagram:
    $$Q = \rho \cdot v = \rho \cdot (w - P(\rho))$$
  - **Impact**: The agent is now rewarded for *moving* cars, not just having empty roads.

### 3. Configuration Updates
- **`TrainingConfig`**: Added `PPOHyperparameters` and `algorithm` field.
- **Defaults**:
  - Algorithm: `DQN`
  - Learning Rate: `3e-4`
  - Steps per Update: `2048`
  - Entropy Coefficient: `0.01` (Encourages exploration)

## How to Run
The system is configured to run **DQN** by default. PPO is available as an option.

### On Kaggle (GPU Required)
Run the standard training script:
```bash
python kaggle_runner/experiments/rl_training_victoria_island.py --timesteps 100000
```

### Local Test (GPU Required)
```bash
python kaggle_runner/experiments/rl_training_victoria_island.py --timesteps 2000
```

## Expected Results
- **Initial Phase**: The agent might perform worse than DQN as it explores.
- **Convergence**: After ~50k-100k steps, PPO should outperform DQN significantly.
- **Metric to Watch**: `mean_reward` should break the `-3.0` ceiling and approach `-2.0` or higher (depending on normalization).
