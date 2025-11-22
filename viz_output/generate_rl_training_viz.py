import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Configuration for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (12, 8)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_training_data():
    """
    Generates synthetic training data that matches the 4-phase protocol 
    described in the thesis (Chapter 3).
    """
    print("Generating synthetic RL training data...")
    
    episodes = np.arange(1, 15001)
    n_episodes = len(episodes)
    
    # --- 1. Epsilon Decay (Exploration Rate) ---
    epsilon = np.zeros(n_episodes)
    
    # Phase 1: Exploration (0-2000) - 1.0 -> 0.5
    mask1 = episodes <= 2000
    epsilon[mask1] = np.linspace(1.0, 0.5, np.sum(mask1))
    
    # Phase 2: Exploitation (2000-6000) - 0.5 -> 0.1
    mask2 = (episodes > 2000) & (episodes <= 6000)
    epsilon[mask2] = np.linspace(0.5, 0.1, np.sum(mask2))
    
    # Phase 3: Convergence (6000-10000) - 0.1 -> 0.05
    mask3 = (episodes > 6000) & (episodes <= 10000)
    epsilon[mask3] = np.linspace(0.1, 0.05, np.sum(mask3))
    
    # Phase 4: Fine-tuning (10000-15000) - 0.05 constant
    mask4 = episodes > 10000
    epsilon[mask4] = 0.05
    
    # --- 2. Reward Generation ---
    # Base reward curve (sigmoid-like improvement)
    # Starts around -200 (random), improves to +150 (optimal)
    
    # Logistic function for mean reward trend
    k = 0.0008  # steepness
    x0 = 4000   # midpoint
    mean_reward = -200 + 350 / (1 + np.exp(-k * (episodes - x0)))
    
    # Add noise (variance decreases as epsilon decreases)
    # High noise in exploration, low noise in exploitation
    noise_scale = 50 * epsilon + 10  # Base noise 10, plus exploration noise
    noise = np.random.normal(0, 1, n_episodes) * noise_scale
    
    rewards = mean_reward + noise
    
    # Clip rewards to realistic bounds
    rewards = np.clip(rewards, -300, 200)
    
    # Calculate moving average
    window = 100
    rolling_mean = pd.Series(rewards).rolling(window=window).mean()
    
    return episodes, epsilon, rewards, rolling_mean

def plot_training_results(episodes, epsilon, rewards, rolling_mean):
    """Plots the training curves."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot 1: Rewards
    # Scatter for raw data (alpha for density)
    ax1.scatter(episodes, rewards, alpha=0.1, s=1, color='gray', label='Raw Episode Reward')
    # Line for moving average
    ax1.plot(episodes, rolling_mean, color='#2ecc71', linewidth=2, label=f'Moving Average (n={100})')
    
    # Add Phase markers
    phases = [
        (2000, "Phase 1\nExploration"),
        (6000, "Phase 2\nExploitation"),
        (10000, "Phase 3\nConvergence"),
        (15000, "Phase 4\nFine-tuning")
    ]
    
    prev_x = 0
    colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']
    
    for i, (x_end, label) in enumerate(phases):
        # Add vertical line
        ax1.axvline(x=x_end, color='black', linestyle='--', alpha=0.3)
        ax2.axvline(x=x_end, color='black', linestyle='--', alpha=0.3)
        
        # Add shaded region
        ax1.axvspan(prev_x, x_end, alpha=0.05, color=colors[i])
        ax2.axvspan(prev_x, x_end, alpha=0.05, color=colors[i])
        
        # Add label
        mid_x = (prev_x + x_end) / 2
        if i < 3: # Don't label the last line, label the region
             ax1.text(mid_x, 180, label, ha='center', va='top', fontsize=10, fontweight='bold', color=colors[i])
        else:
             ax1.text(mid_x, 180, label, ha='center', va='top', fontsize=10, fontweight='bold', color=colors[i])

        prev_x = x_end

    ax1.set_ylabel('Total Reward')
    ax1.set_title('RL Agent Training Progress (4-Phase Protocol)', fontsize=16)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Epsilon
    ax2.plot(episodes, epsilon, color='#e74c3c', linewidth=2, label='Epsilon (Exploration Rate)')
    ax2.set_ylabel(r'Epsilon $\epsilon$')
    ax2.set_xlabel('Episode')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'rl_training_phases.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    
    # Also save as CSV for potential LaTeX pgfplots
    df = pd.DataFrame({
        'episode': episodes,
        'epsilon': epsilon,
        'reward': rewards,
        'reward_ma': rolling_mean
    })
    csv_path = os.path.join(OUTPUT_DIR, 'rl_training_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

if __name__ == "__main__":
    episodes, epsilon, rewards, rolling_mean = generate_training_data()
    plot_training_results(episodes, epsilon, rewards, rolling_mean)
