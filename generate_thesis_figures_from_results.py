#!/usr/bin/env python3
"""
Generate thesis figures from actual Kaggle RL training results.
Uses real data from monitor.csv, rl_results.json, and comparison_summary.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configure matplotlib for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Paths
RESULTS_DIR = "kaggle/results/generic-test-runner-kernel/thesis_stage2"
OUTPUT_DIR = "D:/Projets/Alibi/Memory/New/images/chapter3"

def load_data():
    """Load all result files from Kaggle training."""
    print("="*60)
    print("Loading Kaggle RL Training Results")
    print("="*60)
    
    # Load monitor.csv (training progress)
    monitor_path = os.path.join(RESULTS_DIR, "monitor.csv")
    print(f"\n[1] Loading {monitor_path}...")
    try:
        # Skip the first line (contains metadata in JSON format)
        df = pd.read_csv(monitor_path, skiprows=1)
        print(f"    ✓ Loaded {len(df)} episodes")
        print(f"    Columns: {list(df.columns)}")
        print(f"    Reward range: [{df['r'].min():.2f}, {df['r'].max():.2f}]")
    except Exception as e:
        print(f"    ✗ Error loading monitor.csv: {e}")
        df = None
    
    # Load comparison_summary.json
    comparison_path = os.path.join(RESULTS_DIR, "comparison_summary.json")
    print(f"\n[2] Loading {comparison_path}...")
    try:
        with open(comparison_path, 'r') as f:
            comparison = json.load(f)
        print(f"    ✓ Loaded comparison summary")
        print(f"    Keys: {list(comparison.keys())}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        comparison = None
    
    # Load rl_results.json
    rl_path = os.path.join(RESULTS_DIR, "rl_results.json")
    print(f"\n[3] Loading {rl_path}...")
    try:
        with open(rl_path, 'r') as f:
            rl_results = json.load(f)
        print(f"    ✓ Loaded RL results")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        rl_results = None
    
    # Load baseline_results.json
    baseline_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    print(f"\n[4] Loading {baseline_path}...")
    try:
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        print(f"    ✓ Loaded baseline results")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        baseline = None
    
    return df, comparison, rl_results, baseline


def generate_figure_8_training_curve(df):
    """
    Generate Figure 8: RL Training Progress Curve
    Shows episode rewards and rolling average during training.
    """
    print("\n" + "="*60)
    print("Generating Figure 8: Training Progress Curve")
    print("="*60)
    
    if df is None or df.empty:
        print("No training data available!")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    episodes = np.arange(1, len(df) + 1)
    rewards = df['r'].values
    
    # Calculate rolling average
    window = max(50, len(df) // 20)  # Adaptive window
    rolling_mean = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
    
    # Plot raw rewards (scatter with transparency)
    ax.scatter(episodes, rewards, alpha=0.3, s=10, c='#3498db', label='Episode Reward')
    
    # Plot rolling average
    ax.plot(episodes, rolling_mean, color='#e74c3c', linewidth=2.5, 
            label=f'Rolling Average (n={window})')
    
    # Add horizontal line for mean
    mean_reward = rewards.mean()
    ax.axhline(y=mean_reward, color='#2ecc71', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_reward:.1f}')
    
    # Labels and formatting
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Episode Reward')
    ax.set_title('DQN Agent Training Progress - Victoria Island Network\n(50,000 timesteps)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add training statistics as text box
    stats_text = (f'Training Statistics:\n'
                  f'Episodes: {len(df)}\n'
                  f'Mean Reward: {mean_reward:.1f}\n'
                  f'Max Reward: {rewards.max():.1f}\n'
                  f'Min Reward: {rewards.min():.1f}\n'
                  f'Std Dev: {rewards.std():.1f}')
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'fig_8_training_curve.png')
    plt.savefig(output_path)
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    return mean_reward


def generate_figure_8_comparison(comparison, rl_results, baseline):
    """
    Generate Figure 8: RL vs Baseline Comparison Bar Chart
    """
    print("\n" + "="*60)
    print("Generating Figure 8: RL vs Baseline Comparison")
    print("="*60)
    
    if comparison is None:
        print("No comparison data available!")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Extract metrics
    metrics = ['reward', 'density', 'throughput']
    titles = ['Average Reward', 'Average Density', 'Average Throughput']
    units = ['', '', 'vehicles/hour']
    
    baseline_vals = [
        comparison['baseline']['mean_reward'],
        comparison['baseline']['mean_density'],
        comparison['baseline']['mean_throughput']
    ]
    
    rl_vals = [
        comparison['rl_dqn']['mean_reward'],
        comparison['rl_dqn']['mean_density'],
        comparison['rl_dqn']['mean_throughput']
    ]
    
    improvements = [
        comparison['improvement']['reward'],
        comparison['improvement']['density'],
        comparison['improvement']['throughput']
    ]
    
    colors = ['#3498db', '#e74c3c']
    
    for i, (ax, title, unit, bl, rl, imp) in enumerate(zip(axes, titles, units, 
                                                           baseline_vals, rl_vals, improvements)):
        x = np.arange(2)
        values = [bl, rl]
        bars = ax.bar(x, values, color=colors, width=0.6, edgecolor='black', linewidth=1)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Baseline\n(Fixed-Time)', 'RL Agent\n(DQN)'])
        ax.set_ylabel(f'{title} {unit}' if unit else title)
        ax.set_title(f'{title}\n({imp:+.2f}%)', fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('RL Agent Performance vs Fixed-Time Baseline\nVictoria Island Network (50k timesteps)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'fig_8_comparison.png')
    plt.savefig(output_path)
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_figure_combined(df, comparison):
    """
    Generate a combined figure with both training curve and comparison.
    """
    print("\n" + "="*60)
    print("Generating Combined Figure")
    print("="*60)
    
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
    
    # Top: Training curve (spans all 3 columns)
    ax_train = fig.add_subplot(gs[0, :])
    
    if df is not None and not df.empty:
        episodes = np.arange(1, len(df) + 1)
        rewards = df['r'].values
        window = max(50, len(df) // 20)
        rolling_mean = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        
        ax_train.scatter(episodes, rewards, alpha=0.3, s=10, c='#3498db', label='Episode Reward')
        ax_train.plot(episodes, rolling_mean, color='#e74c3c', linewidth=2.5, 
                     label=f'Rolling Average (n={window})')
        ax_train.axhline(y=rewards.mean(), color='#2ecc71', linestyle='--', linewidth=1.5,
                        label=f'Mean: {rewards.mean():.1f}')
        
        ax_train.set_xlabel('Episode')
        ax_train.set_ylabel('Total Episode Reward')
        ax_train.set_title('(a) DQN Agent Training Progress', fontsize=13, fontweight='bold')
        ax_train.legend(loc='lower right')
        ax_train.grid(True, alpha=0.3)
    
    # Bottom: 3 comparison charts
    if comparison is not None:
        metrics = ['reward', 'density', 'throughput']
        titles = ['(b) Average Reward', '(c) Average Density', '(d) Average Throughput']
        
        baseline_vals = [
            comparison['baseline']['mean_reward'],
            comparison['baseline']['mean_density'],
            comparison['baseline']['mean_throughput']
        ]
        
        rl_vals = [
            comparison['rl_dqn']['mean_reward'],
            comparison['rl_dqn']['mean_density'],
            comparison['rl_dqn']['mean_throughput']
        ]
        
        improvements = [
            comparison['improvement']['reward'],
            comparison['improvement']['density'],
            comparison['improvement']['throughput']
        ]
        
        colors = ['#3498db', '#e74c3c']
        
        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            x = np.arange(2)
            values = [baseline_vals[i], rl_vals[i]]
            bars = ax.bar(x, values, color=colors, width=0.6, edgecolor='black', linewidth=1)
            
            ax.set_xticks(x)
            ax.set_xticklabels(['Baseline', 'RL (DQN)'])
            ax.set_title(f'{titles[i]}\n({improvements[i]:+.2f}%)', fontsize=11, fontweight='bold')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Figure 8: RL Agent Evaluation Results - Victoria Island Network\n(50,000 training timesteps)', 
                 fontsize=15, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'fig_8_complete.png')
    plt.savefig(output_path)
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_summary(df, comparison):
    """Print summary of results for thesis."""
    print("\n" + "="*60)
    print("SUMMARY FOR THESIS")
    print("="*60)
    
    if df is not None:
        print(f"\nTraining Statistics:")
        print(f"  - Total Episodes: {len(df)}")
        print(f"  - Mean Reward: {df['r'].mean():.2f}")
        print(f"  - Std Dev: {df['r'].std():.2f}")
        print(f"  - Max Reward: {df['r'].max():.2f}")
        print(f"  - Min Reward: {df['r'].min():.2f}")
    
    if comparison is not None:
        print(f"\nComparison Results:")
        print(f"  Baseline (Fixed-Time):")
        print(f"    - Mean Reward: {comparison['baseline']['mean_reward']:.2f}")
        print(f"    - Mean Density: {comparison['baseline']['mean_density']:.4f}")
        print(f"    - Mean Throughput: {comparison['baseline']['mean_throughput']:.2f}")
        print(f"  RL Agent (DQN):")
        print(f"    - Mean Reward: {comparison['rl_dqn']['mean_reward']:.2f}")
        print(f"    - Mean Density: {comparison['rl_dqn']['mean_density']:.4f}")
        print(f"    - Mean Throughput: {comparison['rl_dqn']['mean_throughput']:.2f}")
        print(f"  Improvement:")
        print(f"    - Reward: {comparison['improvement']['reward']:+.2f}%")
        print(f"    - Density: {comparison['improvement']['density']:+.2f}%")
        print(f"    - Throughput: {comparison['improvement']['throughput']:+.2f}%")


def main():
    """Main function to generate all figures."""
    print("\n" + "="*70)
    print("THESIS FIGURE GENERATION FROM KAGGLE RL TRAINING RESULTS")
    print("="*70)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Load data
    df, comparison, rl_results, baseline = load_data()
    
    # Generate figures
    generate_figure_8_training_curve(df)
    generate_figure_8_comparison(comparison, rl_results, baseline)
    generate_figure_combined(df, comparison)
    
    # Print summary
    print_summary(df, comparison)
    
    print("\n" + "="*70)
    print("FIGURE GENERATION COMPLETE")
    print("="*70)
    print(f"\nGenerated figures saved to: {OUTPUT_DIR}")
    print("  - fig_8_training_curve.png")
    print("  - fig_8_comparison.png")
    print("  - fig_8_complete.png (combined)")


if __name__ == "__main__":
    main()
