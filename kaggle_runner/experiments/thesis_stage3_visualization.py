"""
Thesis Stage 3: Visualization & Results Generation

This script generates the final figures and tables for the thesis (Sections 7 & 8).
It consumes the data produced by Stage 1 and Stage 2.

Usage:
    python kaggle_runner/experiments/thesis_stage3_visualization.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 80)
print("THESIS STAGE 3: VISUALIZATION")
print("=" * 80)

# Setup paths
RESULTS_DIR = Path("/kaggle/working" if os.path.exists("/kaggle/working") else "results")
STAGE1_DIR = RESULTS_DIR / "thesis_stage1"
STAGE2_DIR = RESULTS_DIR / "thesis_stage2"
OUTPUT_DIR = RESULTS_DIR / "thesis_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['lines.linewidth'] = 2.0


def plot_riemann_profiles():
    """Generate Figures 7.1-7.5: Riemann Problem Profiles"""
    print("\nGenerating Riemann profiles...")
    
    files = list(STAGE1_DIR.glob("riemann_*.npz"))
    if not files:
        print("⚠️ No Riemann data found in", STAGE1_DIR)
        return

    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            U = data['U']
            x = data['x']
            config = data['config'].item()
            name = config['name']
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            
            # Density
            ax1.plot(x, U[0, :], label='Motos', color='blue')
            ax1.plot(x, U[2, :], label='Cars', color='red', linestyle='--')
            ax1.set_ylabel('Density (veh/m)')
            ax1.set_title(f"Riemann Problem: {config['description']} (t={data['t']}s)")
            ax1.legend()
            
            # Velocity (approx w/rho for visualization if p not computed)
            # Or just plot w (momentum) if v not available easily
            # Here we plot w/rho roughly
            rho_m = np.maximum(U[0, :], 1e-6)
            rho_c = np.maximum(U[2, :], 1e-6)
            v_m = U[1, :] / rho_m # Rough approx
            v_c = U[3, :] / rho_c
            
            ax2.plot(x, v_m, label='Motos', color='blue')
            ax2.plot(x, v_c, label='Cars', color='red', linestyle='--')
            ax2.set_ylabel('Velocity (m/s)')
            ax2.set_xlabel('Position (m)')
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"fig_7_{name}.png", dpi=300)
            plt.close()
            print(f"  Saved fig_7_{name}.png")
            
        except Exception as e:
            print(f"  Error plotting {f.name}: {e}")


def plot_rl_training_curve():
    """Generate Figure 8.x: RL Training Curve"""
    print("\nGenerating RL training curve...")
    
    # Look for monitor.csv or progress.csv
    log_dir = STAGE2_DIR / "logs"
    monitor_file = STAGE2_DIR / "monitor.csv" # If Monitor wrapper used
    
    data = None
    if monitor_file.exists():
        import pandas as pd
        try:
            # Skip first line (header info)
            df = pd.read_csv(monitor_file, skiprows=1)
            
            # Calculate rolling mean
            df['reward_smooth'] = df['r'].rolling(window=50).mean()
            
            plt.figure(figsize=(10, 6))
            plt.plot(df['l'].cumsum(), df['r'], alpha=0.3, color='gray', label='Episode Reward')
            plt.plot(df['l'].cumsum(), df['reward_smooth'], color='blue', label='Moving Average (50)')
            
            plt.xlabel('Timesteps')
            plt.ylabel('Total Reward')
            plt.title('PPO Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(OUTPUT_DIR / "fig_8_training_curve.png", dpi=300)
            plt.close()
            print("  Saved fig_8_training_curve.png")
            
        except Exception as e:
            print(f"  Error plotting training curve: {e}")
    else:
        print("⚠️ No training log found.")


def plot_comparison_chart():
    """Generate Figure 8.y: Baseline vs RL Comparison"""
    print("\nGenerating comparison chart...")
    
    summary_file = STAGE2_DIR / "comparison_summary.json"
    if not summary_file.exists():
        print("⚠️ No comparison summary found.")
        return
        
    try:
        with open(summary_file, 'r') as f:
            data = json.load(f)
            
        baseline = data['baseline']
        rl = data['rl_ppo']
        
        metrics = ['mean_reward', 'mean_density', 'mean_throughput']
        labels = ['Reward (Higher is better)', 'Density (Lower is better)', 'Throughput (Higher is better)']
        
        b_vals = [baseline[m] for m in metrics]
        r_vals = [rl[m] for m in metrics]
        
        # Normalize for visualization? Or separate charts.
        # Let's do separate bar charts for clarity
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, ax in enumerate(axes):
            metric = metrics[i]
            vals = [b_vals[i], r_vals[i]]
            colors = ['gray', 'green']
            
            bars = ax.bar(['Fixed Time', 'RL (PPO)'], vals, color=colors)
            ax.set_title(labels[i])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
                        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig_8_comparison.png", dpi=300)
        plt.close()
        print("  Saved fig_8_comparison.png")
        
    except Exception as e:
        print(f"  Error plotting comparison: {e}")


def main():
    plot_riemann_profiles()
    plot_rl_training_curve()
    plot_comparison_chart()
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
