#!/usr/bin/env python3
"""
Launch Thesis Stage 2 v2 on Kaggle
==================================

Improved reward weights to create a more challenging RL problem:
- alpha: 1.0 -> 5.0 (congestion matters 5x more)
- kappa: 0.1 -> 0.3 (switching penalty 3x higher)  
- mu: 0.5 -> 0.1 (throughput less dominant)
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Launch Thesis Stage 2 v2 on Kaggle')
    parser.add_argument('--timesteps', type=int, default=10000, help='Training timesteps')
    parser.add_argument('--episodes', type=int, default=5, help='Evaluation episodes')
    parser.add_argument('--commit-message', type=str, default=None, help='Git commit message')
    args = parser.parse_args()
    
    print("=" * 70)
    print("THESIS STAGE 2 v2: IMPROVED REWARD WEIGHTS")
    print("=" * 70)
    print(f"Timesteps: {args.timesteps}")
    print(f"Eval episodes: {args.episodes}")
    print("\nReward weights:")
    print("  - alpha: 5.0 (was 1.0) - Congestion penalty x5")
    print("  - kappa: 0.3 (was 0.1) - Switch penalty x3")
    print("  - mu: 0.1 (was 0.5) - Throughput /5")
    print("=" * 70)
    
    # Build command
    cmd = [
        sys.executable, "kaggle_runner/executor.py",
        "--target", "kaggle_runner/experiments/thesis_stage2_v2_improved.py",
        "--timeout", "3600",
        "--args", f"--timesteps {args.timesteps} --episodes {args.episodes}"
    ]
    
    if args.commit_message:
        cmd.extend(["--commit-message", args.commit_message])
    
    print(f"\nExecuting: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
