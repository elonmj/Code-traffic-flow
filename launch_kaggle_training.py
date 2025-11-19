"""
Kaggle RL Training Launcher

Quick script to launch RL training on Kaggle GPU with the complete Victoria Island network.

Usage:
    # Quick test (300 steps)
    python launch_kaggle_training.py --timesteps 300
    
    # Full training (10k steps)
    python launch_kaggle_training.py --timesteps 10000 --timeout 7200
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Launch RL training on Kaggle')
    parser.add_argument('--timesteps', type=int, default=100, help='Training timesteps (default: 100)')
    parser.add_argument('--timeout', type=int, default=1800, help='Kaggle timeout in seconds (default: 1800 = 30min)')
    parser.add_argument('--scenario', type=str, default='victoria_island', help='Training scenario')
    parser.add_argument('--commit-message', type=str, default=None, help='Custom commit message')
    args = parser.parse_args()
    
    print("=" * 80)
    print("KAGGLE RL TRAINING LAUNCHER")
    print("=" * 80)
    print(f"Timesteps: {args.timesteps}")
    print(f"Timeout: {args.timeout}s ({args.timeout//60} min)")
    print(f"Scenario: {args.scenario}")
    print(f"Kernel: rl-training-runner (Config: kaggle_runner/config/rl_training_victoria_island.yml)")
    print("=" * 80)
    print()
    
    # Build executor command
    cmd = [
        'python',
        'kaggle_runner/executor.py',
        '--target', 'kaggle_runner/experiments/rl_training_victoria_island.py',
        '--timeout', str(args.timeout)
    ]
    
    if args.commit_message:
        cmd.extend(['--commit-message', args.commit_message])
    else:
        commit_msg = f"RL training {args.timesteps} steps on Victoria Island network"
        cmd.extend(['--commit-message', commit_msg])
    
    print(f"Executing: {' '.join(cmd)}")
    print()
    
    # Run executor
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Kaggle execution failed with code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())
