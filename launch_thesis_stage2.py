"""
Launch Thesis Stage 2 RL Training on Kaggle

Quick launcher for thesis RL training and baseline evaluation.

Usage:
    # Full training (100k steps)
    python launch_thesis_stage2.py --timesteps 100000 --timeout 7200
    
    # Quick test (1000 steps)
    python launch_thesis_stage2.py --quick-test
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Launch Thesis Stage 2 on Kaggle')
    parser.add_argument('--timesteps', type=int, default=100000, help='Training timesteps (default: 100k)')
    parser.add_argument('--timeout', type=int, default=7200, help='Kaggle timeout (default: 7200s = 2h)')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode (1000 steps)')
    parser.add_argument('--commit-message', type=str, default=None, help='Custom commit message')
    args = parser.parse_args()
    
    print("=" * 80)
    print("THESIS STAGE 2: RL TRAINING - KAGGLE LAUNCHER")
    print("=" * 80)
    
    timesteps = 1000 if args.quick_test else args.timesteps
    timeout = 1800 if args.quick_test else args.timeout
    
    print(f"Timesteps: {timesteps}")
    print(f"Timeout: {timeout}s ({timeout//60} min)")
    print("=" * 80)
    print()
    
    # Build executor command
    cmd = [
        'python',
        'kaggle_runner/executor.py',
        '--target', 'kaggle_runner/experiments/thesis_stage2_rl_training.py',
        '--timeout', str(timeout),
        '--args', f"--timesteps {timesteps} --episodes 5"
    ]
    
    # Add commit message
    if args.commit_message:
        cmd.extend(['--commit-message', args.commit_message])
    else:
        mode = "Quick Test" if args.quick_test else "Full Training"
        commit_msg = f"Thesis Stage 2: {mode} (PPO vs Baseline)"
        cmd.extend(['--commit-message', commit_msg])
    
    print(f"Executing: {' '.join(cmd)}")
    print("=" * 80)
    print()
    
    # Execute
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
