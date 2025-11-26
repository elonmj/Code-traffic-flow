"""
Launch Thesis Stage 2 RL Training on Kaggle

Quick launcher for thesis RL training and baseline evaluation.

Usage:
    # Full training (100k steps) - no timeout
    python launch_thesis_stage2.py --timesteps 100000
    
    # Quick test (1000 steps)
    python launch_thesis_stage2.py --quick-test
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Launch Thesis Stage 2 on Kaggle')
    parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps (default: 50k)')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode (1000 steps)')
    parser.add_argument('--commit-message', type=str, default=None, help='Custom commit message')
    parser.add_argument('--no-timeout', action='store_true', default=True, help='Disable timeout (default: True)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("THESIS STAGE 2: RL TRAINING - KAGGLE LAUNCHER")
    print("=" * 80)
    
    timesteps = 1000 if args.quick_test else args.timesteps
    
    print(f"Timesteps: {timesteps}")
    print(f"Timeout: DISABLED (run to completion)")
    print("=" * 80)
    print()
    
    # Build executor command - use --args for passing arguments to target script
    script_args = f"--timesteps {timesteps} --episodes 5"
    
    cmd = [
        'python',
        'kaggle_runner/executor.py',
        '--target', 'kaggle_runner/experiments/thesis_stage2_rl_training.py',
        '--no-timeout',
        '--args', script_args
    ]
    
    # Add commit message BEFORE --args (order matters)
    if args.commit_message:
        commit_msg = args.commit_message
    else:
        mode = "Quick Test" if args.quick_test else "Full Training"
        commit_msg = f"Thesis Stage 2: {mode} ({timesteps} steps)"
    
    # Insert commit message before --args
    cmd.insert(4, '--commit-message')
    cmd.insert(5, commit_msg)
    
    print(f"Executing: {' '.join(cmd)}")
    print("=" * 80)
    print()
    
    # Execute
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
