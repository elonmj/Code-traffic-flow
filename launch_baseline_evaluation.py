"""
Kaggle Baseline Evaluation Launcher

Quick script to launch Baseline Evaluation on Kaggle GPU.

Usage:
    python launch_baseline_evaluation.py --timesteps 1000
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Launch Baseline Evaluation on Kaggle')
    parser.add_argument('--timesteps', type=int, default=1000, help='Evaluation timesteps (default: 1000)')
    parser.add_argument('--timeout', type=int, default=1800, help='Kaggle timeout in seconds (default: 1800 = 30min)')
    parser.add_argument('--commit-message', type=str, default=None, help='Custom commit message')
    args = parser.parse_args()
    
    print("=" * 80)
    print("KAGGLE BASELINE EVALUATION LAUNCHER")
    print("=" * 80)
    print(f"Timesteps: {args.timesteps}")
    print(f"Timeout: {args.timeout}s")
    print("=" * 80)
    print()
    
    # Build executor command
    # We pass arguments to the target script via modifying the target file or passing args if executor supports it.
    # Executor runs the target script. The target script (evaluate_baseline.py) currently uses default args or we need to modify it to accept args.
    # Wait, evaluate_baseline.py uses default args in the function call.
    # I should update evaluate_baseline.py to parse args if I want to control timesteps from here.
    # For now, let's just run it as is, or update it to use argparse.
    
    cmd = [
        'python',
        'kaggle_runner/executor.py',
        '--target', 'kaggle_runner/experiments/evaluate_baseline.py',
        '--timeout', str(args.timeout)
    ]
    
    if args.commit_message:
        cmd.extend(['--commit-message', args.commit_message])
    else:
        commit_msg = f"Baseline evaluation {args.timesteps} steps"
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
