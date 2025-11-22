"""
Launch Thesis Stage 3 Visualization on Kaggle

Quick launcher for thesis visualization generation.

Usage:
    python launch_thesis_stage3.py
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Launch Thesis Stage 3 on Kaggle')
    parser.add_argument('--timeout', type=int, default=1800, help='Kaggle timeout (default: 1800s = 30min)')
    parser.add_argument('--commit-message', type=str, default=None, help='Custom commit message')
    args = parser.parse_args()
    
    print("=" * 80)
    print("THESIS STAGE 3: VISUALIZATION - KAGGLE LAUNCHER")
    print("=" * 80)
    print()
    
    # Build executor command
    cmd = [
        'python',
        'kaggle_runner/executor.py',
        '--target', 'kaggle_runner/experiments/thesis_stage3_visualization.py',
        '--timeout', str(args.timeout)
    ]
    
    # Add commit message
    if args.commit_message:
        cmd.extend(['--commit-message', args.commit_message])
    else:
        cmd.extend(['--commit-message', "Thesis Stage 3: Visualization"])
    
    print(f"Executing: {' '.join(cmd)}")
    print("=" * 80)
    print()
    
    # Execute
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
