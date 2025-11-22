"""
Launch Thesis Stage 1 Validation on Kaggle

Quick launcher for thesis model validation results generation.

Usage:
    # Full validation (all tests)
    python launch_thesis_stage1.py --timeout 3600
    
    # Quick test (skip Riemann)
    python launch_thesis_stage1.py --quick-test --timeout 1800
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Launch Thesis Stage 1 on Kaggle')
    parser.add_argument('--timeout', type=int, default=3600, help='Kaggle timeout (default: 3600s = 1h)')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode (skip Riemann)')
    parser.add_argument('--commit-message', type=str, default=None, help='Custom commit message')
    args = parser.parse_args()
    
    print("=" * 80)
    print("THESIS STAGE 1: MODEL VALIDATION - KAGGLE LAUNCHER")
    print("=" * 80)
    print(f"Timeout: {args.timeout}s ({args.timeout//60} min)")
    print(f"Quick Test: {args.quick_test}")
    print("=" * 80)
    print()
    
    # Build executor command
    cmd = [
        'python',
        'kaggle_runner/executor.py',
        '--target', 'kaggle_runner/experiments/thesis_stage1_validation.py',
        '--timeout', str(args.timeout)
    ]
    
    # Add commit message
    if args.commit_message:
        cmd.extend(['--commit-message', args.commit_message])
    else:
        mode = "Quick Test" if args.quick_test else "Full Validation"
        commit_msg = f"Thesis Stage 1: {mode} (Section 7 results)"
        cmd.extend(['--commit-message', commit_msg])
    
    print(f"Executing: {' '.join(cmd)}")
    print("=" * 80)
    print()
    
    # Execute
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
