#!/usr/bin/env python3
"""
QUICK START - Kaggle RL Training

This script launches a 300-step RL training test on Kaggle GPU to verify everything works.

Run this first before longer trainings!
"""

import subprocess
import sys

print("=" * 80)
print("🚀 QUICK START - Kaggle RL Training Test (300 steps)")
print("=" * 80)
print()
print("This will:")
print("  1. Commit and push your local changes")
print("  2. Create/update Kaggle kernel")
print("  3. Run 300-step training on Kaggle GPU")
print("  4. Download results automatically")
print()
print("Expected duration: ~10 minutes")
print("=" * 80)
print()

response = input("Ready to launch? [y/N]: ")
if response.lower() != 'y':
    print("Cancelled.")
    sys.exit(0)

cmd = [
    'python',
    'launch_kaggle_training.py',
    '--timesteps', '300',
    '--timeout', '600'  # 10 minutes
]

print(f"\nExecuting: {' '.join(cmd)}\n")

try:
    result = subprocess.run(cmd, check=True)
    print("\n" + "=" * 80)
    print("✅ SUCCESS - Check kaggle_runner/results/ for outputs")
    print("=" * 80)
    sys.exit(result.returncode)
except subprocess.CalledProcessError as e:
    print(f"\n❌ FAILED with code {e.returncode}")
    sys.exit(e.returncode)
except KeyboardInterrupt:
    print("\n⚠️  Interrupted")
    sys.exit(1)
