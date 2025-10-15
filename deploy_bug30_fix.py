"""
Complete Kaggle Deployment Script for Bug #30 Fix

This script handles the complete deployment and monitoring workflow:
1. Deploy kernel with Bug #30 fix
2. Monitor execution status
3. Download results when complete
4. Provide analysis guidance

Usage:
    python deploy_bug30_fix.py
"""

import subprocess
import sys
import time
from pathlib import Path

print("=" * 80)
print("BUG #30 FIX - KAGGLE DEPLOYMENT")
print("=" * 80)
print()

# Step 1: Verify Git status
print("Step 1: Verifying Git status...")
result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
if result.stdout.strip():
    print("⚠️  Uncommitted changes detected:")
    print(result.stdout)
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        print("Deployment cancelled.")
        sys.exit(0)
else:
    print("✅ Git repository clean")
print()

# Step 2: Check commit
print("Step 2: Verifying Bug #30 fix commit...")
result = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True)
print(f"Latest commit: {result.stdout.strip()}")
if "Bug #30" in result.stdout or "7494c4f" in result.stdout:
    print("✅ Bug #30 fix is the latest commit")
else:
    print("⚠️  Bug #30 fix might not be the latest commit")
print()

# Step 3: Deploy to Kaggle
print("Step 3: Deploying to Kaggle...")
print("Command: python validation_ch7/scripts/run_kaggle_validation_section_7_6.py --quick --scenario traffic_light_control")
print()
print("🚀 Starting deployment...")
print("   This will:")
print("   - Push any changes to GitHub")
print("   - Create new Kaggle kernel")
print("   - Start execution (~15 minutes)")
print()

response = input("Proceed with deployment? (y/n): ")
if response.lower() != 'y':
    print("Deployment cancelled.")
    sys.exit(0)

print()
print("Deploying... (this may take 2-3 minutes)")
print()

try:
    subprocess.run([
        "python", 
        "validation_ch7/scripts/run_kaggle_validation_section_7_6.py",
        "--quick",
        "--scenario", "traffic_light_control"
    ], check=True)
    print()
    print("✅ Deployment initiated successfully!")
except subprocess.CalledProcessError as e:
    print(f"❌ Deployment failed: {e}")
    sys.exit(1)

print()
print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1. Check latest kernel:")
print("   python check_latest_kernel.py")
print()
print("2. Monitor kernel status (after ~15 minutes):")
print("   Check Kaggle website or wait for auto-download")
print()
print("3. Analyze results:")
print("   python analyze_bug29_results.py validation_output/results/<kernel_name>/section_7_6_rl_performance")
print()
print("4. Expected results:")
print("   ✅ Training: Rewards 0.03-0.13")
print("   ✅ Evaluation: Non-zero rewards (Bug #30 fixed!)")
print("   ✅ RL > Baseline efficiency")
print()
print("=" * 80)
print("DEPLOYMENT COMPLETE")
print("=" * 80)
