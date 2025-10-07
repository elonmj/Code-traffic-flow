#!/usr/bin/env python3
"""
Quick Test Script for Section 7.6 - RL Performance Validation

15-minute validation test on Kaggle GPU:
- Tests RL-ARZ direct coupling integration
- Verifies GPU acceleration works
- Validates environment setup
- Only 10 training timesteps for speed

Usage:
    python quick_test_section_7_6.py

This will:
1. Auto-commit and push to GitHub
2. Launch Kaggle kernel with GPU
3. Run minimal RL training (10 steps)
4. Validate direct coupling works
5. Download results
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("QUICK TEST - Section 7.6 RL Performance (15 minutes)")
print("=" * 80)
print()
print("This quick test validates:")
print("  ✓ TrafficSignalEnvDirect integration")
print("  ✓ GPU device detection and acceleration")
print("  ✓ RL training pipeline (minimal 10 steps)")
print("  ✓ Performance metrics collection")
print()
print("What's tested:")
print("  - 1 scenario only (traffic_light_control)")
print("  - 10 training timesteps")
print("  - 10 minutes simulated time")
print("  - Baseline vs RL comparison")
print()
print("Expected runtime: ~15 minutes on Kaggle GPU")
print("=" * 80)
print()

# Import and run with quick flag
import subprocess
result = subprocess.run([
    sys.executable,
    str(project_root / "validation_ch7" / "scripts" / "run_kaggle_validation_section_7_6.py"),
    "--quick"
], cwd=str(project_root))

sys.exit(result.returncode)
