#!/usr/bin/env python
"""Quick status check for Kaggle kernel execution."""

import json
import subprocess
import sys
from pathlib import Path

kernel_slug = "elonmj/arz-validation-76rlperformance-rjot"

# Try to get kernel status using kaggle CLI
try:
    print(f"[INFO] Checking status of kernel: {kernel_slug}")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "status", kernel_slug],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode == 0:
        print("[SUCCESS] Kernel status output:")
        print(result.stdout)
    else:
        print(f"[ERROR] Command failed: {result.stderr}")
        
except Exception as e:
    print(f"[ERROR] Failed to check status: {e}")

# Check if any recent results exist locally
results_dir = Path("validation_output/results")
if results_dir.exists():
    dirs = sorted(results_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"\n[INFO] Most recent result directories:")
    for d in dirs[:3]:
        mtime = d.stat().st_mtime
        import datetime
        dt = datetime.datetime.fromtimestamp(mtime)
        files = list(d.glob("*"))
        print(f"  - {d.name}: {dt} ({len(files)} files)")
