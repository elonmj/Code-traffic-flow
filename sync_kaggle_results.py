#!/usr/bin/env python
"""
Simplified Kaggle kernel monitoring and results download.
Waits for kernel completion and syncs results locally.
"""

import time
import sys
import subprocess
from pathlib import Path

kernel_slug = "elonmj/arz-validation-76rlperformance-rjot"
output_dir = Path("validation_output/results")

print("=" * 80)
print("KAGGLE KERNEL MONITORING & RESULTS SYNC")
print("=" * 80)
print(f"Kernel: {kernel_slug}")
print(f"Output directory: {output_dir.absolute()}")
print()

# Try to use kaggle API to download outputs
print("[STEP 1] Attempting to download kernel output files...")

try:
    # Try using kaggle.cli directly if available
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    api = KaggleApi()
    api.authenticate()
    
    print("[INFO] Kaggle API authenticated successfully")
    
    # Download kernel output
    print(f"[INFO] Downloading output for {kernel_slug}...")
    api.kernels_output(kernel_slug, str(output_dir.absolute()))
    print(f"[SUCCESS] Kernel output downloaded!")
    
except ImportError:
    print("[WARNING] kaggle module not directly available, trying subprocess...")
    
    # Try using kaggle CLI via subprocess
    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_slug, "-p", str(output_dir.absolute())],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"[SUCCESS] Output downloaded!")
        print(result.stdout)
    else:
        print(f"[ERROR] Download failed: {result.stderr}")

except Exception as e:
    print(f"[ERROR] Exception: {e}")
    import traceback
    traceback.print_exc()

# Check what we have now
print("\n[STEP 2] Checking local results...")
if output_dir.exists():
    dirs = sorted(output_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs[:2]:
        if d.is_dir():
            mtime = d.stat().st_mtime
            import datetime
            dt = datetime.datetime.fromtimestamp(mtime)
            files = list(d.glob("*"))
            print(f"  - {d.name}")
            print(f"    Modified: {dt}")
            print(f"    Files ({len(files)}):")
            for f in files[:5]:
                print(f"      - {f.name} ({f.stat().st_size} bytes)")
            if len(files) > 5:
                print(f"      ... and {len(files)-5} more")

print("\n[DONE]")
