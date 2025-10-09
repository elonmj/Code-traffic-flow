#!/usr/bin/env python3
"""
Download Kaggle kernel output without displaying logs.
"""

import sys
import os
from pathlib import Path
from kaggle import KaggleApi

# Force UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Initialize API
api = KaggleApi()
api.authenticate()

kernel_slug = "elonmj/arz-validation-76rlperformance-ecuf"
output_dir = Path("./kaggle_kernel_output")
output_dir.mkdir(exist_ok=True, parents=True)

print(f"Downloading kernel output from: {kernel_slug}")
print(f"Output directory: {output_dir.absolute()}")

try:
    # Download output files
    api.kernels_output_cli(
        kernel=kernel_slug,
        path=str(output_dir),
        force=True,
        quiet=False
    )
    
    # List downloaded files
    files = list(output_dir.rglob("*"))
    print(f"\n[SUCCESS] Downloaded {len(files)} files:")
    for f in sorted(files):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.relative_to(output_dir)} ({size_kb:.1f} KB)")
    
except Exception as e:
    print(f"[ERROR] Failed to download: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
