#!/usr/bin/env python3
"""Download results from latest Kaggle kernel"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from validation_ch7.scripts.validation_kaggle_manager import ValidationKaggleManager

# Initialize manager
manager = ValidationKaggleManager()

# Download results for latest kernel
kernel_slug = "elonmj/arz-validation-76rlperformance-nboq"
print(f"[DOWNLOAD] Downloading results for: {kernel_slug}")

success = manager.download_results(kernel_slug, output_dir=f"validation_output/results/{kernel_slug.replace('/', '_')}")

if success:
    print(f"\n[SUCCESS] Results downloaded!")
    print(f"[PATH] Check: validation_output/results/{kernel_slug.replace('/', '_')}")
else:
    print("\n[ERROR] Download failed")
    sys.exit(1)
