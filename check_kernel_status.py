#!/usr/bin/env python3
"""
Quick script to check Kaggle kernel status and retrieve logs.
"""

import sys
from pathlib import Path
from kaggle import KaggleApi

# Initialize API
api = KaggleApi()
api.authenticate()

# Kernel slug from recent upload
kernel_slug = "elonmj/arz-validation-76rlperformance-ecuf"

print(f"=" * 80)
print(f"CHECKING KERNEL STATUS: {kernel_slug}")
print(f"=" * 80)

try:
    # Get kernel status
    print("\n[1/3] Fetching kernel status...")
    status = api.kernels_status(kernel_slug)
    
    print(f"\n[STATUS] Kernel Status:")
    print(f"  - Status: {status.status if hasattr(status, 'status') else 'UNKNOWN'}")
    print(f"  - Failure: {status.failureMessage if hasattr(status, 'failureMessage') else 'None'}")
    print(f"  - All attributes: {dir(status)}")
    
    # Get kernel metadata
    print("\n[2/3] Fetching kernel metadata...")
    metadata = api.kernels_pull(kernel_slug, path="./kaggle_kernel_check", metadata=True)
    print(f"  [OK] Metadata downloaded to ./kaggle_kernel_check")
    
    # Get kernel output if available
    print("\n[3/3] Attempting to download kernel output...")
    try:
        api.kernels_output(kernel_slug, path="./kaggle_kernel_output")
        print(f"  [OK] Output downloaded to ./kaggle_kernel_output")
        
        # List downloaded files
        output_dir = Path("./kaggle_kernel_output")
        if output_dir.exists():
            files = list(output_dir.rglob("*"))
            print(f"\n[FILES] Downloaded {len(files)} files:")
            for f in files[:20]:  # Show first 20
                print(f"    - {f.name} ({f.stat().st_size} bytes)")
    except Exception as e:
        print(f"  [WARNING] Could not download output: {e}")
    
    print(f"\n" + "=" * 80)
    print(f"MANUAL CHECK: https://www.kaggle.com/code/{kernel_slug}")
    print(f"=" * 80)
    
except Exception as e:
    print(f"\n[ERROR] Failed to check kernel status: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
