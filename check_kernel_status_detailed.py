#!/usr/bin/env python
"""
Check Kaggle kernel execution status and retrieve logs.
"""

from kaggle.api.kaggle_api_extended import KaggleApi
import json

kernel_slug = "elonmj/arz-validation-76rlperformance-rjot"

try:
    api = KaggleApi()
    api.authenticate()
    
    print("=" * 80)
    print(f"Kernel Information: {kernel_slug}")
    print("=" * 80)
    
    # Try to get kernel information
    try:
        kernels_list = api.kernels_list(search=kernel_slug.split('/')[-1], user=kernel_slug.split('/')[0], page_size=1)
        if kernels_list:
            kernel = kernels_list[0]
            print(f"\n[SUCCESS] Kernel found!")
            print(f"  ID: {kernel.id}")
            print(f"  Title: {kernel.title}")
            print(f"  Status: {kernel.status if hasattr(kernel, 'status') else 'Unknown'}")
            print(f"  Author: {kernel.author}")
            print(f"  Source URL: {kernel.source_url if hasattr(kernel, 'source_url') else 'N/A'}")
            
            # Print all attributes
            print(f"\n[DEBUG] Kernel attributes:")
            for attr in dir(kernel):
                if not attr.startswith('_'):
                    try:
                        val = getattr(kernel, attr)
                        if not callable(val):
                            print(f"  {attr}: {val}")
                    except:
                        pass
    except Exception as e:
        print(f"[ERROR] Could not list kernels: {e}")
    
    # Try to get kernel output logs
    print(f"\n[INFO] Attempting to retrieve kernel logs...")
    try:
        # Try to download kernel logs if available
        api.kernels_output(kernel_slug, ".")
        print("[SUCCESS] Kernel output files downloaded to current directory")
        
        import os
        files = os.listdir(".")
        filtered = [f for f in files if 'arz-validation' in f]
        if filtered:
            print(f"[INFO] Found files: {filtered}")
    except Exception as e:
        print(f"[WARNING] Could not get kernel output: {e}")
        
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
