#!/usr/bin/env python
"""Find and potentially stop running kernels"""
from kaggle.api.kaggle_api_extended import KaggleApi
import requests

api = KaggleApi()
api.authenticate()

# Use Kaggle API to check kernel status
print("🔍 Checking for running kernels...")

# Get auth headers
headers = {
    'Authorization': f'Bearer {api.read_config_file(".kaggle/kaggle.json")["api_token"]}'
}

# Actually, try using the API directly
try:
    # List kernels and check their status
    kernels = api.kernels_list(page_size=100)
    
    # Get kernel versions that might show running status
    for k in kernels:
        print(f"\n📌 {k.ref}")
        
        # Try to get kernel details
        try:
            kernel_detail = api.kernels_view(k.ref)
            print(f"  Status attributes: {dir(kernel_detail)}")
        except:
            pass
            
        # Show what we know
        print(f"  Enable GPU: {k.enable_gpu}")
        print(f"  Last run: {k.last_run_time}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
