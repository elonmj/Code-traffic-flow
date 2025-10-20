#!/usr/bin/env python
"""Check GPU quota and list running kernels"""
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# List kernels
kernels = api.kernels_list(page_size=20, sort_by='dateRun')
print("📊 Recent kernels by joselonm:\n")

gpu_kernels = []
for i, k in enumerate(kernels[:20]):
    kernel_type = getattr(k, 'kernel_type', 'unknown')
    ref = k.ref
    print(f"{i+1}. {ref}")
    print(f"   Type: {kernel_type}")
    
    if kernel_type == 'gpu':
        gpu_kernels.append(ref)
    print()

print(f"\n🔴 GPU Kernels: {len(gpu_kernels)}")
for ref in gpu_kernels:
    print(f"   - {ref}")

if len(gpu_kernels) >= 2:
    print("\n⚠️  GPU QUOTA EXHAUSTED (max 2 GPU sessions)")
    print("   Need to stop one kernel to free quota")
