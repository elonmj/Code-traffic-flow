#!/usr/bin/env python
"""Check joselonm GPU kernels"""
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Get MY kernels (joselonm username)
kernels = api.kernels_list(page_size=50, sort_by='dateRun')
my_kernels = [k for k in kernels if k.ref.startswith('joselonm/')]

print(f"📊 My kernels (joselonm): {len(my_kernels)}\n")

gpu_kernels = []
for k in my_kernels:
    enable_gpu = k.enable_gpu
    print(f"• {k.ref}")
    print(f"  GPU: {enable_gpu}")
    print(f"  Last run: {k.last_run_time}")
    
    if enable_gpu:
        gpu_kernels.append(k.ref)
    print()

print(f"\n🔴 GPU Kernels: {len(gpu_kernels)}")
for ref in gpu_kernels:
    print(f"   - {ref}")

if len(gpu_kernels) >= 2:
    print("\n⚠️  GPU QUOTA AT LIMIT (2/2 sessions)")
elif len(gpu_kernels) == 1:
    print(f"\n✅ GPU AVAILABLE (1/2 sessions used)")
else:
    print(f"\n✅ GPU AVAILABLE (0/2 sessions used)")
