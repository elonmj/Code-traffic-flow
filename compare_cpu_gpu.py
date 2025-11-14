#!/usr/bin/env python3
"""Compare CPU and GPU NPZ outputs to identify the root cause of zeros."""
import numpy as np
import os

print("=" * 80)
print("CPU vs GPU NPZ Comparison")
print("=" * 80)

# Load CPU results
cpu_seg0_path = "arz_model_cpu/arz_model/results/output_network_test_seg_0.npz"
gpu_seg1_path = "kaggle/results/elonmj_generic-test-runner-kernel/simulation_results/final_state_seg-1.npz"

print("\n[1] CPU Results (seg_0):")
print("-" * 80)
if os.path.exists(cpu_seg0_path):
    cpu_seg0 = np.load(cpu_seg0_path, allow_pickle=True)
    for key in sorted(cpu_seg0.files):
        arr = cpu_seg0[key]
        print(f"\n  Key: {key}")
        if isinstance(arr, np.ndarray):
            if arr.dtype == object:
                print(f"    Type: object array")
                print(f"    Content: {arr.item() if arr.size == 1 else 'large object'}")
            else:
                print(f"    Shape: {arr.shape}, Dtype: {arr.dtype}")
                if arr.size > 0:
                    print(f"    Stats: min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")
                    if arr.size <= 20:
                        print(f"    Values: {arr}")
                    else:
                        print(f"    First 5: {arr.flat[:5]}")
else:
    print(f"  File not found: {cpu_seg0_path}")

print("\n[2] GPU Results (seg-1):")
print("-" * 80)
if os.path.exists(gpu_seg1_path):
    gpu_seg1 = np.load(gpu_seg1_path, allow_pickle=True)
    for key in sorted(gpu_seg1.files):
        arr = gpu_seg1[key]
        print(f"\n  Key: {key}")
        if isinstance(arr, np.ndarray):
            if arr.dtype == object:
                print(f"    Type: object array")
            else:
                print(f"    Shape: {arr.shape}, Dtype: {arr.dtype}")
                if arr.size > 0:
                    print(f"    Stats: min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")
                    if arr.size <= 20:
                        print(f"    Values: {arr}")
                    else:
                        print(f"    First 5: {arr.flat[:5]}")
else:
    print(f"  File not found: {gpu_seg1_path}")

print("\n" + "=" * 80)
print("KEY OBSERVATION:")
print("=" * 80)
if os.path.exists(cpu_seg0_path) and os.path.exists(gpu_seg1_path):
    cpu_seg0 = np.load(cpu_seg0_path, allow_pickle=True)
    gpu_seg1 = np.load(gpu_seg1_path, allow_pickle=True)
    
    cpu_states = cpu_seg0.get("states")
    gpu_states = gpu_seg1.get("states")
    
    if cpu_states is not None and gpu_states is not None:
        print(f"\nCPU states shape: {cpu_states.shape}, max value: {cpu_states.max():.6f}")
        print(f"GPU states shape: {gpu_states.shape}, max value: {gpu_states.max():.6f}")
        
        if cpu_states.max() > 0 and gpu_states.max() == 0:
            print("\n⚠️  GPU has ALL ZEROS while CPU has real data!")
            print("   Root cause is likely in initialization or boundary condition application.")
