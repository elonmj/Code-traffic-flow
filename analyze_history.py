#!/usr/bin/env python3
import numpy as np
import os

print("="*80)
print("ANALYZING CPU vs GPU HISTORY")
print("="*80)

# Check CPU
cpu_hist_path = 'arz_model_cpu/arz_model/results/output_network_test_seg_0.npz'
if os.path.exists(cpu_hist_path):
    print(f"\n[CPU] {cpu_hist_path}")
    cpu_hist = np.load(cpu_hist_path, allow_pickle=True)
    print(f"  Files: {list(cpu_hist.files)}")
    
    if 'states' in cpu_hist.files:
        states = cpu_hist['states']
        print(f"  states.shape: {states.shape}")
        print(f"  states.dtype: {states.dtype}")
        print(f"  states min/max: {states.min():.6f} / {states.max():.6f}")
        print(f"  states[0]: {states[0]}")
else:
    print(f"  NOT FOUND: {cpu_hist_path}")

# Check GPU
gpu_hist_path = 'kaggle/results/elonmj_generic-test-runner-kernel/simulation_results/history_seg-1.npz'
if os.path.exists(gpu_hist_path):
    print(f"\n[GPU] {gpu_hist_path}")
    gpu_hist = np.load(gpu_hist_path, allow_pickle=True)
    print(f"  Files: {list(gpu_hist.files)}")
    
    if 'states' in gpu_hist.files:
        states = gpu_hist['states']
        print(f"  states.shape: {states.shape}")
        print(f"  states.dtype: {states.dtype}")
        if states.size > 0:
            print(f"  states min/max: {states.min():.6f} / {states.max():.6f}")
            print(f"  states[0]: {states[0]}")
        else:
            print(f"  states is EMPTY")
else:
    print(f"  NOT FOUND: {gpu_hist_path}")

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

cpu_hist = np.load('arz_model_cpu/arz_model/results/output_network_test_seg_0.npz', allow_pickle=True)
gpu_hist = np.load(gpu_hist_path, allow_pickle=True)

cpu_states = cpu_hist.get('states')
gpu_states = gpu_hist.get('states')

if cpu_states is not None and gpu_states is not None:
    print(f"\nCPU: shape {cpu_states.shape}, max value {cpu_states.max():.6f}")
    print(f"GPU: shape {gpu_states.shape}, max value {gpu_states.max():.6f}")
    
    if gpu_states.max() == 0:
        print("\n>>> PROBLEM: GPU has NO DATA (all zeros)")
        print("    This means initial conditions or boundary conditions are not being applied!")
