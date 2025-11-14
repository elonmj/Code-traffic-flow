"""
Quick script to verify NPZ files contain non-zero data
"""
import numpy as np
from pathlib import Path

results_dir = Path("kaggle/results/elonmj_generic-test-runner-kernel/simulation_results")

print("=" * 80)
print("VERIFYING NPZ OUTPUT DATA")
print("=" * 80)

# Check history files
for segment in ["seg-1", "seg-2"]:
    print(f"\n{'=' * 80}")
    print(f"SEGMENT: {segment}")
    print(f"{'=' * 80}")
    
    history_file = results_dir / f"history_{segment}.npz"
    if not history_file.exists():
        print(f"❌ File not found: {history_file}")
        continue
    
    data = np.load(history_file)
    print(f"\n📦 Loaded: {history_file.name}")
    print(f"   Arrays: {list(data.keys())}")
    
    # Check each array
    for key in data.keys():
        arr = data[key]
        print(f"\n   🔍 {key}:")
        print(f"      Shape: {arr.shape}")
        print(f"      dtype: {arr.dtype}")
        print(f"      Min:   {arr.min():.6f}")
        print(f"      Max:   {arr.max():.6f}")
        print(f"      Mean:  {arr.mean():.6f}")
        print(f"      Std:   {arr.std():.6f}")
        
        # Check for all zeros
        if np.all(arr == 0):
            print(f"      ⚠️  WARNING: ALL ZEROS!")
        else:
            print(f"      ✅ Contains non-zero values")
            
        # Show first few timesteps/values
        if arr.ndim == 1:
            print(f"      First 5 values: {arr[:5]}")
        elif arr.ndim == 2:
            print(f"      First timestep (t=0): min={arr[0].min():.6f}, max={arr[0].max():.6f}, mean={arr[0].mean():.6f}")
            if arr.shape[0] > 1:
                print(f"      Last timestep (t={arr.shape[0]-1}): min={arr[-1].min():.6f}, max={arr[-1].max():.6f}, mean={arr[-1].mean():.6f}")

# Check simulation metadata
print(f"\n{'=' * 80}")
print("SIMULATION METADATA")
print(f"{'=' * 80}")
metadata_file = results_dir / "simulation_metadata.npz"
if metadata_file.exists():
    meta = np.load(metadata_file, allow_pickle=True)
    for key in meta.keys():
        val = meta[key]
        print(f"   {key}: {val}")

print(f"\n{'=' * 80}")
print("VERIFICATION COMPLETE")
print(f"{'=' * 80}")
