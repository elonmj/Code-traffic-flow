#!/usr/bin/env python3
"""
Verify the downloaded NPZ file from Kaggle
"""

import numpy as np
from pathlib import Path

npz_path = Path("validation_output/results/elonmj_arz-validation-kyoz/validation_results/npz/test_minimal_riemann_20251002_120147.npz")

print("=" * 80)
print("NPZ FILE VERIFICATION")
print("=" * 80)

if not npz_path.exists():
    print(f"✗ File not found: {npz_path}")
    exit(1)

print(f"\n✓ File found: {npz_path.name}")
print(f"  Size: {npz_path.stat().st_size / 1024:.1f} KB")

# Load and verify
try:
    data = np.load(str(npz_path), allow_pickle=True)
    
    print("\n✓ NPZ loaded successfully")
    print(f"\nContents:")
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            print(f"  - {key}: {data[key].shape} {data[key].dtype}")
        else:
            print(f"  - {key}: {type(data[key])}")
    
    # Verify critical arrays
    times = data['times']
    states = data['states']
    
    print(f"\n✓ Validation:")
    print(f"  - Times: {len(times)} timesteps from {times[0]:.3f}s to {times[-1]:.3f}s")
    print(f"  - States: {states.shape} (timesteps, variables, cells)")
    print(f"  - Grid info: {data['grid_info']}")
    
    print("\n" + "=" * 80)
    print("SUCCESS - NPZ FILE IS VALID AND COMPLETE")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ Error loading NPZ: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
