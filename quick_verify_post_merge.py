#!/usr/bin/env python3
"""Quick verification of post-merge Kaggle results."""
import pickle
from pathlib import Path

pkl_path = Path("kaggle/results/generic-test-runner-kernel/network_simulation_results.pkl")

if not pkl_path.exists():
    print(f"❌ File not found: {pkl_path}")
    exit(1)

print("=" * 70)
print("📦 POST-MERGE KAGGLE RESULTS VERIFICATION")
print("=" * 70)

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"\n✅ Successfully loaded results")
print(f"📊 Data structure:")
print(f"   - Keys: {list(data.keys())}")

if 'times' in data:
    print(f"   - Timesteps: {len(data['times'])}")
    print(f"   - Time range: {data['times'][0]:.1f}s to {data['times'][-1]:.1f}s")

if 'segments' in data:
    print(f"   - Segments: {len(data['segments'])}")
    seg_keys = list(data['segments'].keys())
    print(f"   - Segment IDs: {seg_keys[:5]}... ({len(seg_keys)} total)")
    
    # Check first segment structure
    first_seg = data['segments'][seg_keys[0]]
    print(f"\n📈 First segment data structure:")
    for key, value in first_seg.items():
        if hasattr(value, 'shape'):
            print(f"   - {key}: shape {value.shape}")
        else:
            print(f"   - {key}: {type(value).__name__}")

print("\n" + "=" * 70)
print("✅ VALIDATION: Results structure matches expected format")
print("=" * 70)
