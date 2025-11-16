"""Detailed inspection of simulation results"""
import pickle
import numpy as np

# Load results
with open('network_simulation_results.pkl', 'rb') as f:
    results = pickle.load(f)

hist = results['history']
print("Segment data structure:")
for seg_id, seg_data in hist['segments'].items():
    print(f"\n{seg_id}:")
    for key, value in seg_data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            print(f"    range: [{np.min(value):.6f}, {np.max(value):.6f}]")
        else:
            print(f"  {key}: {type(value)}")
