"""Quick inspection of simulation results"""
import pickle
import numpy as np

# Load results
with open('network_simulation_results.pkl', 'rb') as f:
    results = pickle.load(f)

print("=" * 60)
print("SIMULATION RESULTS INSPECTION")
print("=" * 60)

print("\n📦 Top-level keys:", list(results.keys()))
print(f"   Final time: {results.get('final_time', 'N/A')}")
print(f"   Total steps: {results.get('total_steps', 'N/A')}")

hist = results['history']
print(f"\n⏱️  History:")
print(f"   Time points: {len(hist['time'])}")
print(f"   Time range: {hist['time'][0]:.1f}s to {hist['time'][-1]:.1f}s")
print(f"   Segments: {list(hist['segments'].keys())}")

for seg_id, seg_data in hist['segments'].items():
    print(f"\n🛣️  Segment: {seg_id}")
    print(f"   Keys: {list(seg_data.keys())}")
    if 'states' in seg_data:
        states = seg_data['states']
        print(f"   States shape: {states.shape}")
        print(f"   States dtype: {states.dtype}")
        print(f"   ρ_m range: [{np.min(states[:, 0, :]):.6f}, {np.max(states[:, 0, :]):.6f}]")
        print(f"   ρ_c range: [{np.min(states[:, 2, :]):.6f}, {np.max(states[:, 2, :]):.6f}]")
    if 'grid_info' in seg_data:
        grid = seg_data['grid_info']
        print(f"   Grid info: {list(grid.keys())}")
        print(f"   Length: {grid.get('length', 'N/A')} m")
        print(f"   Cells: {grid.get('nx', 'N/A')}")

print("\n" + "=" * 60)
