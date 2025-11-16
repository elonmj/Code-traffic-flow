"""Full inspection of simulation results"""
import pickle
import numpy as np

# Load results
with open('network_simulation_results.pkl', 'rb') as f:
    results = pickle.load(f)

hist = results['history']
print("Full structure:")
print(f"Times: {len(hist['time'])} points")
print(f"Time[0:5]: {hist['time'][:5]}")

for seg_id, seg_data in hist['segments'].items():
    print(f"\n{seg_id}:")
    density = seg_data['density']
    speed = seg_data['speed']
    print(f"  density: list of {len(density)} items")
    print(f"  speed: list of {len(speed)} items")
    
    if len(density) > 0:
        print(f"  density[0]: {type(density[0])}")
        if hasattr(density[0], 'shape'):
            print(f"    shape: {density[0].shape}")
            print(f"    sample values: {density[0][:5]}")
    
    if len(speed) > 0:
        print(f"  speed[0]: {type(speed[0])}")
        if hasattr(speed[0], 'shape'):
            print(f"    shape: {speed[0].shape}")
            print(f"    sample values: {speed[0][:5]}")

print("\nFinal states:")
final_states = results['final_states']
for seg_id, state in final_states.items():
    print(f"{seg_id}: {type(state)}")
    if hasattr(state, 'shape'):
        print(f"  shape: {state.shape}")
