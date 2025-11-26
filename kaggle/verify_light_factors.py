import sys
import os
import numpy as np
from numba import cuda

# Add project root to path
sys.path.append(os.getcwd())

from arz_model.simulation.runner import SimulationRunner
from arz_model.config.config_factory import create_victoria_island_config

def verify_light_factors():
    print("Initializing SimulationRunner...")
    # Use the helper function that handles default paths
    config = create_victoria_island_config()
    runner = SimulationRunner(simulation_config=config, device='gpu')
    
    # Get valid segment IDs
    segment_ids = list(runner.network_grid.segments.keys())
    print(f"Found {len(segment_ids)} segments.")
    print(f"Sample IDs: {segment_ids[:5]}")
    
    # Pick a few segments to turn RED
    test_segments = segment_ids[:3]
    red_phases = {seg_id: 1 for seg_id in test_segments} # 1 is NOT 0, so it should be RED (0.01)
    
    print(f"Setting phases to RED for: {test_segments}")
    runner.set_boundary_phases_bulk(red_phases)
    
    # Access GPU pool directly
    pool = runner.network_simulator.gpu_pool
    d_light_factors = pool.d_light_factors
    
    # Copy back to host
    light_factors_host = d_light_factors.copy_to_host()
    
    print("\nVerifying Light Factors on GPU:")
    all_correct = True
    for seg_id in test_segments:
        idx = pool.segment_id_to_index[seg_id]
        val = light_factors_host[idx]
        print(f"Segment {seg_id} (idx {idx}): light_factor = {val}")
        
        if abs(val - 0.01) < 1e-6:
            print("  ✅ CORRECT: Light factor is 0.01 (RED)")
        else:
            print(f"  ❌ ERROR: Light factor is {val} (Expected 0.01)")
            all_correct = False

    # Check a segment that should be GREEN (default)
    green_seg = segment_ids[5]
    idx_green = pool.segment_id_to_index[green_seg]
    val_green = light_factors_host[idx_green]
    print(f"Segment {green_seg} (idx {idx_green}): light_factor = {val_green}")
    if abs(val_green - 1.0) < 1e-6:
        print("  ✅ CORRECT: Light factor is 1.0 (GREEN)")
    else:
        print(f"  ❌ ERROR: Light factor is {val_green} (Expected 1.0)")
        all_correct = False
        
    if all_correct:
        print("\n✅ SUCCESS: All light factors verified correctly on GPU.")
    else:
        print("\n❌ FAILURE: Some light factors are incorrect.")
        sys.exit(1)

if __name__ == "__main__":
    verify_light_factors()
