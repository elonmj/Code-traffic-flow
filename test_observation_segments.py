# Test diagnostic: Observer segments près de l'inflow
# Usage: python test_observation_segments.py

import sys
sys.path.insert(0, 'Code_RL/src')

from env.traffic_signal_env_direct import TrafficSignalEnvDirect
import numpy as np

# Create env with NEAR-INFLOW segments
scenario_path = 'validation_output/results/elonmj_arz-validation-76rlperformance-cuyy/section_7_6_rl_performance/data/scenarios/traffic_light_control.yml'
env = TrafficSignalEnvDirect(
    scenario_config_path=scenario_path,
    observation_segments={
        'upstream': [2, 3, 4],    # x = 20-40m (PRÈS de l'inflow x=0)
        'downstream': [5, 6, 7]   # x = 50-70m (juste après)
    },
    quiet=True
)

print("🔬 DIAGNOSTIC: Testing near-inflow observation segments")
print(f"  Config: {scenario_path}")
print(f"  Segments: upstream=[2,3,4], downstream=[5,6,7]")
print()

# Run 10 steps
obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"First 6 observation values (rho, v for 3 upstream segments): {obs[:6]}")
print()

max_density_seen = 0.0

for step in range(10):
    action = step % 2  # Alternate phases
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Extract densities from observation (first 6 values = 3 segments × 2 vehicle classes)
    rho_m_seg2 = obs[0] * 0.25  # Denormalize (rho_max_m = 0.25 veh/m)
    rho_c_seg2 = obs[1] * 0.12  # Denormalize (rho_max_c = 0.12 veh/m)
    rho_total_seg2 = rho_m_seg2 + rho_c_seg2
    
    max_density_seen = max(max_density_seen, rho_total_seg2)
    
    print(f"Step {step+1}: t={info.get('time', 0):.1f}s action={action} reward={reward:.4f}")
    print(f"  Segment 2 (x=20m): rho_m={rho_m_seg2:.4f} rho_c={rho_c_seg2:.4f} rho_total={rho_total_seg2:.4f}")
    
    if rho_total_seg2 > 0.05:  # Threshold: > 5% jam density
        print(f"  🎉 TRAFFIC DETECTED at x=20m! Density={rho_total_seg2:.4f} veh/m")

print()
print(f"Maximum density observed: {max_density_seen:.4f} veh/m (jam = 0.37 veh/m)")

if max_density_seen < 0.01:
    print("❌ NO TRAFFIC ACCUMULATION - Problem is in ARZ model state evolution")
    print("   → Inflow BC not entering domain OR being dissipated immediately")
else:
    print("✅ Traffic is accumulating! Bug #34 fix works")
    print(f"   → Achieved {max_density_seen/0.37*100:.1f}% of jam density")

env.close()
