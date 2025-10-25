#!/usr/bin/env python
"""
Test if RED vs GREEN metrics now differ with CONGESTED initial conditions.
The fix: Changed initial state from LIGHT (rho=0.01) to MEDIUM-HEAVY (rho=0.185)
This should make RED blocking create visible queueing, unlike before.
"""

import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'Code_RL' / 'src'))
sys.path.insert(0, str(project_root))

print("[TEST] Testing RED vs GREEN with CONGESTED initial conditions (HIGH DENSITY)")
print("="*80)

# Import after path setup
from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect
from Code_RL.src.utils.config import create_scenario_config_with_lagos_data

# Step 1: Create scenario config file with NEW CONGESTED initial conditions
print("\n[SETUP] Creating scenario config with CONGESTED initial state...")
scenario_dir = Path('validation_output/test_congested_fix')
scenario_dir.mkdir(parents=True, exist_ok=True)

scenario_path = scenario_dir / 'traffic_light_control_congested.yml'
config = create_scenario_config_with_lagos_data(
    scenario_type='traffic_light_control',
    output_path=scenario_path,
    duration=180.0  # 3 minutes
)

print(f"  ✓ Config created and saved to: {scenario_path}")
print(f"  ✓ Domain length: {config.get('xmax', 'N/A')} m")
print(f"  ✓ Total simulation time: {config.get('t_final', 'N/A')} s")

# Step 2: Run RED_ONLY baseline
print("\n[RED_ONLY] Running RED traffic light for 180s (12 steps of 15s)...")

env_red = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=15.0,
    episode_max_time=180.0,
    quiet=True
)
obs_red, _ = env_red.reset()

red_states = []
for step in range(12):  # 12 * 15s = 180s (3 minutes)
    action = 0.0  # RED light
    obs_red, reward_red, terminated, truncated, info = env_red.step(action)
    current_state = env_red.runner.U.copy()
    red_states.append(current_state)
    if step < 3 or step % 3 == 0:
        avg_density = (current_state[0].mean() + current_state[2].mean()) / 2
        print(f"  Step {step} (t={step*15}s): ρ_avg={avg_density:.4f} veh/m")

red_final_density = (red_states[-1][0].mean() + red_states[-1][2].mean()) / 2
print(f"  Final average density = {red_final_density:.4f} veh/m")

# Step 3: Run GREEN_ONLY baseline
print("\n[GREEN_ONLY] Running GREEN traffic light for 180s (12 steps of 15s)...")

env_green = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=15.0,
    episode_max_time=180.0,
    quiet=True
)
obs_green, _ = env_green.reset()

green_states = []
for step in range(12):  # 12 * 15s = 180s
    action = 1.0  # GREEN light
    obs_green, reward_green, terminated, truncated, info = env_green.step(action)
    current_state = env_green.runner.U.copy()
    green_states.append(current_state)
    if step < 3 or step % 3 == 0:
        avg_density = (current_state[0].mean() + current_state[2].mean()) / 2
        print(f"  Step {step} (t={step*15}s): ρ_avg={avg_density:.4f} veh/m")

green_final_density = (green_states[-1][0].mean() + green_states[-1][2].mean()) / 2
print(f"  Final average density = {green_final_density:.4f} veh/m")

# Step 4: Compare results
print("\n" + "="*80)
print("[RESULTS] Comparison:")
print(f"  RED_ONLY final density:   {red_final_density:.4f} veh/m")
print(f"  GREEN_ONLY final density: {green_final_density:.4f} veh/m")
print(f"  Absolute difference: {abs(red_final_density - green_final_density):.4f} veh/m")
print(f"  Relative difference: {abs(red_final_density - green_final_density) / green_final_density * 100:.1f}%")

if abs(red_final_density - green_final_density) > 0.01:
    print("\n✅ SUCCESS! Metrics now DIFFER based on traffic signal control!")
    print("   RED congests the domain (higher or same density)")
    print("   GREEN clears the domain (lower density)")
    if red_final_density > green_final_density:
        print("   ✓ RED shows HIGHER density (correct behavior)")
    else:
        print("   ⚠ GREEN shows HIGHER density (investigate)")
else:
    print("\n❌ FAILED! Metrics are still identical or very close")
    print("   The fix may not have worked, or the scenario needs adjustment")

print("="*80)
