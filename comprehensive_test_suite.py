#!/usr/bin/env python
"""
Comprehensive Test Suite: Hardcoded Metrics Fix Verification

Tests:
1. Scenario creation with high initial density
2. RED phase traffic control (block inflow)
3. GREEN phase traffic control (allow inflow)
4. Difference quantification
5. Physics validation
"""

import sys
from pathlib import Path
import numpy as np

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'Code_RL' / 'src'))
sys.path.insert(0, str(project_root))

print("[TEST SUITE] Hardcoded Metrics Fix Verification")
print("="*80)

# Import after path setup
from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect
from Code_RL.src.utils.config import create_scenario_config_with_lagos_data

# TEST 1: Scenario Creation
print("\n[TEST 1] Scenario Creation with High Initial Density")
scenario_dir = Path('validation_output/test_suite')
scenario_dir.mkdir(parents=True, exist_ok=True)
scenario_path = scenario_dir / 'comprehensive_test.yml'

config = create_scenario_config_with_lagos_data(
    scenario_type='traffic_light_control',
    output_path=scenario_path,
    duration=180.0
)

print(f"  ✓ Config created: {scenario_path}")
print(f"  ✓ Domain: 0-{config.get('xmax', 'N/A')} m")
print(f"  ✓ Duration: {config.get('t_final', 'N/A')} s")

# TEST 2: RED Phase Control
print("\n[TEST 2] RED Phase Control (Block Inflow)")
env_red = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=15.0,
    episode_max_time=180.0,
    quiet=True
)
obs_red, _ = env_red.reset()

red_densities = []
for step in range(12):
    action = 0.0  # RED = 0
    obs_red, reward, terminated, truncated, info = env_red.step(action)
    state = env_red.runner.U.copy()
    rho_m = state[0].mean()
    rho_c = state[2].mean()
    avg_density = (rho_m + rho_c) / 2
    red_densities.append(avg_density)

red_final = red_densities[-1]
print(f"  ✓ Initial density: {red_densities[0]:.4f} veh/m")
print(f"  ✓ Final density:   {red_final:.4f} veh/m")
print(f"  ✓ Change: {red_densities[-1] - red_densities[0]:+.4f} veh/m ({(red_densities[-1] / red_densities[0] - 1)*100:+.1f}%)")

# TEST 3: GREEN Phase Control
print("\n[TEST 3] GREEN Phase Control (Allow Inflow)")
env_green = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=15.0,
    episode_max_time=180.0,
    quiet=True
)
obs_green, _ = env_green.reset()

green_densities = []
for step in range(12):
    action = 1.0  # GREEN = 1
    obs_green, reward, terminated, truncated, info = env_green.step(action)
    state = env_green.runner.U.copy()
    rho_m = state[0].mean()
    rho_c = state[2].mean()
    avg_density = (rho_m + rho_c) / 2
    green_densities.append(avg_density)

green_final = green_densities[-1]
print(f"  ✓ Initial density: {green_densities[0]:.4f} veh/m")
print(f"  ✓ Final density:   {green_final:.4f} veh/m")
print(f"  ✓ Change: {green_densities[-1] - green_densities[0]:+.4f} veh/m ({(green_densities[-1] / green_densities[0] - 1)*100:+.1f}%)")

# TEST 4: Difference Quantification
print("\n[TEST 4] Difference Quantification")
abs_diff = abs(red_final - green_final)
rel_diff = abs_diff / max(red_final, green_final) * 100

print(f"  Absolute difference: {abs_diff:.4f} veh/m")
print(f"  Relative difference: {rel_diff:.1f}%")

# TEST 5: Physics Validation
print("\n[TEST 5] Physics Validation")
physics_valid = True
errors = []

# RED should have lower density (inflow blocked)
if red_final >= green_final:
    physics_valid = False
    errors.append(f"  ✗ RED density ({red_final:.4f}) should be < GREEN density ({green_final:.4f})")
else:
    print(f"  ✓ RED density ({red_final:.4f}) < GREEN density ({green_final:.4f})")

# Difference should be significant
if rel_diff < 5:
    physics_valid = False
    errors.append(f"  ✗ Difference {rel_diff:.1f}% is too small (expected >5%)")
else:
    print(f"  ✓ Difference {rel_diff:.1f}% is significant (>5%)")

# Both should be in reasonable range
if red_final < 0.05 or red_final > 0.3:
    errors.append(f"  ✗ RED density {red_final:.4f} outside valid range [0.05, 0.3]")
else:
    print(f"  ✓ RED density {red_final:.4f} in valid range [0.05, 0.3]")

if green_final < 0.05 or green_final > 0.3:
    errors.append(f"  ✗ GREEN density {green_final:.4f} outside valid range [0.05, 0.3]")
else:
    print(f"  ✓ GREEN density {green_final:.4f} in valid range [0.05, 0.3]")

# Final Report
print("\n" + "="*80)
print("[RESULTS] Hardcoded Metrics Fix Verification")
print("="*80)

if physics_valid and len(errors) == 0:
    print("\n✅ **ALL TESTS PASSED**")
    print("\nMetrics Summary:")
    print(f"  RED phase (block inflow):  ρ = {red_final:.4f} veh/m")
    print(f"  GREEN phase (allow inflow): ρ = {green_final:.4f} veh/m")
    print(f"  Difference: {abs_diff:.4f} veh/m ({rel_diff:.1f}%)")
    print("\nConclusion:")
    print("  ✓ Hardcoded metrics bug is FIXED")
    print("  ✓ Traffic signal control is WORKING")
    print("  ✓ Physics behavior is CORRECT")
    print("  ✓ Metrics show SIGNIFICANT differences")
    exit(0)
else:
    print("\n❌ **TESTS FAILED**")
    for error in errors:
        print(error)
    exit(1)
