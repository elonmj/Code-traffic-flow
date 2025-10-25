#!/usr/bin/env python3
"""
Diagnostic script to verify BC changes affect simulation state
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Code_RL" / "src"))
sys.path.insert(0, str(project_root / "validation_ch7" / "scripts"))

from validation_ch7.scripts.test_section_7_6_rl_performance import RLPerformanceValidationTest
from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect
import yaml

print("="*80)
print("DIAGNOSTIC: Verify BC Changes Affect Simulation")
print("="*80)

# Setup test
test = RLPerformanceValidationTest(quick_test=True)
scenario_type = 'traffic_light_control'
scenario_path = test._create_scenario_config(scenario_type)

# Create environment
print(f"\nCreating environment...")
env = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=15.0,
    episode_max_time=60.0,  # Just 60 seconds
    observation_segments={'upstream': [3, 4, 5], 'downstream': [6, 7, 8]},
    device='cpu',
    quiet=False
)

print(f"\nResetting environment...")
obs, info = env.reset()

# Run RED phase for a few steps
print(f"\n" + "="*80)
print("PHASE 1: RED LIGHT (phase_id=0, rho=0.0)")
print("="*80)

red_states = []
for step in range(3):
    print(f"\n--- Step {step} ---")
    action = 0.0  # RED
    obs, reward, terminated, truncated, info = env.step(action)
    current_state = env.runner.d_U.copy_to_host() if env.device == 'gpu' else env.runner.U.copy()
    red_states.append(current_state)
    
    # Extract left boundary state (first cell)
    rho_m_bc = current_state[0, 0]
    w_m_bc = current_state[1, 0]
    rho_c_bc = current_state[2, 0]
    w_c_bc = current_state[3, 0]
    
    print(f"Left BC state: rho_m={rho_m_bc:.6f}, w_m={w_m_bc:.6f}, rho_c={rho_c_bc:.6f}, w_c={w_c_bc:.6f}")
    
    # Mean state across domain
    rho_m_avg = current_state[0, :].mean()
    w_m_avg = current_state[1, :].mean()
    print(f"Mean domain state: rho_m={rho_m_avg:.6f}, w_m={w_m_avg:.6f}")
    
    # Flow at boundary
    if rho_m_bc > 1e-8:
        v_m_bc = w_m_bc / rho_m_bc
        flow_bc = rho_m_bc * v_m_bc
    else:
        flow_bc = 0.0
    print(f"Boundary flow: {flow_bc:.6f} veh/(m·s)")

# Save the final RED state
red_final = red_states[-1].copy()

# Reset and test GREEN
print(f"\n" + "="*80)
print("Resetting environment for PHASE 2...")
print("="*80 + "\n")

obs, info = env.reset()

print(f"="*80)
print("PHASE 2: GREEN LIGHT (phase_id=1)")
print("="*80)

green_states = []
for step in range(3):
    print(f"\n--- Step {step} ---")
    action = 1.0  # GREEN
    obs, reward, terminated, truncated, info = env.step(action)
    current_state = env.runner.d_U.copy_to_host() if env.device == 'gpu' else env.runner.U.copy()
    green_states.append(current_state)
    
    # Extract left boundary state (first cell)
    rho_m_bc = current_state[0, 0]
    w_m_bc = current_state[1, 0]
    rho_c_bc = current_state[2, 0]
    w_c_bc = current_state[3, 0]
    
    print(f"Left BC state: rho_m={rho_m_bc:.6f}, w_m={w_m_bc:.6f}, rho_c={rho_c_bc:.6f}, w_c={w_c_bc:.6f}")
    
    # Mean state across domain
    rho_m_avg = current_state[0, :].mean()
    w_m_avg = current_state[1, :].mean()
    print(f"Mean domain state: rho_m={rho_m_avg:.6f}, w_m={w_m_avg:.6f}")
    
    # Flow at boundary
    if rho_m_bc > 1e-8:
        v_m_bc = w_m_bc / rho_m_bc
        flow_bc = rho_m_bc * v_m_bc
    else:
        flow_bc = 0.0
    print(f"Boundary flow: {flow_bc:.6f} veh/(m·s)")

# Comparison
print(f"\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nRED final boundary state: rho_m={red_final[0, 0]:.6f}, w_m={red_final[1, 0]:.6f}")
print(f"GREEN final boundary state: rho_m={green_states[-1][0, 0]:.6f}, w_m={green_states[-1][1, 0]:.6f}")

# Check if states are significantly different
rho_diff = abs(red_final[0, :].mean() - green_states[-1][0, :].mean())
w_diff = abs(red_final[1, :].mean() - green_states[-1][1, :].mean())

print(f"\nMean density difference: {rho_diff:.6f}")
print(f"Mean momentum difference: {w_diff:.6f}")

if rho_diff > 0.001:
    print(f"\n✅ STATES ARE DIFFERENT - BC changes ARE having effect!")
elif w_diff > 0.001:
    print(f"\n✅ STATES ARE DIFFERENT (momentum) - BC changes ARE having effect!")
else:
    print(f"\n❌ STATES ARE IDENTICAL - BC changes ARE NOT having effect!")
