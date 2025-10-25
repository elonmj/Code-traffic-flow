#!/usr/bin/env python3
"""
Detailed diagnostic: Check boundary state evolution over time for RED vs GREEN
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

print("="*80)
print("DIAGNOSTIC: Boundary state evolution - RED vs GREEN")
print("="*80)

# Setup
test = RLPerformanceValidationTest(quick_test=True)
scenario_type = 'traffic_light_control'
scenario_path = test._create_scenario_config(scenario_type)

# Test RED
print(f"\n=== RED PHASE ===")
env_red = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=15.0,
    episode_max_time=90.0,  # 90 seconds = 6 steps
    observation_segments={'upstream': [3, 4, 5], 'downstream': [6, 7, 8]},
    device='cpu',
    quiet=True
)

obs, _ = env_red.reset()
baseline_red = test.BaselineController(scenario_type, strategy='red_only')

red_boundary_rho = []
red_times = []

for step in range(6):
    action = baseline_red.get_action(obs)
    obs, _, _, _, _ = env_red.step(action)
    
    # Extract boundary state (BOTH ghost and physical)
    current_state = env_red.runner.U.copy()
    
    # Ghost cells
    rho_m_ghost = current_state[0, 0:3]
    
    # First physical cell (after ghost cells)
    rho_m_phys0 = current_state[0, 3]
    rho_m_phys1 = current_state[0, 4]
    
    red_boundary_rho.append(rho_m_phys0)
    red_times.append(env_red.runner.t)
    
    print(f"Step {step} t={env_red.runner.t:6.1f}s: ghost=[{rho_m_ghost[0]:.4f}, {rho_m_ghost[1]:.4f}, {rho_m_ghost[2]:.4f}], phys[0]={rho_m_phys0:.4f}, phys[1]={rho_m_phys1:.4f}")
    
    baseline_red.update(15.0)

env_red.close()

# Test GREEN
print(f"\n=== GREEN PHASE ===")
env_green = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=15.0,
    episode_max_time=90.0,
    observation_segments={'upstream': [3, 4, 5], 'downstream': [6, 7, 8]},
    device='cpu',
    quiet=True
)

obs, _ = env_green.reset()
baseline_green = test.BaselineController(scenario_type, strategy='green_only')

green_boundary_rho = []
green_times = []

for step in range(6):
    action = baseline_green.get_action(obs)
    obs, _, _, _, _ = env_green.step(action)
    
    # Extract boundary state
    current_state = env_green.runner.U.copy()
    
    # Ghost cells
    rho_m_ghost = current_state[0, 0:3]
    
    # First physical cell
    rho_m_phys0 = current_state[0, 3]
    rho_m_phys1 = current_state[0, 4]
    
    green_boundary_rho.append(rho_m_phys0)
    green_times.append(env_green.runner.t)
    
    print(f"Step {step} t={env_green.runner.t:6.1f}s: ghost=[{rho_m_ghost[0]:.4f}, {rho_m_ghost[1]:.4f}, {rho_m_ghost[2]:.4f}], phys[0]={rho_m_phys0:.4f}, phys[1]={rho_m_phys1:.4f}")
    
    baseline_green.update(15.0)

env_green.close()

print(f"\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"\nRED boundary density:   {red_boundary_rho}")
print(f"GREEN boundary density: {green_boundary_rho}")

# Check if they differ
red_avg = np.mean(red_boundary_rho)
green_avg = np.mean(green_boundary_rho)

print(f"\nRED avg boundary rho:   {red_avg:.6f}")
print(f"GREEN avg boundary rho: {green_avg:.6f}")
print(f"Difference: {green_avg - red_avg:.6f}")

if abs(green_avg - red_avg) > 0.001:
    print(f"\n✅ Boundary states DIFFER significantly")
else:
    print(f"\n❌ Boundary states are SAME")
