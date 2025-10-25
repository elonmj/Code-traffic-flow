#!/usr/bin/env python3
"""
Debug test to verify RED_ONLY is actually setting phase=0 
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
print("DEBUG: Verify RED_ONLY strategy is working")
print("="*80)

# Setup
test = RLPerformanceValidationTest(quick_test=True)
scenario_type = 'traffic_light_control'
scenario_path = test._create_scenario_config(scenario_type)

# Test RED_ONLY strategy
print(f"\nCreating RED_ONLY controller...")
baseline_red = test.BaselineController(scenario_type, strategy='red_only')

print(f"\nTesting RED_ONLY get_action() for 5 calls:")
for i in range(5):
    dummy_state = np.array([0.01, 0.01])
    action = baseline_red.get_action(dummy_state)
    print(f"  Call {i+1}: action={action:.1f} (should be 0.0)")
    baseline_red.update(15.0)

# Now test in environment
print(f"\n" + "="*80)
print("Testing RED_ONLY in environment")
print("="*80 + "\n")

env = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=15.0,
    episode_max_time=60.0,
    observation_segments={'upstream': [3, 4, 5], 'downstream': [6, 7, 8]},
    device='cpu',
    quiet=False
)

obs, _ = env.reset()

baseline_red2 = test.BaselineController(scenario_type, strategy='red_only')

for step in range(3):
    print(f"\n--- Step {step} ---")
    action = baseline_red2.get_action(obs)
    print(f"Controller returned action: {action}")
    
    obs, reward, _, _, _ = env.step(action)
    
    # Check current phase in environment
    current_phase = env.current_phase
    print(f"Environment current_phase after step: {current_phase} (should be 0 for RED)")
    
    # Extract boundary state
    current_state = env.runner.d_U.copy_to_host() if env.device == 'gpu' else env.runner.U.copy()
    rho_m_bc = current_state[0, 0]
    print(f"Left boundary rho_m: {rho_m_bc:.6f} (should be 0.0 for RED)")

env.close()
