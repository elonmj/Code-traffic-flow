#!/usr/bin/env python3
"""
Check if RED and GREEN simulations start with identical initial states
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
print("CHECK: Do RED and GREEN start with identical initial states?")
print("="*80)

# Setup test
test = RLPerformanceValidationTest(quick_test=True)
scenario_type = 'traffic_light_control'
scenario_path = test._create_scenario_config(scenario_type)

# Create RED environment and get initial state
print(f"\nCreating RED environment...")
env_red = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=15.0,
    episode_max_time=300.0,
    observation_segments={'upstream': [3, 4, 5], 'downstream': [6, 7, 8]},
    device='cpu',
    quiet=True
)

print(f"Resetting RED environment...")
obs_red, _ = env_red.reset()
state_red_initial = env_red.runner.d_U.copy_to_host() if env_red.device == 'gpu' else env_red.runner.U.copy()

print(f"RED initial state shape: {state_red_initial.shape}")
print(f"RED initial rho_m mean: {state_red_initial[0, :].mean():.6f}")
print(f"RED initial w_m mean: {state_red_initial[1, :].mean():.6f}")

env_red.close()

# Create GREEN environment and get initial state
print(f"\nCreating GREEN environment...")
env_green = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=15.0,
    episode_max_time=300.0,
    observation_segments={'upstream': [3, 4, 5], 'downstream': [6, 7, 8]},
    device='cpu',
    quiet=True
)

print(f"Resetting GREEN environment...")
obs_green, _ = env_green.reset()
state_green_initial = env_green.runner.d_U.copy_to_host() if env_green.device == 'gpu' else env_green.runner.U.copy()

print(f"GREEN initial state shape: {state_green_initial.shape}")
print(f"GREEN initial rho_m mean: {state_green_initial[0, :].mean():.6f}")
print(f"GREEN initial w_m mean: {state_green_initial[1, :].mean():.6f}")

env_green.close()

# Compare
print(f"\n" + "="*80)
print("COMPARISON OF INITIAL STATES")
print("="*80)

diff_rho = np.abs(state_red_initial[0, :] - state_green_initial[0, :]).max()
diff_w = np.abs(state_red_initial[1, :] - state_green_initial[1, :]).max()

print(f"Max difference in rho_m: {diff_rho:.10f}")
print(f"Max difference in w_m: {diff_w:.10f}")

if diff_rho < 1e-10 and diff_w < 1e-10:
    print(f"\n✅ Initial states ARE IDENTICAL (perfect)")
else:
    print(f"\n❌ Initial states DIFFER")
