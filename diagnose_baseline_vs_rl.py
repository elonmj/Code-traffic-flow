#!/usr/bin/env python3
"""Trace: Are baseline and RL simulations producing different states?"""
import sys
from pathlib import Path
import numpy as np
import yaml

project_root = Path('.')
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'Code_RL'))

from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect
from Code_RL.src.utils.config import create_scenario_config_with_lagos_data

print('=' * 80)
print('DIAGNOSTIC: Baseline vs RL State Trajectories')
print('=' * 80)

# Step 1: Generate scenario
print('\n[STEP 1] Generate scenario...')
scenario_path = Path('validation_output/test_diagnostic/traffic_light_control.yml')
scenario_path.parent.mkdir(parents=True, exist_ok=True)

config = create_scenario_config_with_lagos_data(
    scenario_type='traffic_light_control',
    output_path=scenario_path,
    duration=600.0,
    domain_length=1000.0
)
print(f'✓ Scenario: {scenario_path}')

# Step 2: Simulate baseline (deterministic fixed-time control)
print('\n[STEP 2] Run baseline simulation (fixed-time control)...')

class BaselineController:
    def __init__(self):
        self.time_step = 0
    
    def get_action(self, obs):
        # Fixed-time control: 60s green, 60s red
        return 1.0 if (self.time_step % 120) < 60 else 0.0
    
    def update(self, dt):
        self.time_step += dt

baseline_controller = BaselineController()
env = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=15.0,
    episode_max_time=600.0,
    observation_segments={'upstream': [3, 4, 5], 'downstream': [6, 7, 8]},
    device='cpu',
    quiet=True
)

obs, info = env.reset()
baseline_states = []
baseline_actions = []
baseline_rewards = []

for step in range(40):  # 40 steps × 15s = 600s
    action = baseline_controller.get_action(obs)
    baseline_actions.append(action)
    obs, reward, terminated, truncated, info = env.step(action)
    baseline_rewards.append(reward)
    
    # Store state
    state = env.runner.U.copy()
    baseline_states.append(state)
    baseline_controller.update(15.0)
    
    if terminated or truncated:
        break

env.close()
print(f'✓ Baseline: {len(baseline_states)} states, {len(baseline_actions)} actions')
print(f'  Actions: min={np.min(baseline_actions):.2f}, max={np.max(baseline_actions):.2f}')
print(f'  Rewards: min={np.min(baseline_rewards):.4f}, max={np.max(baseline_rewards):.4f}, mean={np.mean(baseline_rewards):.4f}')

# Compute baseline flow
baseline_flows = []
for state in baseline_states:
    # Extract densities and velocities
    rho_m = state[0, 5:-5].mean() if state.ndim == 2 else 0.01
    w_m = state[1, 5:-5].mean() if state.ndim == 2 else 10.0
    if rho_m > 1e-8:
        v_m = w_m / rho_m
        flow = rho_m * v_m
    else:
        flow = 0
    baseline_flows.append(flow)

baseline_flow = np.mean(baseline_flows)
print(f'  Average flow: {baseline_flow:.4f}')

# Step 3: Simulate RL (zero actions - no model)
print('\n[STEP 3] Run "RL simulation" (zero actions - mock)...')

# For this diagnostic, we can't load a real model, so we'll use random actions
# to simulate RL producing different actions

rl_controller_actions = [np.random.uniform(0, 1) for _ in range(40)]

env = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=15.0,
    episode_max_time=600.0,
    observation_segments={'upstream': [3, 4, 5], 'downstream': [6, 7, 8]},
    device='cpu',
    quiet=True
)

obs, info = env.reset()
rl_states = []
rl_rewards = []

for step, action in enumerate(rl_controller_actions):
    obs, reward, terminated, truncated, info = env.step(action)
    rl_rewards.append(reward)
    
    state = env.runner.U.copy()
    rl_states.append(state)
    
    if terminated or truncated:
        break

env.close()
print(f'✓ RL: {len(rl_states)} states, {len(rl_controller_actions)} actions')
print(f'  Actions: min={np.min(rl_controller_actions):.2f}, max={np.max(rl_controller_actions):.2f}')
print(f'  Rewards: min={np.min(rl_rewards):.4f}, max={np.max(rl_rewards):.4f}, mean={np.mean(rl_rewards):.4f}')

# Compute RL flow
rl_flows = []
for state in rl_states:
    rho_m = state[0, 5:-5].mean() if state.ndim == 2 else 0.01
    w_m = state[1, 5:-5].mean() if state.ndim == 2 else 10.0
    if rho_m > 1e-8:
        v_m = w_m / rho_m
        flow = rho_m * v_m
    else:
        flow = 0
    rl_flows.append(flow)

rl_flow = np.mean(rl_flows)
print(f'  Average flow: {rl_flow:.4f}')

# Step 4: Compare
print('\n[STEP 4] Comparison:')
print(f'  Baseline flow: {baseline_flow:.4f}')
print(f'  RL flow: {rl_flow:.4f}')
print(f'  Difference: {rl_flow - baseline_flow:.4f} ({((rl_flow - baseline_flow) / baseline_flow * 100):.1f}%)')

# Check if states are different
state_diffs = []
for i in range(min(len(baseline_states), len(rl_states))):
    diff = np.abs(baseline_states[i] - rl_states[i]).mean()
    state_diffs.append(diff)

print(f'\n[STEP 5] State trajectory differences:')
print(f'  Mean abs diff per state: {np.mean(state_diffs):.6e}')
print(f'  Max abs diff: {np.max(state_diffs):.6e}')

if np.max(state_diffs) > 1e-6:
    print(f'  ✓ States ARE different (good!)')
else:
    print(f'  ✗ States appear IDENTICAL (bad!)')
    print(f'    This suggests controllers are not affecting the simulation!')

print('\n' + '='*80)
