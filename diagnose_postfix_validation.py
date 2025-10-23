#!/usr/bin/env python3
"""Post-fix diagnostic: Verify that round(action) produces proper phase transitions and varied rewards"""
import sys
from pathlib import Path
import numpy as np

project_root = Path('.')
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'Code_RL'))

from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect
from Code_RL.src.utils.config import create_scenario_config_with_lagos_data

print('=' * 80)
print('POST-FIX VALIDATION: Action discretization with round()')
print('=' * 80)

# Step 1: Generate scenario
print('\n[STEP 1] Generate scenario...')
scenario_path = Path('validation_output/test_postfix_validation/traffic_light_control.yml')
scenario_path.parent.mkdir(parents=True, exist_ok=True)

config = create_scenario_config_with_lagos_data(
    scenario_type='traffic_light_control',
    output_path=scenario_path,
    duration=600.0,
    domain_length=1000.0
)
print(f'✓ Scenario: {scenario_path}')

# Test action conversion directly
print('\n[STEP 2] Test action conversion with round()...')
test_actions = [0.0, 0.1, 0.25, 0.3, 0.49, 0.5, 0.51, 0.7, 0.95, 0.99, 1.0]
print("  Action (float) → round(action) → Phase")
print("  " + "-" * 40)
for action in test_actions:
    phase = round(float(action))
    print(f"  {action:5.2f}      →      {phase}      →  {'GREEN' if phase else 'RED  '}")

# Step 3: Simulate with "RL-like" continuous actions
print('\n[STEP 3] Run simulation with continuous actions (simulating RL output)...')

class ContinuousRLController:
    """Simulates RL outputting continuous actions"""
    def __init__(self):
        self.step_count = 0
        # Pattern: smooth transition from 0→1 to see phase changes
        # This simulates an RL agent that learns to transition
        self.actions = np.linspace(0.0, 1.0, 40)  # Gradual ramp from RED to GREEN
    
    def get_action(self, obs):
        if self.step_count < len(self.actions):
            action = self.actions[self.step_count]
        else:
            action = 1.0
        self.step_count += 1
        return action

env = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=15.0,
    episode_max_time=600.0,
    observation_segments={'upstream': [3, 4, 5], 'downstream': [6, 7, 8]},
    device='cpu',
    quiet=False  # Show environment logs to see phase transitions
)

obs, info = env.reset()
controller = ContinuousRLController()

rl_actions_output = []  # What the RL outputs
rl_phases = []          # What phase gets set
rl_rewards = []         # What reward comes back

print("\n  Executing 40 steps with continuous RL actions...")
print("  Step | RL Action | Phase | Reward")
print("  " + "-" * 45)

for step in range(40):
    action = controller.get_action(obs)
    rl_actions_output.append(action)
    
    obs, reward, terminated, truncated, info = env.step(action)
    rl_phases.append(info.get('current_phase', 0))
    rl_rewards.append(reward)
    
    if step % 5 == 0 or step < 5:  # Print every 5 steps + first 5
        print(f"  {step:4d} | {action:9.3f} | {rl_phases[-1]:5} | {reward:7.4f}")
    
    if terminated or truncated:
        break

env.close()

print('\n' + '=' * 80)
print('RESULTS:')
print('=' * 80)

print(f"\nRL Actions (RL output):")
print(f"  Range: {np.min(rl_actions_output):.3f} to {np.max(rl_actions_output):.3f}")
print(f"  Pattern: {rl_actions_output[:5]} ... {rl_actions_output[-5:]}")

print(f"\nPhase transitions (what env.current_phase becomes):")
print(f"  Red (0) steps: {sum(1 for p in rl_phases if p == 0)}")
print(f"  Green (1) steps: {sum(1 for p in rl_phases if p == 1)}")
print(f"  Transitions: {sum(1 for i in range(1, len(rl_phases)) if rl_phases[i] != rl_phases[i-1])}")
print(f"  Phase sequence: {rl_phases[:10]} ... {rl_phases[-10:]}")

print(f"\nRewards (learning signal):")
print(f"  Min: {np.min(rl_rewards):.6f}")
print(f"  Max: {np.max(rl_rewards):.6f}")
print(f"  Mean: {np.mean(rl_rewards):.6f}")
print(f"  Std: {np.std(rl_rewards):.6f}")
print(f"  Unique values: {len(set(np.round(rl_rewards, 6)))}")

if len(set(rl_rewards)) > 1:
    print(f"  ✓ REWARDS VARY - Learning signal present!")
else:
    print(f"  ✗ REWARDS CONSTANT - No learning signal!")

if sum(1 for i in range(1, len(rl_phases)) if rl_phases[i] != rl_phases[i-1]) > 0:
    print(f"  ✓ PHASE TRANSITIONS OCCUR - Agent can affect environment!")
else:
    print(f"  ✗ PHASE STUCK - Agent can't control anything!")

print('\n' + '=' * 80)
print('CONCLUSION:')
print('=' * 80)

if len(set(rl_rewards)) > 1 and sum(1 for i in range(1, len(rl_phases)) if rl_phases[i] != rl_phases[i-1]) > 0:
    print("\n✅ BUG #37 FIX SUCCESSFUL!")
    print("   - RL receives VARIED reward signal (learning possible)")
    print("   - Phase TRANSITIONS as expected")
    print("   - RL agent can NOW learn to control traffic signals")
else:
    print("\n❌ BUG #37 NOT FIXED!")
    if len(set(rl_rewards)) == 1:
        print("   - Rewards still constant (no learning signal)")
    if sum(1 for i in range(1, len(rl_phases)) if rl_phases[i] != rl_phases[i-1]) == 0:
        print("   - Phase transitions missing (agent stuck)")

print('\n' + '=' * 80)
