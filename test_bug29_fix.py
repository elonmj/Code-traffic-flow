"""
Quick local test of Bug #29 reward function fix
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect
import numpy as np

scenario_path = "validation_output/results/joselonm_arz-validation-76rlperformance-vmyo/section_7_6_rl_performance/data/scenarios/traffic_light_control.yml"

env = TrafficSignalEnvDirect(
    scenario_config_path=scenario_path,
    decision_interval=15.0,
    quiet=True,
    device='cpu'
)

print("="*80)
print("🧪 QUICK TEST - BUG #29 REWARD FUNCTION FIX")
print("="*80)
print(f"\nChanges:")
print(f"  1. Queue multiplier: 10.0 → 50.0 (5x amplification)")
print(f"  2. Phase change penalty: -0.1 → -0.01 (10x reduction)")
print(f"  3. Action diversity bonus: +0.02 if using both actions")

env.reset()

print(f"\n🔴 Testing Action 0 (RED) x 10:")
rewards_red = []
for i in range(10):
    obs, reward, done, truncated, info = env.step(0)
    rewards_red.append(reward)
    print(f"  Step {i+1}: reward={reward:+.4f}, phase={info['current_phase']}")

print(f"\n🟢 Testing Action 1 (GREEN) x 10:")
env.reset()
rewards_green = []
for i in range(10):
    obs, reward, done, truncated, info = env.step(1)
    rewards_green.append(reward)
    print(f"  Step {i+1}: reward={reward:+.4f}, phase={info['current_phase']}")

print(f"\n🔄 Testing Alternating (0,1,0,1,...) x 10:")
env.reset()
rewards_alt = []
for i in range(10):
    action = i % 2
    obs, reward, done, truncated, info = env.step(action)
    rewards_alt.append(reward)
    action_str = 'RED' if action == 0 else 'GRN'
    print(f"  Step {i+1}: action={action_str}, reward={reward:+.4f}, phase={info['current_phase']}")

print(f"\n📊 SUMMARY:")
print(f"  RED rewards: min={min(rewards_red):+.4f}, max={max(rewards_red):+.4f}, mean={np.mean(rewards_red):+.4f}, unique={len(set(rewards_red))}")
print(f"  GREEN rewards: min={min(rewards_green):+.4f}, max={max(rewards_green):+.4f}, mean={np.mean(rewards_green):+.4f}, unique={len(set(rewards_green))}")
print(f"  ALT rewards: min={min(rewards_alt):+.4f}, max={max(rewards_alt):+.4f}, mean={np.mean(rewards_alt):+.4f}, unique={len(set(rewards_alt))}")

if len(set(rewards_red + rewards_green + rewards_alt)) > 1:
    print(f"\n✅ SUCCESS: Rewards now have diversity!")
    print(f"   Total unique values: {len(set(rewards_red + rewards_green + rewards_alt))}")
else:
    print(f"\n❌ FAIL: Rewards still all identical")

env.close()

print(f"\n{'='*80}")
print(f"Test complete - Ready for Kaggle deployment!")
print(f"{'='*80}")
