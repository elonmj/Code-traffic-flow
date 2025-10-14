"""
Test script for queue-based reward function fix.
"""
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect

print("="*80)
print("Testing Queue-Based Reward Function Fix")
print("="*80)
print()

# Simplified test using mock environment
print("Creating environment...")
env = TrafficSignalEnvDirect(
    scenario_config_path="Code_RL/configs/traffic_lagos.yaml",
    decision_interval=15.0,
    episode_max_time=300,
    reward_weights={"alpha": 1.0, "kappa": 0.1, "mu": 0.5}
)
print(" Environment created")
print()

# Run 3 quick episodes
print("Running 3 test episodes...")
for ep in range(3):
    obs = env.reset()
    done = False
    ep_reward = 0.0
    actions = {0:0, 1:0}
    steps = 0
    
    while not done and steps < 20:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        actions[action] += 1
        steps += 1
    
    green_pct = (actions[1]/(actions[0]+actions[1]))*100 if (actions[0]+actions[1])>0 else 0
    print(f"Episode {ep+1}: Reward={ep_reward:6.2f}, Steps={steps}, GREEN={actions[1]} ({green_pct:.0f}%), RED={actions[0]}")

env.close()
print()
print(" Test completed successfully - reward function operational")

