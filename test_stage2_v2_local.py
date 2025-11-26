"""Quick test of Stage 2 v2 environment"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing Stage 2 v2 Environment")
print("=" * 50)

# Import only the environment, not SB3
from Code_RL.src.env.traffic_signal_env_direct_v3 import TrafficSignalEnvDirectV3
from arz_model.config import create_victoria_island_config

# Create config
print("Creating config...")
start = time.time()
arz_config = create_victoria_island_config(
    t_final=450.0,
    output_dt=15.0,
    cells_per_100m=4,
    default_density=80.0,
    inflow_density=100.0,
    use_cache=False
)
print(f"Config created in {time.time()-start:.2f}s")

# Setup rl metadata
arz_config.rl_metadata = {
    'observation_segment_ids': [s.id for s in arz_config.segments],
    'decision_interval': 15.0,
}

# Create env with new reward weights
print("\nCreating environment with improved reward weights...")
print("  alpha=5.0, kappa=0.3, mu=0.1")
start = time.time()
env = TrafficSignalEnvDirectV3(
    simulation_config=arz_config,
    decision_interval=15.0,
    observation_segment_ids=None,
    reward_weights={'alpha': 5.0, 'kappa': 0.3, 'mu': 0.1},
    quiet=True
)
print(f"Environment created in {time.time()-start:.2f}s")

# Test reset
print("\nTesting reset...")
start = time.time()
obs, info = env.reset()
print(f"Reset in {time.time()-start:.2f}s")
print(f"Observation shape: {obs.shape}")

# Test a few steps
print("\nTesting steps...")
total_reward = 0
for i in range(5):
    start = time.time()
    action = 0 if i % 2 == 0 else 1
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    print(f"  Step {i+1}: action={action}, reward={reward:.3f}, time={time.time()-start:.2f}s")

print(f"\nTotal reward (5 steps): {total_reward:.2f}")
print(f"Avg density: {info.get('avg_density', 'N/A')}")

env.close()
print("\n✅ Local test PASSED!")
