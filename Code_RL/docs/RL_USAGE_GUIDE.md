# RL Traffic Signal Control - Usage Guide

This guide demonstrates how to use the RL environment for training traffic signal control agents with the ARZ macroscopic traffic flow model.

## Quick Start

```python
from Code_RL.src.config import create_rl_training_config
from Code_RL.src.env import TrafficSignalEnvDirectV2

# 1. Create RL-optimized configuration
config = create_rl_training_config(
    csv_topology_path='arz_model/data/fichier_de_travail_corridor_utf8.csv',
    episode_duration=1800.0,   # 30 min episodes
    decision_interval=15.0,     # RL decision every 15s
    default_density=20.0,       # Initial density (veh/km)
    inflow_density=30.0         # Entry traffic density
)

# 2. Initialize environment
env = TrafficSignalEnvDirectV2(simulation_config=config)

# 3. Standard Gymnasium interface
obs, info = env.reset()
for step in range(100):
    action = env.action_space.sample()  # Random policy
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episode ended at step {step}")
        obs, info = env.reset()
```

## Architecture

### MDP Formulation

**State Space** (Normalized [0, 1]):
- Density (ρ_m, ρ_c) for each observed segment
- Velocity (v_m, v_c) for each observed segment
- Current phase (one-hot encoding)
- Total dimension: 4 × N_segments + 2

**Action Space** (Discrete):
- 0 = Maintain current phase
- 1 = Switch to alternate phase

**Reward Function**:
```
r(s, a) = -α·congestion + μ·throughput - κ·phase_change

Where:
- congestion = average normalized density across segments
- throughput = velocity-weighted density (approximation)
- phase_change = 1 if action == 1, else 0
```

**Default Weights**:
- α = 1.0 (congestion penalty)
- μ = 0.5 (throughput reward)
- κ = 0.1 (phase change penalty)

### Direct GPU Coupling

The environment uses **direct in-process coupling** with the ARZ simulator:

1. **No HTTP overhead**: GPU arrays accessed directly via shared memory
2. **100-200x faster**: ~0.2-0.6ms action latency vs 50-100ms for HTTP
3. **Pydantic configuration**: Type-safe, no YAML parsing
4. **Cache system**: Config generation <10ms (vs 500-2000ms fresh)

## Configuration Examples

### Example 1: Short Episodes for Debugging

```python
config = create_rl_training_config(
    csv_topology_path='data/topology.csv',
    episode_duration=120.0,    # 2 minutes
    decision_interval=5.0,      # 5s decisions
    default_density=15.0,       # Light traffic
    quiet=True                  # Suppress output
)

env = TrafficSignalEnvDirectV2(
    simulation_config=config,
    reward_weights={'alpha': 1.0, 'kappa': 0.05, 'mu': 0.3}
)
```

### Example 2: Long Episodes for Training

```python
config = create_rl_training_config(
    csv_topology_path='data/victoria_island_topology.csv',
    episode_duration=3600.0,    # 1 hour
    decision_interval=15.0,      # 15s decisions (standard)
    default_density=25.0,        # Moderate traffic
    inflow_density=35.0,         # Heavy entry traffic
    cells_per_100m=10,           # High resolution
    quiet=False                  # Show progress
)
```

### Example 3: Custom Observation Segments

```python
config = create_rl_training_config(
    csv_topology_path='data/topology.csv',
    episode_duration=1800.0,
    decision_interval=10.0,
    observation_segment_ids=[   # Specify segments to observe
        '5902583245->95636900',
        '31674711->36240967',
        '36240967->95636908',
        '36240967->31674708'
    ]
)
```

## Training Examples

### DQN Training (Stable-Baselines3)

```python
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

# Create environment
config = create_rl_training_config(
    csv_topology_path='data/victoria_island_topology.csv',
    episode_duration=3600.0,
    decision_interval=15.0
)
env = TrafficSignalEnvDirectV2(simulation_config=config, quiet=True)

# Create DQN agent
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    tau=0.005,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1
)

# Train agent
model.learn(total_timesteps=100000, log_interval=100)

# Save model
model.save("dqn_traffic_signal")
```

### PPO Training

```python
from stable_baselines3 import PPO

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)

model.learn(total_timesteps=200000)
model.save("ppo_traffic_signal")
```

### Custom RL Algorithm

```python
import numpy as np

# Simple Q-learning example
Q = {}  # Q-table: (state, action) -> value
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate

obs, info = env.reset()
for episode in range(1000):
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(240):  # 1 hour @ 15s decisions
        # Discretize observation for Q-table
        state = tuple(np.round(obs, 1))
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_vals = [Q.get((state, a), 0.0) for a in range(2)]
            action = np.argmax(q_vals)
        
        # Take action
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        # Q-learning update
        next_state = tuple(np.round(next_obs, 1))
        max_next_q = max([Q.get((next_state, a), 0.0) for a in range(2)])
        Q[(state, action)] = Q.get((state, action), 0.0) + \
                             alpha * (reward + gamma * max_next_q - Q.get((state, action), 0.0))
        
        obs = next_obs
        
        if terminated or truncated:
            break
    
    print(f"Episode {episode}: reward = {episode_reward:.2f}")
```

## RLNetworkConfig Helper

The `RLNetworkConfig` class provides utilities for extracting signalized segments and managing phase mappings:

```python
from Code_RL.src.config.rl_network_config import RLNetworkConfig

# Initialize helper
rl_config = RLNetworkConfig(simulation_config)

# Get signalized segment IDs
signalized_ids = rl_config.signalized_segment_ids
print(f"Signalized segments: {signalized_ids}")

# Get phase mapping
phase_map = rl_config.phase_map
print(f"Phase map: {phase_map}")  # {0: 'green_NS', 1: 'green_EW'}

# Get phase updates for bulk API
updates = rl_config.get_phase_updates(phase=1)
# Returns: {'seg1': 'green_EW', 'seg2': 'green_EW', ...}

# Use with runner directly
runner.set_boundary_phases_bulk(updates, validate=False)
```

## Performance Optimization

### Disable Validation in Training

```python
# In environment's _apply_phase_to_network():
self.runner.set_boundary_phases_bulk(
    phase_updates=phase_updates,
    validate=False  # Skip validation for speed (already validated in config)
)
```

### Batch Episodes

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# Create vectorized environment (4 parallel workers)
def make_env():
    def _init():
        config = create_rl_training_config(...)
        return TrafficSignalEnvDirectV2(simulation_config=config, quiet=True)
    return _init

envs = SubprocVecEnv([make_env() for _ in range(4)])

# Train with vectorized environment
model = DQN("MlpPolicy", envs, verbose=1)
model.learn(total_timesteps=400000)  # 4x speedup
```

### Profile Performance

```python
import time

obs, info = env.reset()
step_times = []

for step in range(100):
    start = time.time()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    step_time = (time.time() - start) * 1000  # ms
    step_times.append(step_time)

print(f"Mean step time: {np.mean(step_times):.2f}ms")
print(f"Target: <1000ms")
```

## Troubleshooting

### CUDA Not Available

```
RuntimeError: CUDA not available. This GPU-only build requires an NVIDIA GPU.
```

**Solution**: Run on a machine with NVIDIA GPU and CUDA toolkit installed. Use Kaggle, Google Colab, or local GPU.

### No Signalized Nodes Detected

```
WARNING: No signalized nodes found!
```

**Solution**: Ensure OSM-enriched data is available:
```python
config = create_victoria_island_config(
    enriched_path="data/fichier_de_travail_complet_enriched.xlsx"
)
```

### Segment ID Not Found

```
KeyError: Segment '12345->67890' not found.
```

**Solution**: Use `rl_config.signalized_segment_ids` to get valid segment IDs:
```python
rl_config = RLNetworkConfig(simulation_config)
valid_ids = rl_config.signalized_segment_ids
print(f"Valid segment IDs: {valid_ids}")
```

### Slow Episode Execution

**Causes**:
1. Too many observation segments (reduce to 6-10)
2. High grid resolution (`cells_per_100m` too large)
3. Validation enabled (disable with `validate=False`)
4. CPU-GPU transfers (shouldn't happen with direct coupling)

**Solutions**:
```python
# Reduce observation segments
config = create_rl_training_config(
    observation_segment_ids=['seg1', 'seg2', 'seg3']  # Fewer segments
)

# Lower resolution
config = create_rl_training_config(
    cells_per_100m=4  # Default: 10
)

# Disable validation in training
env._apply_phase_to_network(phase=1)  # Already has validate=False
```

## Testing

### API Integration Test (No GPU Required)

```bash
python Code_RL/tests/test_rl_api_integration.py
```

Validates:
- Config generation API
- RLNetworkConfig helper class
- SimulationRunner API methods
- Environment integration

### Full Integration Test (GPU Required)

```bash
python Code_RL/tests/test_rl_signal_integration.py
```

Runs complete RL episode with phase switching and performance profiling.

## Next Steps

1. **Baseline Comparison**: Compare RL agent vs fixed-time signals
2. **Multi-Intersection Coordination**: Extend to coordinated control
3. **Real-World Validation**: Test with real traffic demand data
4. **Advanced Rewards**: Implement queue length, delay, or throughput rewards
5. **Transfer Learning**: Train on Victoria Island, transfer to other cities

## References

- **ARZ Model**: Aw & Rascle (2000), Zhang (2002)
- **RL Traffic Control Survey**: Wei et al. (2019) - arXiv:1904.08117
- **PressLight**: Wei et al. (2019) - AAAI 2019
- **IntelliLight**: Wei et al. (2018) - KDD 2018
- **CityFlow**: Zhang et al. (2019) - WWW 2019

## Support

For issues or questions:
- Check `Code_RL/tests/` for examples
- See `arz_model/README.md` for simulator details
- Review `.copilot-tracking/plans/` for implementation notes
