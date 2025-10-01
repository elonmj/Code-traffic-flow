# Traffic Signal RL Environment

A reinforcement learning system for optimizing traffic signal control using an external ARZ (Aw-Rascle-Zhang) simulator adapted for West African urban contexts.

## Overview

This project implements a complete RL training pipeline for traffic signal optimization:

- **ARZ Simulator Interface**: External traffic simulator with multi-class vehicle support
- **RL Environment**: Gymnasium-compatible environment for signal control
- **Safety Layer**: Ensures traffic signal timing constraints and safety rules
- **Training Pipeline**: DQN baseline with evaluation and comparison tools

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test Setup**
   ```bash
   python test_setup.py
   ```

3. **Train Agent (Mock Mode)**
   ```bash
   python train.py --use-mock --timesteps 10000
   ```

4. **Train with Real ARZ Simulator**
   ```bash
   python train.py --config-dir configs --timesteps 100000
   ```

## Project Structure

```
Code_RL/
├── src/                    # Source code
│   ├── endpoint/          # ARZ simulator client
│   ├── signals/           # Traffic signal controller
│   ├── env/               # RL environment
│   ├── rl/                # Training algorithms
│   └── utils/             # Utilities and configuration
├── configs/               # Configuration files
├── tests/                 # Unit tests
├── results/               # Training outputs
├── data/                  # Traffic data
└── docs/                  # Documentation
```

## Configuration

The system uses YAML configuration files:

- `endpoint.yaml`: ARZ simulator connection settings
- `signals.yaml`: Traffic signal phases and timing
- `network.yaml`: Road network topology
- `env.yaml`: RL environment parameters

## Key Features

### ARZ Simulator Integration
- HTTP/IPC/Python binding support
- Mock simulator for testing
- Robust error handling and retries
- Real-time traffic state monitoring

### Safety Layer
- Minimum/maximum green time enforcement
- Yellow and all-red clearance intervals
- Conflict prevention
- Fail-safe operation

### RL Environment
- Gymnasium-compatible interface
- Normalized observations (densities, velocities, queues)
- Multi-objective reward function
- Comprehensive KPI tracking

### Training Pipeline
- DQN baseline implementation
- Fixed-time baseline comparison
- Experiment tracking and logging
- Performance evaluation metrics

## Usage Examples

### Basic Training
```python
from src.utils.config import load_configs
from src.rl.train_dqn import main

# Train with default settings
exit_code = main()
```

### Custom Training
```bash
python train.py \
    --config-dir configs \
    --output-dir my_results \
    --experiment-name custom_experiment \
    --timesteps 50000 \
    --eval-episodes 20 \
    --seed 123
```

### Environment Testing
```python
import numpy as np
from src.env.traffic_signal_env import TrafficSignalEnv
from src.endpoint.client import create_endpoint_client, EndpointConfig
from src.signals.controller import create_signal_controller
from src.utils.config import load_configs

# Load configuration
configs = load_configs("configs")

# Create components
endpoint = create_endpoint_client(EndpointConfig(protocol="mock"))
controller = create_signal_controller(configs["signals"])
branch_ids = ["north_in", "south_in", "east_in", "west_in"]

# Create environment
env = TrafficSignalEnv(endpoint, controller, config, branch_ids)

# Test episode
obs, info = env.reset(seed=42)
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
```

## Performance Metrics

The system tracks multiple KPIs:

- **Queue Length**: Average and maximum queue lengths
- **Wait Time**: Estimated vehicle waiting time
- **Throughput**: Traffic flow rate through intersection
- **Phase Switches**: Number of signal changes per episode
- **Speed**: Average vehicle velocities by class

## Configuration Details

### Signal Phases
```yaml
phases:
  - id: 0
    name: "north_south"
    movements: ["north_through", "south_through"]
  - id: 1  
    name: "east_west"
    movements: ["east_through", "west_through"]
```

### Reward Function
```yaml
reward:
  w_wait_time: 1.0        # Waiting time penalty
  w_queue_length: 0.5     # Queue length penalty
  w_stops: 0.3            # Stop/start penalty
  w_switch_penalty: 0.1   # Phase change penalty
  w_throughput: 0.8       # Throughput reward
```

### Normalization
```yaml
normalization:
  rho_max_motorcycles: 300.0  # Max motorcycle density
  rho_max_cars: 150.0         # Max car density
  v_free_motorcycles: 40.0    # Free flow speed (motorcycles)
  v_free_cars: 50.0           # Free flow speed (cars)
```

## West African Traffic Specifics

The system is designed for West African traffic contexts:

- **Multi-class modeling**: Separate motorcycle and car dynamics
- **Behavioral parameters**: Gap-filling, interweaving, creeping behaviors
- **Infrastructure quality**: Variable road surface conditions
- **Data adaptation**: Uses Lagos corridor data with Benin behavioral patterns

## Testing

Run unit tests:
```bash
pytest tests/
```

Run integration test:
```bash
python test_setup.py
```

## Results Analysis

Training results include:

- Episode rewards and KPIs
- Performance comparison vs fixed-time baseline
- Phase switching patterns
- Traffic flow analysis
- Model checkpoints and logs

## Troubleshooting

### Common Issues

1. **Configuration Validation Errors**
   - Check dt_decision is multiple of dt_sim
   - Verify all branch IDs are unique
   - Ensure phase definitions are consistent

2. **Endpoint Connection Issues**
   - Verify ARZ simulator is running
   - Check endpoint URL and port
   - Test with mock client first

3. **Training Instability**
   - Adjust reward clipping parameters
   - Reduce learning rate
   - Increase replay buffer size

### Debug Mode
```bash
python train.py --use-mock --timesteps 1000 --experiment-name debug
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{traffic_rl_west_africa,
  title={Reinforcement Learning for Traffic Signal Control in West African Urban Contexts},
  author={Research Team},
  year={2025},
  note={Digital Twin Implementation}
}
```

## License

This project is developed for research purposes. See LICENSE file for details.
