# ARZ Traffic Model - GPU-Only Build

**🔥 IMPORTANT: This is a GPU-only version of the ARZ traffic model. 🔥**

This repository contains a high-performance, **GPU-only** implementation of the multi-class ARZ traffic flow simulation. The CPU fallback has been completely removed to maximize performance by eliminating CPU/GPU data transfer overhead.

**CUDA is now a mandatory requirement.** If you do not have a compatible NVIDIA GPU and the CUDA Toolkit installed, this code will not run.

## Requirements

- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA Toolkit 11.x or 12.x
- Python 3.9+
- Numba 0.56+
- CuPy (matching your CUDA version, e.g., `cupy-cuda11x`)
- Pydantic

## Running a Simulation

Simulations are now configured and run directly in Python using Pydantic configuration objects, not YAML files. The main entry point is the `SimulationRunner` class.

**Example Usage:**

```python
# main_network_simulation.py

from arz_model.config import NetworkSimulationConfig # ... and other configs
from arz_model.simulation.runner import SimulationRunner
from arz_model.network.network_grid import NetworkGrid

# 1. Create a detailed simulation configuration
config = NetworkSimulationConfig(
    # ... define your network, physics, time, etc.
)

# 2. Initialize the network grid from the configuration
network_grid = NetworkGrid.from_config(config)

# 3. Initialize the runner
# The runner will automatically detect and use the GPU.
runner = SimulationRunner(
    network_grid=network_grid,
    simulation_config=config,
    quiet=False
)

# 4. Run the simulation
results = runner.run()

# 5. Process results
print("Simulation finished!")
print(f"Final time: {results['final_time']}")

```

## Benchmarking & Profiling

A dedicated script is available to benchmark performance and profile memory usage, ensuring the GPU-only architecture meets its performance targets and is free of memory leaks.

**Usage:**

Run the benchmark script as a module from the project's parent directory (`Code project`):

```bash
cd 'd:\\Projets\\Alibi\\Code project'
python -m arz_model.benchmarks.benchmark_gpu_only
```

The script will:
1.  Run a performance test to measure simulation steps per second.
2.  Run a memory profiling test to check for memory leaks.

The script will automatically skip the tests if a compatible GPU is not found.

## Visualizing Results

The `visualization/plotting.py` and `visualization/network_visualizer.py` modules provide tools to plot simulation outputs.

**Example:**

```python
# Assuming 'results' is the output from runner.run()

from arz_model.visualization.plotting import plot_spacetime_density
from arz_model.visualization.network_visualizer import plot_network

# Plot the network topology
plot_network(runner.network_grid)

# Plot the density for a specific segment
plot_spacetime_density(
    U=results['states']['seg-0'],
    grid=runner.network_grid.segments['seg-0'].grid,
    params=runner.params,
    class_index=0, # 0 for motorways, 1 for cars
    title="Spacetime Density (Motorways) on Segment 0"
)
```

## Code Structure

*   `core/`: Basic physics, parameters.
*   `grid/`: Grid definition (Grid1D).
*   `numerics/`: Numerical methods (Riemann solvers, time integration, CFL, boundary conditions).
*   `simulation/`: Simulation setup (initial conditions, runner).
*   `io/`: Input/Output (loading/saving data, configuration).
*   `visualization/`: Plotting functions.
*   `analysis/`: Functions for analyzing results (e.g., metrics).
*   `tests/`: Unit tests.
*   `config/`: Pydantic configuration system with multi-city support and caching.

## RL Traffic Signal Control

The ARZ model provides runtime API for reinforcement learning-based traffic signal control. This enables RL agents to observe traffic state and dynamically adjust signal timing.

### Quick Example

```python
from arz_model.simulation.runner import SimulationRunner
from arz_model.config import create_victoria_island_config

# Create configuration with traffic signals
config = create_victoria_island_config(
    enriched_path="data/fichier_de_travail_complet_enriched.xlsx"  # OSM data
)

# Initialize simulator
runner = SimulationRunner(network_grid, config, device='gpu')

# RL control: Switch traffic signal phase at runtime
runner.set_boundary_phase(
    segment_id='5902583245->95636900',  # Segment with traffic signal
    phase='green_EW'                     # East-West green phase
)

# Run simulation step
runner.step(dt=1.0)

# Bulk updates for multiple signals (atomic, fail-fast)
runner.set_boundary_phases_bulk({
    '5902583245->95636900': 'green_NS',
    '31674711->36240967': 'green_EW',
    '36240967->95636908': 'green_NS'
})
```

### Runtime Control API

**`set_boundary_phase(segment_id, phase, validate=True)`**
- Change single traffic signal phase at runtime
- `segment_id`: String segment ID (e.g., '5902583245->95636900') or int node ID
- `phase`: Phase name from traffic_signal_phases config (e.g., 'green_NS', 'green_EW')
- `validate`: Enable input validation (disable for performance in RL training)
- Latency: <0.5ms (CPU dict update only, no GPU transfers)

**`set_boundary_phases_bulk(phase_updates, validate=True)`**
- Atomic multi-signal update (all or nothing)
- `phase_updates`: Dict mapping segment_id -> phase_name
- Fail-fast validation: checks all updates before applying any
- Ideal for coordinated RL control of multiple intersections

**`_validate_segment_phase(segment_id, phase)`**
- Validation helper (called automatically when validate=True)
- Checks: segment exists, has traffic_signal BC, phase in config
- Provides helpful error messages with available options

### Integration with Code_RL

The `Code_RL` package provides a complete Gymnasium environment for training RL agents:

```python
from Code_RL.src.config import create_rl_training_config
from Code_RL.src.env import TrafficSignalEnvDirectV2

# Create RL-optimized configuration
config = create_rl_training_config(
    csv_topology_path='data/victoria_island_topology.csv',
    episode_duration=3600.0,  # 1 hour episodes
    decision_interval=15.0     # RL decision every 15s
)

# Initialize environment
env = TrafficSignalEnvDirectV2(simulation_config=config)

# Standard RL training loop
obs, info = env.reset()
for step in range(1000):
    action = agent.select_action(obs)  # 0 = maintain, 1 = switch
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Performance Characteristics

- **Action Latency**: <0.5ms (dict update on CPU)
- **Step Latency**: ~200-600ms (GPU simulation + observation extraction)
- **Episode Throughput**: ~1000+ steps/sec
- **Architecture**: 100-200x faster than HTTP-based coupling
- **Memory**: Direct GPU array access (no serialization overhead)

See `Code_RL/tests/test_rl_api_integration.py` for complete integration examples.