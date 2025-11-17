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