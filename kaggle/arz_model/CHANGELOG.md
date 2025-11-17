# Changelog: GPU-Only Architecture Migration

This document details the major changes introduced during the migration to a pure GPU-only architecture.

## v2.0.0 (GPU-Only) - 2025-11-12

### 💥 Breaking Changes

- **CPU Fallback Removed**: The simulation now requires an NVIDIA GPU with CUDA support (Compute Capability 6.0+). All CPU-based execution paths have been deleted. The application will raise a `RuntimeError` on startup if a compatible GPU is not found.
- **Device Parameter Removed**: The `device` parameter has been removed from all configuration objects and function calls. The GPU is now automatically detected and used.
- **YAML Configuration Deprecated**: Simulations are no longer configured using `.yml` files. All configurations are now managed through Pydantic models directly within Python scripts (e.g., `NetworkSimulationConfig`).
- **167 Functions Deleted**: A massive amount of dead and redundant code has been purged from the codebase. This includes unused utility functions, legacy builders, and CPU-specific duplicates. See `.copilot-tracking/deleted_functions.txt` for a complete list.
- **13 CPU/GPU Function Pairs Unified**: All functions with separate `_cpu` and `_gpu` implementations have been unified into single, GPU-only functions.

### ✨ New Features

- **Persistent GPU Memory Pool**: A new `GPUMemoryPool` class (`numerics/gpu/memory_pool.py`) now manages all core simulation arrays. This pre-allocates memory on the GPU at initialization, completely eliminating runtime memory allocation and transfer overhead during the simulation loop.
- **GPU-Native Network Coupling**: The performance-critical network node solver has been rewritten as a native CUDA kernel. This eliminates the costly GPU-CPU-GPU round trip that was previously required at every time step for network coupling, keeping all data on the GPU.
- **Asynchronous Checkpoint System**: The `StateManager` has been refactored to be GPU-native. It now supports periodic, asynchronous checkpointing to save simulation progress to the CPU without blocking the main simulation loop.
- **5-10x Performance Improvement**: By eliminating all data transfers during the simulation loop, the new architecture is designed to be 5 to 10 times faster than the previous hybrid model.

### 🚀 Migration Guide

Migrating from the hybrid version to the new GPU-only version requires a few adjustments to your simulation scripts.

#### Before (Hybrid, YAML-based)

```python
# Old way: Running from command line with YAML
# > python -m code.main_simulation --scenario config/my_scenario.yml --device gpu
```

#### After (GPU-Only, Pydantic-based)

```python
# New way: Running directly in Python

from arz_model.config import NetworkSimulationConfig # ... and other configs
from arz_model.simulation.runner import SimulationRunner
from arz_model.network.network_grid import NetworkGrid

# 1. Define your configuration in Python using Pydantic models
config = NetworkSimulationConfig(
    time=TimeConfig(t_final=3600.0),
    physics=PhysicsConfig(v_max_c_kmh=120.0),
    # ... other configuration for segments, nodes, etc.
)

# 2. Initialize the network grid and runner
network_grid = NetworkGrid.from_config(config)
runner = SimulationRunner(
    network_grid=network_grid,
    simulation_config=config
)

# 3. Run the simulation (GPU is used automatically)
results = runner.run()

print("Simulation complete!")
```
