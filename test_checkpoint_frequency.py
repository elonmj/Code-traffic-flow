"""Test checkpoint frequency with time-based logic."""
import sys
sys.path.insert(0, 'arz_model')

from arz_model.config.config_factory import create_victoria_island_config
from arz_model.simulation.runner import SimulationRunner
from arz_model.network.network_grid import NetworkGrid
import pickle

# Short simulation: 30 seconds, checkpoint every 5 seconds = 6 expected checkpoints
config = create_victoria_island_config(
    t_final=30.0,
    output_dt=5.0,
    default_density=20.0,
    inflow_density=30.0
)

# Build network from config
network_grid = NetworkGrid.from_config(config)

# Run simulation
runner = SimulationRunner(simulation_config=config, network_grid=network_grid, quiet=False)
results = runner.run()

# Validate checkpoint count
checkpoint_count = len(results['history']['time'])
print(f"\n{'='*60}")
print(f"CHECKPOINT VALIDATION")
print(f"{'='*60}")
print(f"Expected checkpoints: ~6 (t=0, 5, 10, 15, 20, 25, 30)")
print(f"Actual checkpoints: {checkpoint_count}")
print(f"Times: {results['history']['time']}")

assert checkpoint_count >= 6, f"Expected at least 6 checkpoints, got {checkpoint_count}"
print(f"✅ TEST PASSED - Sufficient checkpoints for animation")

# Save for animation testing
with open('network_simulation_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print(f"✅ Results saved to network_simulation_results.pkl")
