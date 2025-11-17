#!/usr/bin/env python3
"""
Test: Checkpoint Frequency Validation (Phase 1)

Validates that time-based checkpointing generates regular checkpoints at output_dt intervals,
fixing the animation blocker where step-based logic created zero intermediate saves.

Expected behavior:
- 30s simulation with output_dt=5.0s should generate 6+ checkpoints
- Checkpoints should appear at approximately: t=0, 5, 10, 15, 20, 25, 30
- Final state must always be saved even if not on exact interval

This test requires GPU/CUDA environment - use kaggle_runner to execute:
    python kaggle_runner/executor.py --target arz_model/tests/test_checkpoint_frequency.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from arz_model.config.config_factory import create_victoria_island_config
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner


def test_checkpoint_frequency():
    """Test that time-based checkpointing generates sufficient checkpoints for animation."""
    
    print("=" * 80)
    print("PHASE 1 TEST: Checkpoint Frequency Validation")
    print("=" * 80)
    
    # Create short test simulation (30s with 5s output interval)
    print("\n[1/4] Creating test configuration...")
    config = create_victoria_island_config(
        t_final=30.0,
        output_dt=5.0,  # Should generate checkpoints at t=0, 5, 10, 15, 20, 25, 30
        default_density=20.0,
        inflow_density=30.0
    )
    print(f"✓ Config created: t_final={config.time.t_final}s, output_dt={config.time.output_dt}s")
    
    # Build network grid
    print("\n[2/4] Building network grid...")
    network_grid = NetworkGrid.from_config(config)
    print(f"✓ Network built: {len(network_grid.segments)} segments, {len(network_grid.nodes)} nodes")
    
    # Run simulation
    print("\n[3/4] Running simulation...")
    runner = SimulationRunner(
        simulation_config=config,
        network_grid=network_grid,
        quiet=False
    )
    results = runner.run()
    print(f"✓ Simulation completed: final_time={results['final_time']:.2f}s")
    
    # Validate checkpoint frequency
    print("\n[4/4] Validating checkpoints...")
    checkpoint_count = len(results['states'])
    checkpoint_times = [state['time'] for state in results['states']]
    
    print(f"  Total checkpoints: {checkpoint_count}")
    print(f"  Checkpoint times: {[f'{t:.1f}s' for t in checkpoint_times]}")
    
    # Expected: at least 6 checkpoints (t=0, 5, 10, 15, 20, 25, 30)
    expected_min = 6
    
    print("\n" + "=" * 80)
    if checkpoint_count >= expected_min:
        print("✅ TEST PASSED - Sufficient checkpoints for animation")
        print(f"   Expected: >={expected_min} checkpoints")
        print(f"   Actual: {checkpoint_count} checkpoints")
        print(f"   Intervals: ~{config.time.output_dt}s")
        print("=" * 80)
        return True
    else:
        print("❌ TEST FAILED - Insufficient checkpoints for animation")
        print(f"   Expected: >={expected_min} checkpoints")
        print(f"   Actual: {checkpoint_count} checkpoints")
        print(f"   This would result in single-frame animation (BLOCKER)")
        print("=" * 80)
        return False


if __name__ == '__main__':
    try:
        success = test_checkpoint_frequency()
        sys.exit(0 if success else 1)
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST ERROR")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
