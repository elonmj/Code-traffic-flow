"""
Integration tests for the GPU-only simulation architecture.

These tests validate the end-to-end functionality of the simulation,
ensuring that the GPU-only workflow is correct, efficient, and robust.
"""
import pytest
import os
import sys
import numpy as np
from numba import cuda

# Add project root to path to allow module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arz_model.config import (
    NetworkSimulationConfig, TimeConfig, PhysicsConfig, GridConfig,
    SegmentConfig, NodeConfig, ICConfig, UniformIC,
    BoundaryConditionsConfig, InflowBC, OutflowBC, ReflectiveBC
)
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner
from arz_model.core.parameters import ModelParameters

# Helper function to create a standard test configuration
def create_test_config() -> NetworkSimulationConfig:
    """Creates a simple, valid NetworkSimulationConfig for testing."""
    return NetworkSimulationConfig(
        time=TimeConfig(t_final=0.01, output_dt=0.01), # Very short simulation for speed
        physics=PhysicsConfig(
            alpha=0.6,
            v_max_c_kmh=120.0,
            v_max_m_kmh=100.0,
            tau_c=1.5,
            tau_m=1.0,
            k_c=10.0,
            k_m=5.0,
            gamma_c=2.0,
            gamma_m=2.0,
            rho_max=200.0 / 1000.0,
            v_creeping_kmh=10.0,
            epsilon=1e-6
        ),
        grid=GridConfig(num_ghost_cells=3),
        segments=[
            SegmentConfig(
                id="seg-1",
                x_min=0.0,
                x_max=1000.0,
                N=100,
                initial_conditions=ICConfig(config=UniformIC(density=50.0, velocity=60.0)),
                boundary_conditions=BoundaryConditionsConfig(
                    left=InflowBC(density=50.0, velocity=60.0),
                    right=OutflowBC(density=0.0, velocity=0.0)
                ),
                start_node="node-1",
                end_node="node-2"
            ),
            SegmentConfig(
                id="seg-2",
                x_min=0.0,
                x_max=500.0,
                N=50,
                initial_conditions=ICConfig(config=UniformIC(density=20.0, velocity=80.0)),
                boundary_conditions=BoundaryConditionsConfig(
                    left=InflowBC(density=20.0, velocity=80.0),
                    right=OutflowBC(density=0.0, velocity=0.0)
                ),
                start_node="node-2",
                end_node="node-3"
            )
        ],
        nodes=[
            NodeConfig(id="node-1", type="boundary", incoming_segments=[], outgoing_segments=["seg-1"]),
            NodeConfig(id="node-2", type="junction", incoming_segments=["seg-1"], outgoing_segments=["seg-2"]),
            NodeConfig(id="node-3", type="boundary", incoming_segments=["seg-2"], outgoing_segments=[]),
        ]
    )

@pytest.mark.skipif(not cuda.is_available(), reason="GPU not available")
def test_simulation_runs_end_to_end_on_gpu():
    """
    Tests that a simple simulation can run from start to finish on the GPU.
    """
    print("Running test: test_simulation_runs_end_to_end_on_gpu")
    try:
        config = create_test_config()
        network_grid = NetworkGrid.from_config(config)
        runner = SimulationRunner(network_grid=network_grid, simulation_config=config, quiet=True)
        results = runner.run()

        # Verify results structure
        assert "final_time" in results
        assert "total_steps" in results
        assert "final_states" in results
        assert "seg-1" in results["final_states"]
        assert "seg-2" in results["final_states"]

        # Verify state array shape
        final_state_seg1 = results["final_states"]["seg-1"]
        expected_shape = (4, config.segments[0].N + 2 * config.grid.num_ghost_cells)
        assert final_state_seg1.shape == expected_shape, f"Expected shape {expected_shape}, but got {final_state_seg1.shape}"
        
        print("✅ Test passed. Simulation ran end-to-end and produced valid results.")
    except Exception as e:
        pytest.fail(f"End-to-end GPU simulation test failed with an exception: {e}", pytrace=True)


# This test is designed to be run in an environment *without* a GPU
@pytest.mark.skipif(cuda.is_available(), reason="This test is for CPU-only environments")
def test_gpu_required_error():
    """
    Verifies that the SimulationRunner raises a RuntimeError if CUDA is not available.
    """
    print("Running test: test_gpu_required_error")
    with pytest.raises(RuntimeError, match="CUDA not available"):
        config = create_test_config()
        network_grid = NetworkGrid.from_config(config)
        SimulationRunner(network_grid=network_grid, simulation_config=config, quiet=True)
    print("✅ Test passed.")


@pytest.mark.skipif(not cuda.is_available(), reason="GPU not available")
def test_no_cpu_transfers_in_loop():
    """
    Hooks into CUDA transfer functions to verify that no transfers occur
    during the main simulation loop.
    """
    print("Running test: test_no_cpu_transfers_in_loop")
    
    transfer_log = []
    original_to_device = cuda.to_device
    original_copy_to_host = cuda.devicearray.DeviceNDArray.copy_to_host

    def tracked_to_device(obj, *args, **kwargs):
        transfer_log.append(f"to_device: {type(obj)}")
        return original_to_device(obj, *args, **kwargs)

    def tracked_copy_to_host(self, *args, **kwargs):
        transfer_log.append(f"copy_to_host: shape={self.shape}")
        return original_copy_to_host(self, *args, **kwargs)

    try:
        config = create_test_config()
        
        cuda.to_device = tracked_to_device
        cuda.devicearray.DeviceNDArray.copy_to_host = tracked_copy_to_host
        
        network_grid = NetworkGrid.from_config(config)
        runner = SimulationRunner(network_grid=network_grid, simulation_config=config, quiet=True)
        
        # Test 1: Check transfers during the main `run()` method
        transfer_log.clear()
        runner.run()
        
        # The only transfers allowed are the final ones in `get_final_results()`
        # One for each segment
        assert len(transfer_log) == 2, f"Expected 2 transfers for final results, but found {len(transfer_log)}: {transfer_log}"
        assert "copy_to_host" in transfer_log[0]
        assert "copy_to_host" in transfer_log[1]
        print("✅ Test passed: `runner.run()` only performed final transfers.")

        # Test 2: Check transfers during a single `step()`
        transfer_log.clear()
        runner.network_simulator.step()
        
        assert len(transfer_log) == 0, f"Unexpected GPU-CPU transfers during a single step: {transfer_log}"
        print("✅ Test passed: `network_simulator.step()` performed zero transfers.")

    finally:
        cuda.to_device = original_to_device
        cuda.devicearray.DeviceNDArray.copy_to_host = original_copy_to_host

@pytest.mark.skipif(not cuda.is_available(), reason="GPU not available")
def test_mass_conservation_gpu():
    """
    Verifies that the total mass (rho) in the system is conserved on the GPU
    when using reflective boundary conditions.
    """
    print("Running test: test_mass_conservation_gpu")
    
    config = create_test_config()
    # Use reflective "wall" boundary conditions to ensure mass is conserved
    config.segments[0].boundary_conditions.left = ReflectiveBC()
    config.segments[1].boundary_conditions.right = ReflectiveBC()
    
    network_grid = NetworkGrid.from_config(config)
    
    # Calculate initial mass from the config
    initial_mass = 0.0
    for seg_config in config.segments:
        segment = network_grid.segments[seg_config.id]
        ic_config = seg_config.initial_conditions.config
        if isinstance(ic_config, UniformIC):
            # Mass = density * length
            initial_mass += ic_config.density * (segment.grid.x_max - segment.grid.x_min)

    runner = SimulationRunner(network_grid=network_grid, simulation_config=config, quiet=True)
    results = runner.run()
    
    # Calculate final mass from the results
    final_mass = 0.0
    final_states = results["final_states"]
    for seg_id, state_array in final_states.items():
        segment = network_grid.segments[seg_id]
        # state_array is on CPU now. Shape: (4, N_total)
        # We only consider the physical cells for mass calculation.
        physical_cells_rho = state_array[0, segment.grid.physical_cell_indices]
        final_mass += np.sum(physical_cells_rho) * segment.grid.dx
        
    # Allow for small numerical precision errors
    assert np.isclose(initial_mass, final_mass, rtol=1e-5), f"Mass not conserved! Initial: {initial_mass}, Final: {final_mass}"
    print(f"✅ Test passed. Mass conserved (Initial: {initial_mass:.5f}, Final: {final_mass:.5f}).")

