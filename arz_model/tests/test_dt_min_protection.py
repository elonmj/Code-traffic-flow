"""
Unit tests for dt_min protection mechanism.

Tests that the simulation correctly halts when dt falls below dt_min threshold,
preventing infinite loops in numerically unstable scenarios.

References:
    - #file:../.copilot-tracking/changes/20251116-numerical-stability-plan.md
"""
import pytest
import numpy as np
import tempfile
import os

# Import the REAL workflow components
from arz_model.config.config_factory import create_victoria_island_config
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner


def test_dt_min_protection_triggers():
    """
    Test that simulation raises RuntimeError when dt < dt_min.
    
    Uses a pathological network configuration that forces dt collapse:
    - Very small dx → Small stable dt
    - Low dt_min threshold to allow initial steps
    - Aggressive physics to force instability
    - Short t_final to avoid long test times
    
    Expected: RuntimeError with message about dt_min violation
    """
    # Create a minimal CSV for a single segment that will collapse
    csv_content = """u,v,segment_id,length,lane_count,v_max_m,v_max_c
1,2,seg_test,0.5,2,80,80"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_path = f.name
    
    try:
        # Create config using REAL factory with pathological parameters
        config = create_victoria_island_config(
            csv_path=csv_path,
            default_density=150.0,  # Very high density → near jam
            default_velocity=10.0,  # Low velocity
            inflow_density=160.0,   # Even higher inflow
            inflow_velocity=5.0,    # Very slow inflow
            t_final=5.0,           # Short test (5s as requested)
            output_dt=1.0,
            cells_per_100m=50,     # Very fine grid → small dx
        )
        
        # Override time config to set aggressive dt_min protection
        config.time.dt_min = 0.0001  # Very strict - will trigger
        config.time.dt_max = 1.0
        config.time.dt_collapse_threshold = 0.01
        
        # Make physics more aggressive to force collapse
        config.physics.k_m = 3.0
        config.physics.gamma_m = 6.0
        config.physics.k_c = 3.0
        config.physics.gamma_c = 6.0
        
        # Build network using REAL workflow
        network_grid = NetworkGrid.from_config(config)
        
        # Initialize runner using REAL workflow
        runner = SimulationRunner(
            network_grid=network_grid,
            simulation_config=config,
            quiet=True,
            debug=False
        )
        
        # Run simulation and expect RuntimeError about dt_min
        with pytest.raises(RuntimeError, match="dt.*dt_min"):
            runner.run(timeout=None)
            
    finally:
        # Cleanup temp file
        if os.path.exists(csv_path):
            os.remove(csv_path)


def test_dt_min_protection_allows_stable_runs():
    """
    Test that simulation completes successfully when dt stays above dt_min.
    
    Uses a well-behaved network configuration:
    - Larger dx → Larger stable dt
    - Conservative CFL
    - Low initial density → Stable flow
    - Conservative physics parameters
    
    Expected: Simulation completes without error
    """
    # Create a minimal CSV for a stable single segment
    csv_content = """u,v,segment_id,length,lane_count,v_max_m,v_max_c
1,2,seg_stable,2.0,3,100,100"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_path = f.name
    
    try:
        # Create config using REAL factory with stable parameters
        config = create_victoria_island_config(
            csv_path=csv_path,
            default_density=20.0,   # Low density
            default_velocity=60.0,  # Moderate velocity
            inflow_density=25.0,    # Light inflow
            inflow_velocity=55.0,   # Stable inflow
            t_final=12.0,          # Short test (12s as requested)
            output_dt=2.0,
            cells_per_100m=4,      # Coarse grid → large dx
        )
        
        # Conservative time config
        config.time.dt_min = 0.001
        config.time.dt_max = 1.0
        
        # Conservative CFL
        config.cfl_number = 0.5
        
        # Build network using REAL workflow
        network_grid = NetworkGrid.from_config(config)
        
        # Initialize runner using REAL workflow
        runner = SimulationRunner(
            network_grid=network_grid,
            simulation_config=config,
            quiet=True,
            debug=False
        )
        
        # Should complete without raising
        results = runner.run(timeout=None)
        
        # Validate results
        assert results is not None
        assert "t_values" in results
        assert len(results["t_values"]) > 0
        assert results["t_values"][-1] >= 12.0  # Reached t_final
        
    finally:
        # Cleanup temp file
        if os.path.exists(csv_path):
            os.remove(csv_path)
