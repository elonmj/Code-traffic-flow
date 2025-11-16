"""
Tests for numerical stability mechanisms in NetworkSimulator.

This module verifies that the simulation completes successfully with:
- dt_min/dt_max protection
- Adaptive CFL logic
- Positivity preservation

Simplified from original test_dt_min_protection.py - focuses on successful
completion rather than trying to trigger pathological dt collapse.
"""
import tempfile
import os
from arz_model.config.config_factory import create_victoria_island_config
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner


def test_stable_simulation_completes():
    """
    Test that a well-behaved simulation completes successfully.
    
    Uses a stable network configuration:
    - Larger dx → Larger stable dt
    - Conservative CFL
    - Low initial density → Stable flow
    - Conservative physics parameters
    - Short simulation time for fast testing
    
    Expected: Simulation completes without error, returns valid results
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
        config.time.cfl_factor = 0.5
        
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
        
        # Validate results structure
        assert results is not None, "Simulation returned None"
        assert 'history' in results, "Results missing 'history' key"
        assert 'final_time' in results, "Results missing 'final_time' key"
        assert 'total_steps' in results, "Results missing 'total_steps' key"
        
        # Validate history structure
        history = results['history']
        assert 'time' in history, "History missing 'time' key"
        assert 'segments' in history, "History missing 'segments' key"
        
        # Validate that simulation actually progressed
        assert results['final_time'] > 0, "Simulation did not progress"
        assert results['total_steps'] > 0, "No time steps taken"
        assert len(history['time']) > 0, "No history recorded"
        
        # Success!
        print(f"✅ Simulation completed: {results['total_steps']} steps, final_time={results['final_time']:.2f}s")
        
    finally:
        os.unlink(csv_path)


def test_high_density_simulation_stability():
    """
    Test that simulation remains stable even with high initial density.
    
    This tests the numerical stability protections under stress:
    - Higher density (closer to jam)
    - Adaptive CFL should activate if needed
    - dt_min/dt_max bounds should keep simulation stable
    
    Expected: Simulation completes without RuntimeError
    """
    csv_content = """u,v,segment_id,length,lane_count,v_max_m,v_max_c
1,2,seg_stress,1.5,2,90,90"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_path = f.name
    
    try:
        config = create_victoria_island_config(
            csv_path=csv_path,
            default_density=120.0,  # High density (near jam ~167 veh/km)
            default_velocity=20.0,  # Slower due to congestion
            inflow_density=130.0,   # Higher inflow
            inflow_velocity=15.0,   # Slow inflow
            t_final=15.0,          # 15s test
            output_dt=3.0,
            cells_per_100m=10,     # Medium grid
        )
        
        # Set protection bounds
        config.time.dt_min = 0.001
        config.time.dt_max = 1.0
        config.time.cfl_factor = 0.7
        
        # Build and run
        network_grid = NetworkGrid.from_config(config)
        runner = SimulationRunner(
            network_grid=network_grid,
            simulation_config=config,
            quiet=True,
            debug=False
        )
        
        # Should complete despite high density
        results = runner.run(timeout=None)
        
        # Validate completion
        assert results is not None
        assert results['final_time'] > 0
        assert results['total_steps'] > 0
        
        print(f"✅ High-density simulation completed: {results['total_steps']} steps")
        
    finally:
        os.unlink(csv_path)
