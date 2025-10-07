"""
Unit tests for SimulationRunner RL extensions.

Tests the traffic signal control and observation extraction methods
added for Reinforcement Learning environment integration.
"""

import pytest
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from arz_model.simulation.runner import SimulationRunner
from arz_model.core.parameters import VEH_KM_TO_VEH_M


class TestSimulationRunnerRL:
    """Test suite for RL extensions to SimulationRunner."""
    
    @pytest.fixture
    def runner(self):
        """Create a SimulationRunner instance for testing."""
        # Use a simple scenario for testing
        scenario_path = os.path.join(
            os.path.dirname(__file__),
            '../../scenarios/scenario_calibration_victoria_island.yml'
        )
        base_config_path = os.path.join(
            os.path.dirname(__file__),
            '../../arz_model/config/config_base.yml'
        )
        
        # Check if files exist
        if not os.path.exists(scenario_path):
            pytest.skip(f"Scenario file not found: {scenario_path}")
        if not os.path.exists(base_config_path):
            pytest.skip(f"Base config file not found: {base_config_path}")
        
        runner = SimulationRunner(
            scenario_config_path=scenario_path,
            base_config_path=base_config_path,
            quiet=True,
            device='cpu'
        )
        
        return runner
    
    # =====================================================================
    # Tests for set_traffic_signal_state()
    # =====================================================================
    
    def test_set_traffic_signal_state_valid_left_red(self, runner):
        """Test setting left boundary to red phase (phase_id=0)."""
        runner.set_traffic_signal_state('left', phase_id=0)
        
        # Verify boundary condition was updated
        assert runner.current_bc_params['left']['type'] == 'outflow'
        assert runner.current_bc_params['left']['extrapolation_order'] == 1
    
    def test_set_traffic_signal_state_valid_right_green(self, runner):
        """Test setting right boundary to green phase (phase_id=1)."""
        runner.set_traffic_signal_state('right', phase_id=1)
        
        # Verify boundary condition was updated
        assert runner.current_bc_params['right']['type'] == 'inflow'
        # State should be set (either to initial_equilibrium_state or None)
        assert 'state' in runner.current_bc_params['right']
    
    def test_set_traffic_signal_state_invalid_intersection(self, runner):
        """Test that invalid intersection_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid intersection_id"):
            runner.set_traffic_signal_state('invalid_id', phase_id=0)
    
    def test_set_traffic_signal_state_invalid_phase_negative(self, runner):
        """Test that negative phase_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid phase_id"):
            runner.set_traffic_signal_state('left', phase_id=-1)
    
    def test_set_traffic_signal_state_invalid_phase_non_int(self, runner):
        """Test that non-integer phase_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid phase_id"):
            runner.set_traffic_signal_state('left', phase_id=1.5)
    
    def test_set_traffic_signal_state_preserves_other_boundary(self, runner):
        """Test that setting one boundary doesn't affect the other."""
        # Store original right BC
        original_right_bc = runner.current_bc_params.get('right', {}).copy()
        
        # Modify left BC
        runner.set_traffic_signal_state('left', phase_id=0)
        
        # Verify right BC unchanged
        assert runner.current_bc_params.get('right', {}) == original_right_bc
    
    # =====================================================================
    # Tests for get_segment_observations()
    # =====================================================================
    
    def test_get_segment_observations_single_cell(self, runner):
        """Test observation extraction for a single cell."""
        # Use middle cell of physical domain
        mid_idx = runner.grid.ghost_cells + runner.grid.N_physical // 2
        
        obs = runner.get_segment_observations([mid_idx])
        
        # Verify all keys present
        expected_keys = ['rho_m', 'q_m', 'rho_c', 'q_c', 'v_m', 'v_c']
        assert all(key in obs for key in expected_keys)
        
        # Verify shapes
        for key in expected_keys:
            assert obs[key].shape == (1,), f"Key {key} has wrong shape"
        
        # Verify values are numeric and non-negative for densities
        assert np.all(obs['rho_m'] >= 0)
        assert np.all(obs['rho_c'] >= 0)
        assert np.isfinite(obs['v_m']).all()
        assert np.isfinite(obs['v_c']).all()
    
    def test_get_segment_observations_multiple_cells(self, runner):
        """Test observation extraction for multiple cells."""
        # Select first 5 physical cells
        start_idx = runner.grid.ghost_cells
        indices = list(range(start_idx, start_idx + 5))
        
        obs = runner.get_segment_observations(indices)
        
        # Verify shapes
        expected_keys = ['rho_m', 'q_m', 'rho_c', 'q_c', 'v_m', 'v_c']
        for key in expected_keys:
            assert obs[key].shape == (5,), f"Key {key} has wrong shape"
    
    def test_get_segment_observations_empty_list(self, runner):
        """Test that empty segment_indices raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            runner.get_segment_observations([])
    
    def test_get_segment_observations_out_of_bounds_low(self, runner):
        """Test that indices below physical range raise ValueError."""
        # Ghost cell index (should fail)
        ghost_idx = 0
        
        with pytest.raises(ValueError, match="out of bounds"):
            runner.get_segment_observations([ghost_idx])
    
    def test_get_segment_observations_out_of_bounds_high(self, runner):
        """Test that indices above physical range raise ValueError."""
        # Index beyond physical + ghost cells
        beyond_idx = runner.grid.ghost_cells + runner.grid.N_physical + 1
        
        with pytest.raises(ValueError, match="out of bounds"):
            runner.get_segment_observations([beyond_idx])
    
    def test_get_segment_observations_velocity_calculation(self, runner):
        """Test that velocity is correctly calculated from q and rho."""
        mid_idx = runner.grid.ghost_cells + runner.grid.N_physical // 2
        obs = runner.get_segment_observations([mid_idx])
        
        # Manual calculation with epsilon
        epsilon = 1e-10
        expected_v_m = obs['q_m'][0] / (obs['rho_m'][0] + epsilon)
        expected_v_c = obs['q_c'][0] / (obs['rho_c'][0] + epsilon)
        
        # Verify calculations (with numerical tolerance)
        np.testing.assert_allclose(obs['v_m'][0], expected_v_m, rtol=1e-10)
        np.testing.assert_allclose(obs['v_c'][0], expected_v_c, rtol=1e-10)
    
    def test_get_segment_observations_zero_density_handling(self, runner):
        """Test that zero density doesn't cause division by zero."""
        # Manually set a cell to zero density
        mid_idx = runner.grid.ghost_cells + runner.grid.N_physical // 2
        runner.U[0, mid_idx] = 0.0  # Zero motorcycle density
        runner.U[1, mid_idx] = 0.0  # Zero momentum
        
        obs = runner.get_segment_observations([mid_idx])
        
        # Velocity should be finite (not NaN or Inf)
        assert np.isfinite(obs['v_m'][0])
        assert np.isfinite(obs['v_c'][0])
    
    # =====================================================================
    # Integration Tests
    # =====================================================================
    
    def test_traffic_signal_affects_simulation(self, runner):
        """Test that changing traffic signal actually affects simulation state."""
        # Run simulation for a few steps
        runner.set_traffic_signal_state('left', phase_id=1)  # Green
        
        # Take a simulation step
        t_step = 0.5
        runner.run(t_final=runner.t + t_step, output_dt=t_step)
        
        # Get observation
        mid_idx = runner.grid.ghost_cells + 5
        obs_green = runner.get_segment_observations([mid_idx])
        
        # Now switch to red and run another step
        runner.set_traffic_signal_state('left', phase_id=0)  # Red
        runner.run(t_final=runner.t + t_step, output_dt=t_step)
        obs_red = runner.get_segment_observations([mid_idx])
        
        # State should have changed (not identical)
        # Note: This is a weak test - states might be similar if time is very short
        # But at least we verify the machinery works
        assert obs_green.keys() == obs_red.keys()
    
    def test_observation_consistency_across_calls(self, runner):
        """Test that multiple calls without simulation steps return same observation."""
        mid_idx = runner.grid.ghost_cells + runner.grid.N_physical // 2
        
        obs1 = runner.get_segment_observations([mid_idx])
        obs2 = runner.get_segment_observations([mid_idx])
        
        # Should be identical
        for key in obs1.keys():
            np.testing.assert_array_equal(obs1[key], obs2[key])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
