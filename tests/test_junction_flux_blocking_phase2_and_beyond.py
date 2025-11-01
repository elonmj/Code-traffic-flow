"""
Junction Flux Blocking Tests - Phase 2 and Beyond

Tests comprehensive junction flux blocking functionality including:
- BOTH spatial schemes (first_order and weno5)
- Various red_light_factor values
- RED vs GREEN signal behavior
- Different junction positions

Phase 2 continuation of test_junction_flux_blocking_phase1.py
"""

import pytest
import numpy as np
from arz_model.config.builders import RLNetworkConfigBuilder
from arz_model.network.network_grid import NetworkGrid
from arz_model.junction.node import Node
from arz_model.junction.traffic_light import TrafficLightController


class TestJunctionFluxBlockingComprehensive:
    """Comprehensive tests for junction flux blocking in production scenarios."""
    
    @pytest.fixture
    def base_config(self):
        """Base configuration for testing."""
        config = RLNetworkConfigBuilder.simple_corridor(segments=2)
        return config
    
    def create_network_from_config(
        self, 
        base_config,
        red_light_factor=0.05,
        spatial_scheme='first_order'
    ):
        """
        Create network grid from configuration.
        
        Args:
            base_config: Base network configuration
            red_light_factor: Flux reduction during RED (default 0.05 = 95% blocking)
            spatial_scheme: 'first_order' or 'weno5'
        
        Returns:
            NetworkGrid instance
        """
        # Modify configuration
        config = base_config.copy()
        config.physics.red_light_factor = red_light_factor
        config.physics.spatial_scheme = spatial_scheme
        
        # Create network
        from arz_model.network.network_simulator import NetworkGridSimulator
        simulator = NetworkGridSimulator(config)
        network = simulator.network
        
        return network
    
    # ==================== Test Both Spatial Schemes ====================
    
    def test_junction_blocking_first_order_scheme(self, base_config):
        """Test junction blocking with first_order spatial scheme (PRODUCTION DEFAULT)."""
        network = self.create_network_from_config(
            base_config, 
            red_light_factor=0.05,
            spatial_scheme='first_order'
        )
        
        # Simulate during RED signal (t=0 is RED for seg_0)
        dt = 0.5
        t_end = 30.0  # 30 seconds of RED
        current_time = 0.0
        
        initial_density = network.segments['seg_0']['U'][0, 2].copy()
        
        while current_time < t_end:
            network.step(dt, current_time)
            current_time += dt
        
        final_density = network.segments['seg_0']['U'][0, 2]
        
        # During RED, density should increase (congestion forming)
        assert final_density > initial_density, \
            f"Density should increase during RED (first_order). Initial: {initial_density:.4f}, Final: {final_density:.4f}"
    
    def test_junction_blocking_weno5_scheme(self, base_params):
        """Test junction blocking with weno5 spatial scheme (PHASE 1 TESTS)."""
        network = self.create_2segment_network_with_light(
            base_params, 
            red_light_factor=0.05,
            spatial_scheme='weno5'
        )
        
        # Simulate during RED signal
        dt = 0.5
        t_end = 30.0
        current_time = 0.0
        
        initial_density = network.segments['seg_0']['U'][0, 2].copy()
        
        while current_time < t_end:
            network.step(dt, current_time)
            current_time += dt
        
        final_density = network.segments['seg_0']['U'][0, 2]
        
        # During RED, density should increase
        assert final_density > initial_density, \
            f"Density should increase during RED (weno5). Initial: {initial_density:.4f}, Final: {final_density:.4f}"
    
    def test_both_schemes_produce_similar_blocking(self, base_params):
        """Verify both spatial schemes produce comparable junction blocking behavior."""
        # First order scheme
        network_first = self.create_2segment_network_with_light(
            base_params.copy(), 
            red_light_factor=0.05,
            spatial_scheme='first_order'
        )
        
        # WENO5 scheme
        network_weno = self.create_2segment_network_with_light(
            base_params.copy(), 
            red_light_factor=0.05,
            spatial_scheme='weno5'
        )
        
        # Simulate both
        dt = 0.5
        t_end = 20.0
        current_time = 0.0
        
        initial_first = network_first.segments['seg_0']['U'][0, 2].copy()
        initial_weno = network_weno.segments['seg_0']['U'][0, 2].copy()
        
        while current_time < t_end:
            network_first.step(dt, current_time)
            network_weno.step(dt, current_time)
            current_time += dt
        
        final_first = network_first.segments['seg_0']['U'][0, 2]
        final_weno = network_weno.segments['seg_0']['U'][0, 2]
        
        # Both should show density increase during RED
        density_increase_first = final_first - initial_first
        density_increase_weno = final_weno - initial_weno
        
        assert density_increase_first > 0, "first_order should show density increase"
        assert density_increase_weno > 0, "weno5 should show density increase"
        
        # Both should be within same order of magnitude
        # (Not exact due to different numerical schemes, but qualitatively similar)
        ratio = density_increase_first / density_increase_weno
        assert 0.1 < ratio < 10.0, \
            f"Both schemes should show similar blocking behavior. Ratio: {ratio:.2f}"
    
    # ==================== Test Various red_light_factor Values ====================
    
    def test_flux_reduction_factor_05_percent(self, base_params):
        """Test 95% flux blocking (red_light_factor=0.05) - PRODUCTION DEFAULT."""
        network = self.create_2segment_network_with_light(
            base_params, 
            red_light_factor=0.05,
            spatial_scheme='first_order'
        )
        
        # Measure flux at junction during RED
        dt = 0.5
        t_end = 20.0
        current_time = 0.0
        
        initial_mass = network.segments['seg_0']['U'][0, :].sum()
        
        while current_time < t_end:
            network.step(dt, current_time)
            current_time += dt
        
        final_mass = network.segments['seg_0']['U'][0, :].sum()
        
        # Mass should accumulate (increase) during RED blocking
        assert final_mass > initial_mass * 1.02, \
            f"Mass should accumulate with 95% blocking. Initial: {initial_mass:.2f}, Final: {final_mass:.2f}"
    
    def test_flux_reduction_factor_10_percent(self, base_params):
        """Test 90% flux blocking (red_light_factor=0.10) - Moderate blocking."""
        network = self.create_2segment_network_with_light(
            base_params, 
            red_light_factor=0.10,
            spatial_scheme='first_order'
        )
        
        dt = 0.5
        t_end = 20.0
        current_time = 0.0
        
        initial_density = network.segments['seg_0']['U'][0, 2].copy()
        
        while current_time < t_end:
            network.step(dt, current_time)
            current_time += dt
        
        final_density = network.segments['seg_0']['U'][0, 2]
        
        # Should still show congestion but less severe than 0.05
        assert final_density > initial_density, \
            f"90% blocking should still cause congestion. Initial: {initial_density:.4f}, Final: {final_density:.4f}"
    
    def test_flux_reduction_factor_50_percent(self, base_params):
        """Test 50% flux blocking (red_light_factor=0.50) - Weak blocking."""
        network = self.create_2segment_network_with_light(
            base_params, 
            red_light_factor=0.50,
            spatial_scheme='first_order'
        )
        
        dt = 0.5
        t_end = 20.0
        current_time = 0.0
        
        initial_density = network.segments['seg_0']['U'][0, 2].copy()
        
        while current_time < t_end:
            network.step(dt, current_time)
            current_time += dt
        
        final_density = network.segments['seg_0']['U'][0, 2]
        
        # 50% blocking should still show some congestion
        density_change = final_density - initial_density
        assert density_change >= -0.01, \
            f"50% blocking should not cause drainage. Change: {density_change:.4f}"
    
    def test_flux_reduction_factor_100_percent(self, base_params):
        """Test NO flux blocking (red_light_factor=1.0) - GREEN signal behavior."""
        network = self.create_2segment_network_with_light(
            base_params, 
            red_light_factor=1.0,
            spatial_scheme='first_order'
        )
        
        dt = 0.5
        t_end = 20.0
        current_time = 0.0
        
        initial_density = network.segments['seg_0']['U'][0, 2].copy()
        
        while current_time < t_end:
            network.step(dt, current_time)
            current_time += dt
        
        final_density = network.segments['seg_0']['U'][0, 2]
        
        # With no blocking (GREEN), density should decrease or stay stable
        # (outflow rate equals or exceeds inflow rate)
        assert final_density <= initial_density * 1.05, \
            f"No blocking should allow normal flow. Initial: {initial_density:.4f}, Final: {final_density:.4f}"
    
    # ==================== Test RED vs GREEN Behavior ====================
    
    def test_red_blocks_green_flows(self, base_params):
        """Test that RED blocks while GREEN allows flow."""
        # RED signal scenario
        network_red = self.create_2segment_network_with_light(
            base_params.copy(), 
            red_light_factor=0.05,
            spatial_scheme='first_order'
        )
        
        # GREEN signal scenario (light_factor=1.0)
        network_green = self.create_2segment_network_with_light(
            base_params.copy(), 
            red_light_factor=1.0,
            spatial_scheme='first_order'
        )
        
        # Simulate both
        dt = 0.5
        t_end = 30.0
        current_time = 0.0
        
        initial_red = network_red.segments['seg_0']['U'][0, 2].copy()
        initial_green = network_green.segments['seg_0']['U'][0, 2].copy()
        
        while current_time < t_end:
            network_red.step(dt, current_time)
            network_green.step(dt, current_time)
            current_time += dt
        
        final_red = network_red.segments['seg_0']['U'][0, 2]
        final_green = network_green.segments['seg_0']['U'][0, 2]
        
        # RED should accumulate density
        red_increase = final_red - initial_red
        assert red_increase > 0.01, \
            f"RED should cause congestion. Increase: {red_increase:.4f}"
        
        # GREEN should not accumulate (or accumulate much less)
        green_increase = final_green - initial_green
        
        # The key test: RED accumulates MORE than GREEN
        assert red_increase > green_increase * 2.0, \
            f"RED should accumulate more than GREEN. RED: {red_increase:.4f}, GREEN: {green_increase:.4f}"
    
    # ==================== Test Different Junction Positions ====================
    
    def test_junction_at_different_segment_lengths(self, base_params):
        """Test junction blocking works with different segment lengths."""
        # Short segments
        network_short = NetworkGrid(base_params.copy())
        base_params.spatial_scheme = 'first_order'
        base_params.red_light_factor = 0.05
        
        topology_short = [
            {'segment_id': 'seg_0', 'start_node': 'node_0', 'end_node': 'node_1', 'length': 100.0},
            {'segment_id': 'seg_1', 'start_node': 'node_1', 'end_node': 'node_2', 'length': 100.0}
        ]
        
        node_0 = Node(node_id='node_0', position=0.0)
        node_1 = Node(node_id='node_1', position=100.0)
        node_2 = Node(node_id='node_2', position=200.0)
        
        light = TrafficLightController(
            node_id='node_1',
            cycle_time=120.0,
            green_splits={'seg_0': 0.5, 'seg_1': 0.5}
        )
        node_1.add_traffic_light(light)
        
        nodes = {
            'node_0': node_0,
            'node_1': node_1,
            'node_2': node_2
        }
        
        network_short.initialize_from_topology(topology_short, nodes)
        
        # Set ICs
        network_short.set_initial_conditions('seg_0', rho_m=0.08, rho_c=0.0, v_m=8.89, w_m=0.0)
        network_short.set_initial_conditions('seg_1', rho_m=0.02, rho_c=0.0, v_m=10.0, w_m=0.0)
        
        # Simulate
        dt = 0.5
        t_end = 20.0
        current_time = 0.0
        
        initial_density = network_short.segments['seg_0']['U'][0, 2].copy()
        
        while current_time < t_end:
            network_short.step(dt, current_time)
            current_time += dt
        
        final_density = network_short.segments['seg_0']['U'][0, 2]
        
        # Should still work with shorter segments
        assert final_density > initial_density, \
            f"Junction blocking should work with L=100m segments. Initial: {initial_density:.4f}, Final: {final_density:.4f}"
    
    # ==================== Numerical Stability Tests ====================
    
    def test_no_negative_densities_during_blocking(self, base_params):
        """Verify junction blocking does not cause negative densities."""
        network = self.create_2segment_network_with_light(
            base_params, 
            red_light_factor=0.05,
            spatial_scheme='first_order'
        )
        
        dt = 0.5
        t_end = 60.0
        current_time = 0.0
        
        while current_time < t_end:
            network.step(dt, current_time)
            current_time += dt
            
            # Check all segments for negative densities
            for seg_id, segment in network.segments.items():
                U = segment['U']
                rho_m = U[0, :]
                rho_c = U[1, :]
                
                assert np.all(rho_m >= 0), \
                    f"Negative rho_m detected in {seg_id} at t={current_time:.1f}s"
                assert np.all(rho_c >= 0), \
                    f"Negative rho_c detected in {seg_id} at t={current_time:.1f}s"
    
    def test_no_inf_or_nan_during_blocking(self, base_params):
        """Verify junction blocking does not cause NaN or Inf values."""
        network = self.create_2segment_network_with_light(
            base_params, 
            red_light_factor=0.05,
            spatial_scheme='first_order'
        )
        
        dt = 0.5
        t_end = 60.0
        current_time = 0.0
        
        while current_time < t_end:
            network.step(dt, current_time)
            current_time += dt
            
            # Check all segments for NaN/Inf
            for seg_id, segment in network.segments.items():
                U = segment['U']
                
                assert np.all(np.isfinite(U)), \
                    f"NaN or Inf detected in {seg_id} at t={current_time:.1f}s"
    
    # ==================== Integration with Behavioral Coupling ====================
    
    def test_junction_blocking_with_theta_coupling(self, base_params):
        """Test junction flux blocking works correctly with θ_k behavioral coupling."""
        # Enable behavioral coupling
        base_params.use_behavioral_coupling = True
        base_params.theta_k = 0.5  # Moderate coupling
        
        network = self.create_2segment_network_with_light(
            base_params, 
            red_light_factor=0.05,
            spatial_scheme='first_order'
        )
        
        # Simulate
        dt = 0.5
        t_end = 30.0
        current_time = 0.0
        
        initial_density = network.segments['seg_0']['U'][0, 2].copy()
        
        while current_time < t_end:
            network.step(dt, current_time)
            current_time += dt
        
        final_density = network.segments['seg_0']['U'][0, 2]
        
        # Junction blocking should still work with behavioral coupling enabled
        assert final_density > initial_density, \
            f"Junction blocking should work with θ_k={base_params.theta_k}. Initial: {initial_density:.4f}, Final: {final_density:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
