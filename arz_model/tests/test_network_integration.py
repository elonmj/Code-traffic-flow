"""
Integration tests for NetworkGrid with time integration.

Tests the complete simulation loop:
1. NetworkGrid.step() advances all segments
2. Node coupling applies θ_k at junctions
3. Traffic lights update
"""

import unittest
import numpy as np
from arz_model.core.parameters import ModelParameters
from arz_model.network import NetworkGrid


class TestNetworkIntegration(unittest.TestCase):
    """Test NetworkGrid with real time integration."""
    
    def setUp(self):
        """Setup test parameters and network."""
        self.params = ModelParameters()
        self.params.device = 'cpu'
        self.params.rho_jam = 0.2
        self.params.Vmax_m = {1: 60/3.6, 2: 60/3.6}  # Motorcycles max speed by road quality
        self.params.Vmax_c = {1: 50/3.6, 2: 50/3.6}  # Cars max speed by road quality
        self.params.V_creeping = 1.4  # Creeping speed (m/s) ~5 km/h
        self.params.gamma_m = 2.0
        self.params.K_m = 5.0
        self.params.gamma_c = 2.0
        self.params.K_c = 5.0
        self.params.tau_m = 0.5  # Relaxation time motorcycles
        self.params.tau_c = 0.8  # Relaxation time cars
        self.params.theta_moto_signalized = 0.8
        self.params.theta_car_signalized = 0.5
        self.params.theta_moto_secondary = 0.1
        self.params.theta_car_secondary = 0.1
        self.params.alpha = 0.7
        self.params.epsilon = 1e-10
        # ODE solver parameters
        self.params.ode_solver = 'RK45'
        self.params.ode_rtol = 1e-6
        self.params.ode_atol = 1e-8
        # Numerical scheme parameters
        self.params.spatial_scheme = 'first_order'
        self.params.time_scheme = 'ssprk3'
        
    def test_simple_two_segment_network(self):
        """Test 2-segment network with junction."""
        network = NetworkGrid(self.params)
        
        # Add two segments: seg_1 → junction → seg_2
        seg1 = network.add_segment('seg_1', xmin=0, xmax=100, N=20,
                                   start_node=None, end_node='junction')
        seg2 = network.add_segment('seg_2', xmin=100, xmax=200, N=20,
                                   start_node='junction', end_node=None)
        
        # Set initial conditions (some traffic in seg_1)
        seg1['U'][0, :] = 0.05  # ρ_m = 5% jam density
        seg1['U'][1, :] = 5.0   # w_m = 5 m/s
        seg1['U'][2, :] = 0.03  # ρ_c = 3% jam density
        seg1['U'][3, :] = 8.0   # w_c = 8 m/s
        
        # seg_2 starts empty
        seg2['U'][:, :] = 0.0
        
        # Add junction node with traffic light
        from arz_model.core.traffic_lights import TrafficLightController, Phase
        phases = [
            Phase(duration=30.0, green_segments=['seg_2']),  # seg_2 always green
            Phase(duration=30.0, green_segments=['seg_1'])
        ]
        traffic_lights = TrafficLightController(cycle_time=60.0, phases=phases)
        
        node = network.add_node('junction', position=(100, 0),
                               incoming_segments=['seg_1'],
                               outgoing_segments=['seg_2'],
                               node_type='signalized',
                               traffic_lights=traffic_lights)
        
        # Add link connecting segments through junction
        link = network.add_link('seg_1', 'seg_2', 'junction', coupling_type='sequential')
        
        # Initialize network
        network.initialize()
        
        # Run simulation for a few steps
        dt = 0.1
        for i in range(10):
            network.step(dt, current_time=i*dt)
        
        # Check that seg_2 now has some traffic (propagated from seg_1)
        seg2_after = network.segments['seg_2']['U']
        total_seg2 = np.sum(seg2_after[[0, 2], :])  # Total density (moto + car)
        
        # After 10 steps, some traffic should have propagated
        self.assertGreater(total_seg2, 0.0, 
                          "Traffic should propagate from seg_1 to seg_2 through junction")
        
    def test_network_state_retrieval(self):
        """Test that get_network_state() returns valid data after step()."""
        network = NetworkGrid(self.params)
        
        seg1 = network.add_segment('seg_1', xmin=0, xmax=100, N=10)
        seg2 = network.add_segment('seg_2', xmin=100, xmax=200, N=10)
        network.add_node('node_1', position=(100, 0),
                        incoming_segments=['seg_1'], outgoing_segments=['seg_2'],
                        node_type='priority')
        network.add_link('seg_1', 'seg_2', 'node_1')
        network.initialize()
        
        # Step once
        network.step(0.1, current_time=0.0)
        
        # Get state
        state = network.get_network_state()
        self.assertIn('seg_1', state)
        self.assertEqual(state['seg_1'].shape, (4, 14))  # 10 physical + 4 ghost cells
        
    def test_network_metrics_after_simulation(self):
        """Test that get_network_metrics() computes correctly after steps."""
        network = NetworkGrid(self.params)
        
        seg1 = network.add_segment('seg_1', xmin=0, xmax=100, N=20)
        seg2 = network.add_segment('seg_2', xmin=100, xmax=200, N=20)
        network.add_node('node_1', position=(100, 0),
                        incoming_segments=['seg_1'], outgoing_segments=['seg_2'],
                        node_type='priority')
        network.add_link('seg_1', 'seg_2', 'node_1')
        network.initialize()
        
        # Set initial conditions
        seg1['U'][0, :] = 0.05  # Some motorcycles
        seg1['U'][1, :] = 10.0
        seg1['U'][2, :] = 0.03  # Some cars
        seg1['U'][3, :] = 12.0
        
        # Step a few times
        for _ in range(5):
            network.step(0.1, current_time=0.0)
        
        # Get metrics
        metrics = network.get_network_metrics()
        
        self.assertIn('total_vehicles', metrics)
        self.assertIn('avg_speed', metrics)
        self.assertIn('total_flux', metrics)
        
        self.assertGreater(metrics['total_vehicles'], 0.0)
        self.assertGreater(metrics['avg_speed'], 0.0)


if __name__ == '__main__':
    unittest.main()
