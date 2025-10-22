"""
Unit tests for network module (NetworkGrid, Node, Link, topology).

These tests validate the network infrastructure following professional
patterns from SUMO and CityFlow. They ensure:
- Node wraps Intersection correctly with topology
- Link applies θ_k coupling between segments
- NetworkGrid coordinates multi-segment networks
- Topology utilities validate network structure

Academic Reference:
    - Garavello & Piccoli (2005): Network formulation validation
    - Kolb et al. (2018): θ_k coupling validation
"""

import unittest
import numpy as np
from typing import Dict

from arz_model.core.parameters import ModelParameters
from arz_model.core.intersection import Intersection
from arz_model.core.traffic_lights import TrafficLightController
from arz_model.grid.grid1d import Grid1D
from arz_model.network import NetworkGrid, Node, Link
from arz_model.network import topology as topo


class TestNode(unittest.TestCase):
    """Test Node class (junction wrapper)."""
    
    def setUp(self):
        """Setup test parameters and intersection."""
        self.params = ModelParameters()
        self.params.rho_jam = 0.2
        self.params.Vmax_c = {1: 50/3.6}
        
        # Intersection requires: node_id, position, segments list, traffic_lights (optional)
        self.intersection = Intersection(
            node_id='test_node',
            position=100.0,
            segments=['seg_A', 'seg_B', 'seg_C', 'seg_D']
        )
        
    def test_node_creation(self):
        """Test basic node creation with topology."""
        node = Node(
            node_id='node_1',
            position=(100.0, 50.0),
            intersection=self.intersection,
            incoming_segments=['seg_A', 'seg_B'],
            outgoing_segments=['seg_C', 'seg_D'],
            node_type='signalized'
        )
        
        self.assertEqual(node.node_id, 'node_1')
        self.assertEqual(node.position, (100.0, 50.0))
        self.assertEqual(len(node.incoming_segments), 2)
        self.assertEqual(len(node.outgoing_segments), 2)
        self.assertEqual(node.node_type, 'signalized')
        
    def test_node_requires_incoming_segments(self):
        """Test that node requires at least one incoming segment."""
        with self.assertRaises(ValueError):
            Node(
                node_id='node_bad',
                position=(0, 0),
                intersection=self.intersection,
                incoming_segments=[],  # Empty!
                outgoing_segments=['seg_C']
            )
            
    def test_node_get_incoming_states(self):
        """Test retrieving boundary states from incoming segments."""
        # Create mock segments (dict format used by NetworkGrid)
        segments = {}
        for seg_id in ['seg_A', 'seg_B']:
            grid = Grid1D(10, 0, 100, num_ghost_cells=2)
            U = np.random.rand(4, grid.N_total) * 0.1  # Random states
            segments[seg_id] = {'grid': grid, 'U': U}
            
        node = Node(
            node_id='node_1',
            position=(100, 0),
            intersection=self.intersection,
            incoming_segments=['seg_A', 'seg_B'],
            outgoing_segments=['seg_C']
        )
        
        states = node.get_incoming_states(segments)
        
        self.assertEqual(len(states), 2)
        self.assertIn('seg_A', states)
        self.assertIn('seg_B', states)
        self.assertEqual(states['seg_A'].shape, (4,))  # Boundary state


class TestLink(unittest.TestCase):
    """Test Link class (segment coupling)."""
    
    def setUp(self):
        """Setup test parameters and components."""
        self.params = ModelParameters()
        self.params.rho_jam = 0.2
        self.params.Vmax_c = {1: 50/3.6}
        self.params.gamma_m = 2.0
        self.params.K_m = 5.0
        self.params.gamma_c = 2.0
        self.params.K_c = 5.0
        # Signalized junction θ_k values
        self.params.theta_moto_signalized = 0.8
        self.params.theta_car_signalized = 0.5
        # Priority junction θ_k values
        self.params.theta_moto_priority = 0.9
        self.params.theta_car_priority = 0.9
        # Secondary junction θ_k values (yield/stop)
        self.params.theta_moto_secondary = 0.1
        self.params.theta_car_secondary = 0.1
        # Memory distance θ_k values
        self.params.theta_moto_insertion = 0.2
        self.params.theta_moto_circulation = 0.8
        self.params.alpha = 0.7
        self.params.epsilon = 1e-10
        
        # Create traffic light with 'seg_out' as green segment
        from arz_model.core.traffic_lights import TrafficLightController, Phase
        phases = [
            Phase(duration=30.0, green_segments=['seg_out']),  # seg_out is always green
            Phase(duration=30.0, green_segments=['seg_in'])     # alternates
        ]
        traffic_lights = TrafficLightController(cycle_time=60.0, phases=phases)
        
        intersection = Intersection(
            node_id='junction_1',
            position=100.0,
            segments=['seg_in', 'seg_out'],
            traffic_lights=traffic_lights
        )
        
        self.node = Node(
            node_id='junction_1',
            position=(100, 0),
            intersection=intersection,
            incoming_segments=['seg_in'],
            outgoing_segments=['seg_out'],
            node_type='signalized'
        )
        
    def test_link_creation(self):
        """Test basic link creation."""
        link = Link(
            link_id='link_1',
            from_segment='seg_in',
            to_segment='seg_out',
            via_node=self.node,
            coupling_type='sequential',
            params=self.params
        )
        
        self.assertEqual(link.link_id, 'link_1')
        self.assertEqual(link.from_segment, 'seg_in')
        self.assertEqual(link.to_segment, 'seg_out')
        self.assertEqual(link.coupling_type, 'sequential')
        
    def test_link_validates_segments(self):
        """Test that link validates segment connectivity."""
        with self.assertRaises(ValueError):
            Link(
                link_id='bad_link',
                from_segment='nonexistent',  # Not in node's incoming!
                to_segment='seg_out',
                via_node=self.node,
                params=self.params
            )
            
    def test_link_apply_coupling(self):
        """Test θ_k coupling application."""
        link = Link(
            link_id='link_1',
            from_segment='seg_in',
            to_segment='seg_out',
            via_node=self.node,
            coupling_type='sequential',
            params=self.params
        )
        
        # Create test states
        U_in = np.array([0.05, 20.0, 0.08, 15.0])
        U_out = np.array([0.03, 18.0, 0.05, 14.0])
        
        # Apply coupling
        U_coupled = link.apply_coupling(U_in, U_out, vehicle_class='motorcycle')
        
        # Verify shape preserved
        self.assertEqual(U_coupled.shape, (4,))
        
        # Verify w values changed (coupling applied)
        self.assertNotEqual(U_coupled[1], U_out[1])  # w_m changed
        
    def test_link_get_coupling_strength(self):
        """Test retrieving θ_k value."""
        link = Link(
            link_id='link_1',
            from_segment='seg_in',
            to_segment='seg_out',
            via_node=self.node,
            params=self.params
        )
        
        theta_moto = link.get_coupling_strength('motorcycle')
        theta_car = link.get_coupling_strength('car')
        
        # At signalized junction (defaults to green)
        self.assertGreater(theta_moto, 0.0)
        self.assertGreater(theta_car, 0.0)


class TestNetworkGrid(unittest.TestCase):
    """Test NetworkGrid coordinator."""
    
    def setUp(self):
        """Setup test parameters."""
        self.params = ModelParameters()
        self.params.rho_jam = 0.2
        self.params.Vmax_c = {1: 50/3.6}
        self.params.gamma_m = 2.0
        self.params.K_m = 5.0
        
    def test_network_creation(self):
        """Test creating empty network."""
        network = NetworkGrid(self.params)
        
        self.assertEqual(len(network.segments), 0)
        self.assertEqual(len(network.nodes), 0)
        self.assertEqual(len(network.links), 0)
        self.assertFalse(network._initialized)
        
    def test_add_segment(self):
        """Test adding segment to network."""
        network = NetworkGrid(self.params)
        
        segment = network.add_segment(
            segment_id='seg_1',
            xmin=0,
            xmax=100,
            N=10
        )
        
        self.assertIn('seg_1', network.segments)
        self.assertEqual(segment['grid'].xmin, 0)
        self.assertEqual(segment['grid'].xmax, 100)
        self.assertEqual(segment['grid'].N_physical, 10)
        
    def test_add_node(self):
        """Test adding node to network."""
        network = NetworkGrid(self.params)
        
        # Add segments first
        network.add_segment('seg_in', 0, 100, 10)
        network.add_segment('seg_out', 100, 200, 10)
        
        # Add node
        node = network.add_node(
            node_id='node_1',
            position=(100, 0),
            incoming_segments=['seg_in'],
            outgoing_segments=['seg_out'],
            node_type='signalized'
        )
        
        self.assertIn('node_1', network.nodes)
        self.assertEqual(node.node_type, 'signalized')
        
    def test_add_link(self):
        """Test adding link to network."""
        network = NetworkGrid(self.params)
        
        # Build simple 2-segment network
        network.add_segment('seg_1', 0, 100, 10)
        network.add_segment('seg_2', 100, 200, 10)
        network.add_node(
            'node_A',
            position=(100, 0),
            incoming_segments=['seg_1'],
            outgoing_segments=['seg_2']
        )
        
        # Add link
        link = network.add_link('seg_1', 'seg_2', 'node_A')
        
        self.assertEqual(len(network.links), 1)
        self.assertEqual(link.from_segment, 'seg_1')
        self.assertEqual(link.to_segment, 'seg_2')
        
    def test_network_initialization(self):
        """Test network initialization and validation."""
        network = NetworkGrid(self.params)
        
        # Build minimal valid network
        network.add_segment('seg_1', 0, 100, 10)
        network.add_segment('seg_2', 100, 200, 10)
        network.add_node(
            'node_A',
            position=(100, 0),
            incoming_segments=['seg_1'],
            outgoing_segments=['seg_2']
        )
        network.add_link('seg_1', 'seg_2', 'node_A')
        
        # Initialize
        network.initialize()
        
        self.assertTrue(network._initialized)
        
    def test_network_step_requires_initialization(self):
        """Test that step() requires initialized network."""
        network = NetworkGrid(self.params)
        
        with self.assertRaises(RuntimeError):
            network.step(0.1)
            
    def test_get_network_metrics(self):
        """Test computing network-wide metrics."""
        network = NetworkGrid(self.params)
        
        # Add segment with known state
        seg = network.add_segment('seg_1', 0, 100, 10)
        grid = seg['grid']
        U = seg['U']
        
        # Set physical cells to known values
        phys_slice = grid.physical_cell_indices
        U[0, phys_slice] = 0.05  # ρ_m
        U[1, phys_slice] = 10.0  # w_m
        U[2, phys_slice] = 0.08  # ρ_c
        U[3, phys_slice] = 8.0   # w_c
        
        metrics = network.get_network_metrics()
        
        self.assertIn('total_vehicles', metrics)
        self.assertIn('avg_speed', metrics)
        self.assertIn('total_flux', metrics)
        self.assertGreater(metrics['total_vehicles'], 0)


class TestTopology(unittest.TestCase):
    """Test topology utilities."""
    
    def setUp(self):
        """Setup test network components."""
        self.params = ModelParameters()
        self.params.rho_jam = 0.2
        self.params.Vmax_c = {1: 50/3.6}
        
    def test_find_upstream_segments(self):
        """Test finding upstream segments."""
        intersection = Intersection(
            node_id='node_1',
            position=100.0,
            segments=['seg_A', 'seg_B', 'seg_C', 'seg_D']
        )
        node = Node(
            'node_1',
            (0, 0),
            intersection,
            incoming_segments=['seg_A', 'seg_B'],
            outgoing_segments=['seg_C', 'seg_D']
        )
        
        nodes = {'node_1': node}
        upstream = topo.find_upstream_segments('node_1', nodes)
        
        self.assertEqual(len(upstream), 2)
        self.assertIn('seg_A', upstream)
        self.assertIn('seg_B', upstream)
        
    def test_find_downstream_segments(self):
        """Test finding downstream segments."""
        intersection = Intersection(
            node_id='node_1',
            position=100.0,
            segments=['seg_A', 'seg_B', 'seg_C', 'seg_D']
        )
        node = Node(
            'node_1',
            (0, 0),
            intersection,
            incoming_segments=['seg_A', 'seg_B'],
            outgoing_segments=['seg_C', 'seg_D']
        )
        
        nodes = {'node_1': node}
        downstream = topo.find_downstream_segments('node_1', nodes)
        
        self.assertEqual(len(downstream), 2)
        self.assertIn('seg_C', downstream)
        self.assertIn('seg_D', downstream)
        
    def test_validate_topology_detects_isolated_segments(self):
        """Test topology validation catches isolated segments."""
        network = NetworkGrid(self.params)
        
        # Add segment but no nodes (isolated!)
        network.add_segment('seg_isolated', 0, 100, 10)
        
        # Validate
        is_valid, errors = topo.validate_topology(
            None,
            network.segments,
            network.nodes
        )
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)


if __name__ == '__main__':
    unittest.main()
