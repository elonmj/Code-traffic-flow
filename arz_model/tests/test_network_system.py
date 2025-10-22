import unittest
import numpy as np
from unittest.mock import Mock, patch

from ..core.intersection import Intersection, create_intersection_from_config
from ..core.traffic_lights import TrafficLightController, Phase
from ..core.node_solver import solve_node_fluxes
from ..numerics.network_coupling import NetworkCoupling
from ..core.parameters import ModelParameters


class TestIntersection(unittest.TestCase):
    """Test cases for Intersection class."""

    def setUp(self):
        """Set up test fixtures."""
        self.traffic_lights = TrafficLightController(cycle_time=60.0)
        self.intersection = Intersection(
            node_id="test_intersection",
            position=100.0,
            segments=["seg1", "seg2", "seg3", "seg4"],
            traffic_lights=self.traffic_lights
        )

    def test_initialization(self):
        """Test intersection initialization."""
        self.assertEqual(self.intersection.node_id, "test_intersection")
        self.assertEqual(self.intersection.position, 100.0)
        self.assertEqual(len(self.intersection.segments), 4)
        self.assertIsInstance(self.intersection.traffic_lights, TrafficLightController)

    def test_queue_management(self):
        """Test queue length updates."""
        # Initial queues should be empty
        self.assertEqual(self.intersection.queues['motorcycle'], 0.0)
        self.assertEqual(self.intersection.queues['car'], 0.0)

        # Update queues
        incoming_fluxes = {'motorcycle': 10.0, 'car': 15.0}
        outgoing_capacities = {'motorcycle': 8.0, 'car': 12.0}
        dt = 1.0

        self.intersection.update_queues(incoming_fluxes, outgoing_capacities, dt)

        # Check queue accumulation
        expected_m_queue = (10.0 - 8.0) * dt
        expected_c_queue = (15.0 - 12.0) * dt
        self.assertEqual(self.intersection.queues['motorcycle'], expected_m_queue)
        self.assertEqual(self.intersection.queues['car'], expected_c_queue)

    def test_creeping_speed(self):
        """Test creeping speed calculation."""
        # No creeping when queue is small
        self.intersection.queues['motorcycle'] = 10.0
        speed = self.intersection.get_creeping_speed('motorcycle')
        self.assertEqual(speed, 0.0)

        # Creeping when queue exceeds threshold
        self.intersection.queues['motorcycle'] = 60.0
        speed = self.intersection.get_creeping_speed('motorcycle')
        self.assertGreater(speed, 0.0)

    def test_outgoing_capacity(self):
        """Test outgoing capacity calculation."""
        # Test with green light
        self.intersection.traffic_lights.phases[0].green_segments = ["seg1"]
        capacity = self.intersection.get_outgoing_capacity("seg1", 'motorcycle', 0.0)
        self.assertGreater(capacity, 0.0)

        # Test with red light
        capacity_red = self.intersection.get_outgoing_capacity("seg2", 'motorcycle', 0.0)
        self.assertEqual(capacity_red, 0.0)


class TestTrafficLights(unittest.TestCase):
    """Test cases for TrafficLightController."""

    def setUp(self):
        """Set up test fixtures."""
        phases = [
            Phase(duration=30.0, green_segments=["north", "south"]),
            Phase(duration=30.0, green_segments=["east", "west"])
        ]
        self.controller = TrafficLightController(cycle_time=60.0, phases=phases)

    def test_phase_transitions(self):
        """Test phase transitions over time."""
        # Phase 1: North-South green
        current_phase = self.controller.get_current_phase(0.0)
        self.assertIn("north", current_phase.green_segments)
        self.assertIn("south", current_phase.green_segments)

        # Phase 2: East-West green
        current_phase = self.controller.get_current_phase(35.0)
        self.assertIn("east", current_phase.green_segments)
        self.assertIn("west", current_phase.green_segments)

    def test_cycle_completion(self):
        """Test cycle completion and reset."""
        # End of cycle should wrap to beginning
        current_phase = self.controller.get_current_phase(60.0)
        first_phase = self.controller.get_current_phase(0.0)
        self.assertEqual(current_phase.duration, first_phase.duration)


class TestNodeSolver(unittest.TestCase):
    """Test cases for node solver functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = ModelParameters()
        self.params.alpha = 0.5
        self.params.rho_jam = 0.2
        self.params.K_m = 10.0
        self.params.K_c = 15.0
        self.params.gamma_m = 1.0
        self.params.gamma_c = 1.0
        self.params.red_light_factor = 0.1
        self.params.epsilon = 1e-6

    def test_solve_node_fluxes(self):
        """Test node flux solving."""
        # Create mock intersection
        intersection = Mock()
        intersection.segments = ["seg1", "seg2"]
        intersection.traffic_lights.get_current_green_segments.return_value = ["seg1"]
        intersection.queues = {'motorcycle': 0.0, 'car': 0.0}

        # Mock incoming states
        incoming_states = {
            "seg1": np.array([0.1, 5.0, 0.05, 10.0]),  # rho_m, w_m, rho_c, w_c
            "seg2": np.array([0.08, 3.0, 0.04, 8.0])
        }

        # Solve fluxes
        fluxes = solve_node_fluxes(intersection, incoming_states, 0.1, self.params, 0.0)

        # Check that fluxes are returned for all segments
        self.assertIn("seg1", fluxes)
        self.assertIn("seg2", fluxes)
        self.assertEqual(len(fluxes["seg1"]), 4)  # Should be 4-component state vector


class TestNetworkCoupling(unittest.TestCase):
    """Test cases for NetworkCoupling."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = ModelParameters()
        self.params.has_network = True
        self.nodes = []  # Empty for basic tests
        self.network_coupling = NetworkCoupling(self.nodes, self.params)

    def test_initialization(self):
        """Test network coupling initialization."""
        self.assertEqual(len(self.network_coupling.nodes), 0)
        self.assertIsInstance(self.network_coupling.node_states, dict)

    def test_apply_coupling_no_nodes(self):
        """Test coupling application with no nodes."""
        U = np.random.rand(4, 100)
        grid = Mock()
        grid.N_total = 100

        # Should return U unchanged when no nodes
        result = self.network_coupling.apply_network_coupling(U, 0.1, grid, 0.0)
        np.testing.assert_array_equal(result, U)


class TestIntersectionCreation(unittest.TestCase):
    """Test intersection creation from configuration."""

    def test_create_from_config(self):
        """Test creating intersection from YAML config."""
        config = {
            'id': 'test_node',
            'position': 150.0,
            'segments': ['seg1', 'seg2'],
            'traffic_lights': {
                'cycle_time': 90.0,
                'phases': [
                    {'duration': 40.0, 'green_segments': ['seg1']},
                    {'duration': 50.0, 'green_segments': ['seg2']}
                ]
            },
            'max_queue_lengths': {
                'motorcycle': 80.0,
                'car': 80.0
            }
        }

        intersection = create_intersection_from_config(config)

        self.assertEqual(intersection.node_id, 'test_node')
        self.assertEqual(intersection.position, 150.0)
        self.assertEqual(len(intersection.segments), 2)
        self.assertIsNotNone(intersection.traffic_lights)
        self.assertEqual(intersection.max_queue_lengths['motorcycle'], 80.0)


class TestBehavioralCoupling(unittest.TestCase):
    """Test cases for θ_k behavioral coupling (thesis contribution)."""

    def setUp(self):
        """Set up test fixtures with θ_k parameters."""
        self.params = ModelParameters()
        # Set θ_k values
        self.params.theta_moto_insertion = 0.2
        self.params.theta_moto_circulation = 0.8
        self.params.theta_moto_signalized = 0.8
        self.params.theta_car_signalized = 0.5
        self.params.theta_moto_priority = 0.9
        self.params.theta_car_priority = 0.9
        self.params.theta_moto_secondary = 0.1
        self.params.theta_car_secondary = 0.1
        
        # Set physical parameters
        self.params.gamma_m = 1.5
        self.params.gamma_c = 2.0
        self.params.K_m = 10.0 / 3.6  # Convert km/h to m/s
        self.params.K_c = 15.0 / 3.6
        self.params.rho_jam = 250.0 / 1000.0  # Convert veh/km to veh/m
        self.params.epsilon = 1e-10
        self.params.Vmax_c = {3: 35.0 / 3.6}  # Urban default in m/s

    def test_get_coupling_parameter_green_light(self):
        """Test θ_k selection for green light (signalized intersection)."""
        from ..core.node_solver import _get_coupling_parameter
        
        # Create intersection with traffic lights
        phases = [Phase(duration=30.0, green_segments=['seg1'], yellow_segments=[])]
        traffic_lights = TrafficLightController(cycle_time=60.0, phases=phases)
        node = Intersection(
            node_id='test', position=0.0, segments=['seg1', 'seg2'],
            traffic_lights=traffic_lights
        )
        
        # Green light - motorcycles should have higher θ than cars
        theta_moto = _get_coupling_parameter(node, 'seg1', 'motorcycle', self.params, time=0.0)
        theta_car = _get_coupling_parameter(node, 'seg1', 'car', self.params, time=0.0)
        
        self.assertEqual(theta_moto, 0.8)  # theta_moto_signalized
        self.assertEqual(theta_car, 0.5)   # theta_car_signalized
        self.assertGreater(theta_moto, theta_car, "Motos should preserve more memory than cars")

    def test_get_coupling_parameter_red_light(self):
        """Test θ_k=0 for red light (complete behavioral reset)."""
        from ..core.node_solver import _get_coupling_parameter
        
        # Create intersection with traffic lights
        phases = [Phase(duration=30.0, green_segments=['seg1'], yellow_segments=[])]
        traffic_lights = TrafficLightController(cycle_time=60.0, phases=phases)
        node = Intersection(
            node_id='test', position=0.0, segments=['seg1', 'seg2'],
            traffic_lights=traffic_lights
        )
        
        # Red light (seg2 not in green) - should give θ=0
        theta_moto = _get_coupling_parameter(node, 'seg2', 'motorcycle', self.params, time=0.0)
        theta_car = _get_coupling_parameter(node, 'seg2', 'car', self.params, time=0.0)
        
        self.assertEqual(theta_moto, 0.0, "Red light should reset motorcycle behavior")
        self.assertEqual(theta_car, 0.0, "Red light should reset car behavior")

    def test_apply_behavioral_coupling_theta_zero(self):
        """Test behavioral coupling with θ=0 (complete reset to equilibrium)."""
        from ..core.node_solver import _apply_behavioral_coupling
        
        # Incoming state: high w value
        U_in = np.array([0.05, 15.0, 0.03, 12.0])  # [ρ_m, w_m, ρ_c, w_c]
        # Outgoing state: equilibrium conditions
        U_out = np.array([0.02, 8.0, 0.01, 6.0])
        
        theta_k = 0.0  # Complete reset
        
        # Apply coupling for motorcycles
        U_coupled = _apply_behavioral_coupling(U_in, U_out, theta_k, self.params, 'motorcycle')
        
        # With θ=0, w should reset to equilibrium (w_eq_out)
        # The exact value depends on V_e + p calculation, but should NOT preserve w_in
        self.assertNotEqual(U_coupled[1], U_in[1], "θ=0 should not preserve incoming w")
        self.assertEqual(U_coupled[0], U_out[0], "Density should be preserved")

    def test_apply_behavioral_coupling_theta_one(self):
        """Test behavioral coupling with θ=1 (perfect memory preservation)."""
        from ..core.node_solver import _apply_behavioral_coupling
        
        # Incoming state
        U_in = np.array([0.05, 15.0, 0.03, 12.0])
        # Outgoing state (different w)
        U_out = np.array([0.05, 8.0, 0.03, 6.0])  # Same ρ to simplify
        
        theta_k = 1.0  # Perfect preservation
        
        # Apply coupling for motorcycles
        U_coupled = _apply_behavioral_coupling(U_in, U_out, theta_k, self.params, 'motorcycle')
        
        # With θ=1, w should be influenced by incoming w
        # w_out = w_eq_out + 1.0 * (w_in - w_eq_in)
        # Since ρ same, w_eq should be same, so w_out ≈ w_in
        self.assertNotEqual(U_coupled[1], U_out[1], "θ=1 should modify outgoing w")

    def test_coupling_motorcycle_vs_car_difference(self):
        """Test that motorcycles preserve more memory than cars at green lights."""
        from ..core.node_solver import _get_coupling_parameter
        
        # Signalized intersection with green light
        phases = [Phase(duration=30.0, green_segments=['seg1'], yellow_segments=[])]
        traffic_lights = TrafficLightController(cycle_time=60.0, phases=phases)
        node = Intersection(
            node_id='test', position=0.0, segments=['seg1', 'seg2'],
            traffic_lights=traffic_lights
        )
        
        # Green light - compare θ values
        theta_moto = _get_coupling_parameter(node, 'seg1', 'motorcycle', self.params, time=0.0)
        theta_car = _get_coupling_parameter(node, 'seg1', 'car', self.params, time=0.0)
        
        # Thesis hypothesis: motos preserve more aggressive behavior
        self.assertGreater(theta_moto, theta_car,
                          "Motorcycles should have higher θ (more memory) than cars at green lights")
        self.assertGreaterEqual(theta_moto, 0.7, "Moto θ should be ≥0.7")
        self.assertLessEqual(theta_car, 0.6, "Car θ should be ≤0.6")


if __name__ == '__main__':
    unittest.main()
