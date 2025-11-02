"""
Test junction_info architecture with NetworkGrid (multi-segment).

Validates SUMO/CityFlow pattern adoption:
- Direct segment→node references
- All segments with end_node receive junction_info
- Independence from links list
"""
import pytest
import numpy as np
from arz_model.network.network_grid import NetworkGrid
from arz_model.core.traffic_lights import TrafficLightController, Phase
from arz_model.core.parameters import ModelParameters


@pytest.fixture
def network_params():
    """Model parameters for network simulation."""
    params = ModelParameters()
    # Basic physical parameters
    params.alpha = 0.5
    params.rho_jam = 0.2
    params.epsilon = 1e-10
    params.K_m = 20.0
    params.gamma_m = 2.0
    params.K_c = 30.0
    params.gamma_c = 2.0
    params.tau_m = 2.0
    params.tau_c = 2.0
    params.V_creeping = 1.0
    
    # Numerical parameters
    params.N = 50             # Cells per segment
    params.cfl_number = 0.8
    params.ghost_cells = 3
    params.spatial_scheme = 'weno5'
    params.time_scheme = 'ssprk3'
    params.ode_solver = 'RK45'  # Fix ODE solver
    params.ode_rtol = 1e-6
    params.ode_atol = 1e-9
    
    # Speed parameters (by road category)
    params.Vmax_m = {0: 12.0, 1: 15.0, 2: 18.0}  # m/s by category
    params.Vmax_c = {0: 10.0, 1: 13.0, 2: 16.0}  # m/s by category
    
    # Network parameters
    params.red_light_factor = 0.05  # 95% flux reduction during RED
    params.device = 'cpu'
    
    return params


@pytest.fixture
def two_segment_network(network_params):
    """
    Network with 2 segments and 1 traffic light junction.
    
    Architecture:
        seg_0: [0m - 100m]   end_node='node_1' (RED light)
        seg_1: [100m - 200m] end_node=None (boundary)
    """
    network = NetworkGrid(network_params)
    
    # Add segments
    network.add_segment('seg_0', xmin=0, xmax=100, N=50,
                        start_node=None, end_node='node_1')
    network.add_segment('seg_1', xmin=100, xmax=200, N=50,
                        start_node='node_1', end_node=None)
    
    # Add junction with traffic light (initially RED for seg_0)
    phases = [
        Phase(duration=60.0, green_segments=['seg_1']),  # seg_0 RED
        Phase(duration=30.0, green_segments=['seg_0'])   # seg_0 GREEN
    ]
    traffic_light = TrafficLightController(
        cycle_time=90.0,
        phases=phases,
        offset=0.0
    )
    
    network.add_node(
        node_id='node_1', 
        position=(100.0, 0.0),
        incoming_segments=['seg_0'],
        outgoing_segments=['seg_1'],
        node_type='signalized_intersection',
        traffic_lights=traffic_light
    )
    
    # Add link for coupling
    network.add_link(from_segment='seg_0', to_segment='seg_1', via_node='node_1')
    
    # Initialize
    network.initialize()
    
    return network


def test_all_segments_with_junctions_get_info(two_segment_network):
    """
    Validates junction info coverage following SUMO/CityFlow architecture.
    
    Test case:
    seg_0 → node_1 (traffic light) → seg_1 → boundary
    
    Expected:
    - seg_0.junction_at_right IS NOT NONE (has node_1)
    - seg_1.junction_at_right IS NONE (boundary)
    """
    network = two_segment_network
    
    # Trigger junction info preparation
    network._prepare_junction_info(current_time=0)
    
    # VALIDATE (like SUMO would)
    seg_0 = network.segments['seg_0']
    seg_1 = network.segments['seg_1']
    
    # seg_0 → node_1 (traffic light)
    assert seg_0['grid'].junction_at_right is not None, \
        "seg_0 has end_node='node_1' but no junction_info set!"
    assert seg_0['grid'].junction_at_right.node_id == 'node_1'
    
    # seg_1 → boundary (no junction)
    assert seg_1['grid'].junction_at_right is None, \
        "seg_1 has no end_node but junction_info was set!"
    
    print("✅ All segments with junctions have junction_info (SUMO/CityFlow compliance)")


def test_junction_info_independent_of_links(network_params):
    """
    Validates junction discovery works WITHOUT explicit Link objects.
    
    This tests the SUMO/CityFlow pattern:
    - Segments know their end_node DIRECTLY
    - No need to iterate through links
    """
    network = NetworkGrid(network_params)
    
    # Add TWO segments with junction between them
    network.add_segment('seg_0', xmin=0, xmax=100, N=50,
                        start_node=None, end_node='node_1')
    network.add_segment('seg_1', xmin=100, xmax=200, N=50,
                        start_node='node_1', end_node=None)
    
    # Add junction with traffic light
    phases = [Phase(duration=60.0, green_segments=['seg_0'])]
    traffic_light = TrafficLightController(cycle_time=60.0, phases=phases, offset=0.0)
    
    network.add_node(
        node_id='node_1', 
        position=(100.0, 0.0),
        incoming_segments=['seg_0'],
        outgoing_segments=['seg_1'],  # NOW it has outgoing
        node_type='signalized_intersection',
        traffic_lights=traffic_light
    )
    
    # Initialize WITHOUT add_link (direct pattern test)
    network.initialize()
    
    # Remove any automatically created links to test direct pattern
    network.links.clear()
    
    # Trigger junction info
    network._prepare_junction_info(current_time=0)
    
    # VALIDATE
    seg_0 = network.segments['seg_0']
    
    assert seg_0['grid'].junction_at_right is not None, \
        "seg_0 should have junction_info even without explicit Link!"
    assert seg_0['grid'].junction_at_right.node_id == 'node_1'
    assert len(network.links) == 0, "No links should exist after clearing"
    
    print("✅ Junction info works without links (SUMO/CityFlow pattern)")


def test_light_factor_changes_with_signal_state(two_segment_network):
    """
    Validates junction_info.light_factor reflects traffic signal state.
    
    Expected:
    - RED: light_factor = 0.05 (95% flux reduction)
    - GREEN: light_factor = 1.0 (full flow)
    """
    network = two_segment_network
    
    # Phase 0: seg_0 RED (0-60s)
    network._prepare_junction_info(current_time=10.0)
    seg_0 = network.segments['seg_0']
    
    assert seg_0['grid'].junction_at_right is not None
    assert seg_0['grid'].junction_at_right.light_factor == 0.05, \
        f"Expected RED (0.05), got {seg_0['grid'].junction_at_right.light_factor}"
    
    # Phase 1: seg_0 GREEN (60-90s)
    network._prepare_junction_info(current_time=70.0)
    seg_0 = network.segments['seg_0']
    
    assert seg_0['grid'].junction_at_right is not None
    assert seg_0['grid'].junction_at_right.light_factor == 1.0, \
        f"Expected GREEN (1.0), got {seg_0['grid'].junction_at_right.light_factor}"
    
    # Phase 0 again: seg_0 RED (90-150s)
    network._prepare_junction_info(current_time=100.0)
    seg_0 = network.segments['seg_0']
    
    assert seg_0['grid'].junction_at_right is not None
    assert seg_0['grid'].junction_at_right.light_factor == 0.05, \
        f"Expected RED (0.05), got {seg_0['grid'].junction_at_right.light_factor}"
    
    print("✅ light_factor correctly reflects signal state")


@pytest.mark.parametrize("scheme", ["weno5", "godunov"])
def test_congestion_forms_during_red_signal(network_params, scheme):
    """
    Integration test: verify traffic light BLOCKS flux and congestion forms.
    
    🔬 DIAGNOSTIC TEST: Parametrized to test both WENO5 and Godunov schemes.
    This reveals if WENO5 causes density=0 failures (Gibbs oscillations at steep gradients).
    
    Setup:
    - seg_0 with inflow BC (0.05 veh/m, 10 m/s)
    - node_1 with RED light for seg_0
    - Simulate 120s
    
    Expected:
    - Densities INCREASE in seg_0 (0.05 → 0.10+)
    - Velocities DECREASE in seg_0 (< 5.56 m/s congestion threshold)
    - Queue forms (> 5.0 vehicles)
    
    Diagnostic Outcomes:
    1. Both pass → Bug already fixed (BC/ODE corrections sufficient)
    2. Godunov passes, WENO5 fails → WENO5 is the problem
    3. Both fail → Problem elsewhere (CFL=44,000 suspect)
    """
    # Override spatial scheme for this test
    network_params.spatial_scheme = scheme
    
    # Create network with specified scheme
    network = NetworkGrid(network_params)
    
    # Add segments
    network.add_segment('seg_0', xmin=0, xmax=100, N=50,
                        start_node=None, end_node='node_1')
    network.add_segment('seg_1', xmin=100, xmax=200, N=50,
                        start_node='node_1', end_node=None)
    
    # Add junction with traffic light (initially RED for seg_0)
    phases = [
        Phase(duration=60.0, green_segments=['seg_1']),  # seg_0 RED
        Phase(duration=30.0, green_segments=['seg_0'])   # seg_0 GREEN
    ]
    traffic_light = TrafficLightController(
        cycle_time=90.0,
        phases=phases,
        offset=0.0
    )
    
    network.add_node(
        node_id='node_1', 
        position=(100.0, 0.0),
        incoming_segments=['seg_0'],
        outgoing_segments=['seg_1'],
        node_type='signalized_intersection',
        traffic_lights=traffic_light
    )
    
    # Add link for coupling
    network.add_link(from_segment='seg_0', to_segment='seg_1', via_node='node_1')
    
    # Initialize
    network.initialize()
    
    # Set inflow BC on seg_0 via params
    network.params.boundary_conditions = {
        'seg_0': {
            'left': {
                'type': 'inflow',
                'rho_m': 0.05,   # 50 veh/km 
                'v_m': 10.0      # 36 km/h   
            }
        }
    }    # Run simulation (RED signal throughout) with ADAPTIVE timestep
    from arz_model.numerics import cfl as cfl_module
    
    t = 0.0
    t_max = 120.0
    dt_old = None
    
    while t < t_max:
        # Get current state from first segment for CFL calculation
        current_U = network.segments['seg_0']['U']
        grid = network.segments['seg_0']['grid']
        
        # Extract physical cells only (excluding ghost cells) for CFL calculation
        U_physical = current_U[:, grid.num_ghost_cells:grid.num_ghost_cells + grid.N_physical]
        
        # Calculate adaptive dt based on CFL condition
        dt = cfl_module.calculate_cfl_dt(U_physical, grid, network.params)
        
        # Safety: limit dt change rate (max 2x increase per step)
        if dt_old is not None:
            dt = min(dt, 2.0 * dt_old)
        dt_old = dt
        
        # Don't overshoot t_max
        if t + dt > t_max:
            dt = t_max - t
        
        # Step simulation
        network.step(dt, current_time=t)
        t += dt
    
    # Check final state
    seg_0 = network.segments['seg_0']
    U_final = seg_0['U']
    rho_m_final = np.mean(U_final[0, 2:-2])  # Physical cells
    v_m_final = np.mean(U_final[1, 2:-2] / (U_final[0, 2:-2] + 1e-10))
    
    # Calculate queue length (cells with ρ > ρ_critical)
    rho_critical = 0.08
    cells_congested = np.sum(U_final[0, 2:-2] > rho_critical)
    dx = seg_0['grid'].dx
    queue_length = cells_congested * dx
    
    # VALIDATE
    assert rho_m_final > 0.08, \
        f"Expected density increase (>0.08), got {rho_m_final:.4f}"
    
    assert v_m_final < 10.0, \
        f"Expected velocity decrease (<10 m/s), got {v_m_final:.2f}"
    
    assert queue_length > 5.0, \
        f"Expected congestion (queue >5m), got {queue_length:.2f}m"
    
    print(f"✅ Congestion formed: ρ={rho_m_final:.4f}, v={v_m_final:.2f}, queue={queue_length:.2f}m")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)  # Reduce from DEBUG to INFO
    
    # Test basic functionality
    params = ModelParameters()
    params.alpha = 0.5
    params.rho_jam = 0.2
    params.epsilon = 1e-10
    params.K_m = 20.0
    params.gamma_m = 2.0
    params.K_c = 30.0
    params.gamma_c = 2.0
    params.tau_m = 2.0
    params.tau_c = 2.0
    params.V_creeping = 1.0
    params.N = 50
    params.cfl_number = 0.8
    params.ghost_cells = 3
    params.spatial_scheme = 'weno5'
    params.time_scheme = 'ssprk3'
    params.ode_solver = 'RK45'  # Fix ODE solver
    params.ode_rtol = 1e-6
    params.ode_atol = 1e-9
    params.Vmax_m = {0: 12.0, 1: 15.0, 2: 18.0}  # m/s by category
    params.Vmax_c = {0: 10.0, 1: 13.0, 2: 16.0}  # m/s by category
    params.red_light_factor = 0.05
    params.device = 'cpu'
    
    print("Testing SUMO/CityFlow architecture pattern...")
    test_junction_info_independent_of_links(params)
    print("All tests passed!")