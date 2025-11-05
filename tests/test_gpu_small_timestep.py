"""
Test with GPU and small timestep to stabilize inflow BC instability.

Based on user suggestion: "avec gpu, on pourra régler l'instabilité et avec de plus petits dt"
"""
import pytest
import numpy as np
from arz_model.network.network_grid import NetworkGrid
from arz_model.core.parameters import ModelParameters
from arz_model.core.traffic_lights import TrafficLightController, Phase
from arz_model.core import physics


@pytest.mark.parametrize("spatial_scheme", ["weno5", "godunov"])
def test_gpu_small_dt_inflow_stability(spatial_scheme):
    """
    Test congestion formation with GPU + small timestep.
    
    Configuration:
    - Device: GPU (if available, else skip)
    - Timestep: dt = 0.0001s (10x smaller than standard 0.001s)
    - BC: v_m = 10.0 m/s (high velocity, known to cause instability on CPU)
    - Duration: 15 seconds
    - Expected: Stable velocity (v_max < 20 m/s), congestion forms (rho > 0.08)
    """
    # Try to use GPU if available
    try:
        import cupy as cp
        device = 'gpu'
        print("[TEST] GPU (CuPy) available - using GPU mode!")
    except ImportError:
        device = 'cpu'
        print("[TEST] GPU not available - using CPU with small dt")
    
    # Create parameters directly (same as other tests)
    params = ModelParameters()
    # Basic physical parameters
    params.alpha = 0.5
    params.rho_jam = 1.0  # veh/m
    params.epsilon = 1e-10
    params.K_m = 20.0
    params.gamma_m = 2.0
    params.K_c = 30.0
    params.gamma_c = 2.0
    params.tau_m = 2.0
    params.tau_c = 2.0
    params.V_creeping = 1.0
    
    # Numerical parameters
    params.N = 50
    params.cfl_number = 0.8
    params.ghost_cells = 3
    params.spatial_scheme = spatial_scheme
    params.time_scheme = 'ssprk3'
    params.ode_solver = 'RK45'
    params.ode_rtol = 1e-6
    params.ode_atol = 1e-9
    
    # Speed parameters
    params.Vmax_m = {0: 12.0, 1: 15.0, 2: 18.0}
    params.Vmax_c = {0: 10.0, 1: 13.0, 2: 16.0}
    
    # Network parameters
    params.red_light_factor = 0.05  # 95% blocking
    params.device = device  # ⚡ GPU or CPU
    
    # Create network
    network = NetworkGrid(params)
    network.add_segment('seg_0', xmin=0, xmax=100, N=50,
                        start_node=None, end_node='node_1')
    network.add_segment('seg_1', xmin=100, xmax=200, N=50,
                        start_node='node_1', end_node=None)
    
    # Add junction
    phases = [Phase(duration=60.0, green_segments=['seg_1'])]
    traffic_light = TrafficLightController(cycle_time=60.0, phases=phases, offset=0.0)
    
    network.add_node(
        node_id='node_1', 
        position=(100.0, 0.0),
        incoming_segments=['seg_0'],
        outgoing_segments=['seg_1'],
        node_type='signalized_intersection',
        traffic_lights=traffic_light
    )
    
    network.add_link(from_segment='seg_0', to_segment='seg_1', via_node='node_1')
    
    # ⚡ HIGH VELOCITY BC (known to cause CPU instability)
    network.params.boundary_conditions = {
        'seg_0': {
            'left': {
                'type': 'inflow',
                'rho_m': 0.15,
                'v_m': 10.0  # HIGH VELOCITY - CPU fails at this
            }
        }
    }
    
    network.initialize()
    
    # ⚡ WARM START with equilibrium
    rho_m_init = 0.12  # 80% of BC density
    for seg_id, segment in network.segments.items():
        grid = segment['grid']
        U = segment['U']
        
        rho_m_arr = np.full(grid.N_physical, rho_m_init)
        rho_c_arr = np.zeros(grid.N_physical)
        R_local = grid.road_quality[grid.physical_cell_indices]
        
        v_m_eq, v_c_eq = physics.calculate_equilibrium_speed(
            rho_m_arr, rho_c_arr, R_local, params
        )
        
        p_m, p_c = physics.calculate_pressure(
            rho_m_arr, rho_c_arr,
            params.alpha, params.rho_jam, params.epsilon,
            params.K_m, params.gamma_m, params.K_c, params.gamma_c
        )
        
        U[0, grid.physical_cell_indices] = rho_m_init
        U[1, grid.physical_cell_indices] = v_m_eq + p_m
        U[2, grid.physical_cell_indices] = 0.0
        U[3, grid.physical_cell_indices] = 0.0
    
    # ⚡ SMALL TIMESTEP SIMULATION
    t = 0.0
    t_max = 15.0
    dt = 0.0001  # 10x SMALLER than standard (0.001s)
    step = 0
    
    print(f"\n[GPU SMALL DT TEST] {spatial_scheme.upper()}")
    print(f"  Device: {params.device}")
    print(f"  dt: {dt}s (10x smaller)")
    print(f"  BC: rho_m=0.15, v_m=10.0 m/s (HIGH VELOCITY)")
    print(f"  Duration: {t_max}s")
    
    # Track max velocity
    max_v_observed = 0.0
    
    while t < t_max:
        # Use fixed small timestep (no CFL adaptation)
        network.step(dt, t)
        t += dt
        step += 1
        
        # Monitor every 1 second
        if step % 10000 == 0:  # 10000 steps = 1 second with dt=0.0001
            U_seg0 = network.segments['seg_0']['U']
            grid_seg0 = network.segments['seg_0']['grid']
            
            # Get physical cells (remove ghost cells)
            g = grid_seg0.num_ghost_cells
            rho_m = U_seg0[0, g:-g]
            w_m = U_seg0[1, g:-g]
            
            # Calculate physical velocity
            p_m, _ = physics.calculate_pressure(
                rho_m, np.zeros_like(rho_m),
                params.alpha, params.rho_jam, params.epsilon,
                params.K_m, params.gamma_m, params.K_c, params.gamma_c
            )
            v_m = w_m - p_m
            
            rho_m_mean = np.mean(rho_m)
            v_m_mean = np.mean(v_m)
            v_m_max = np.max(v_m)
            
            max_v_observed = max(max_v_observed, v_m_max)
            
            print(f"  t={t:6.2f}s: rho_m={rho_m_mean:.4f}, v_m_mean={v_m_mean:.2f}, v_m_max={v_m_max:.2f}")
            
            # Early exit if explosion detected
            if v_m_max > 100.0:
                print(f"\n❌ EXPLOSION DETECTED at t={t:.2f}s: v_max={v_m_max:.2f} m/s")
                pytest.fail(f"[{spatial_scheme}] Velocity explosion with GPU + small dt: v_max={v_m_max:.2f}")
    
    # Final state
    U_final = network.segments['seg_0']['U']
    grid_final = network.segments['seg_0']['grid']
    g = grid_final.num_ghost_cells
    rho_m_final = np.mean(U_final[0, g:-g])
    w_m_final = U_final[1, g:-g]
    p_m_final, _ = physics.calculate_pressure(
        U_final[0, g:-g], np.zeros_like(U_final[0, g:-g]),
        params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c
    )
    v_m_final = w_m_final - p_m_final
    v_m_final_mean = np.mean(v_m_final)
    v_m_final_max = np.max(v_m_final)
    
    print(f"\n[FINAL STATE]")
    print(f"  rho_m_final: {rho_m_final:.4f}")
    print(f"  v_m_final (mean): {v_m_final_mean:.2f} m/s")
    print(f"  v_m_final (max): {v_m_final_max:.2f} m/s")
    print(f"  max_v_observed: {max_v_observed:.2f} m/s")
    
    # ✅ SUCCESS CRITERIA
    # 1. No velocity explosion (v_max < 20 m/s throughout)
    # 2. Congestion forms (rho_m > 0.08)
    
    print(f"\n[VALIDATION]")
    print(f"  Velocity stability: {'✅ PASS' if max_v_observed < 20.0 else '❌ FAIL'} (max_v={max_v_observed:.2f} < 20.0)")
    print(f"  Congestion formation: {'✅ PASS' if rho_m_final > 0.08 else '❌ FAIL'} (rho={rho_m_final:.4f} > 0.08)")
    
    assert max_v_observed < 20.0, \
        f"[{spatial_scheme}] Velocity explosion with GPU + small dt: max_v={max_v_observed:.2f} m/s (expected <20)"
    
    assert rho_m_final > 0.08, \
        f"[{spatial_scheme}] Insufficient congestion with GPU + small dt: rho={rho_m_final:.4f} (expected >0.08)"
    
    print(f"\n🎉 [{spatial_scheme.upper()}] GPU + small dt TEST PASSED!")
    print(f"   Stability: ✅ v_max={max_v_observed:.2f} m/s")
    print(f"   Congestion: ✅ rho={rho_m_final:.4f}")
