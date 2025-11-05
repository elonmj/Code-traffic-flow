"""
Component Isolation Diagnostic Test
====================================

Isolates ODE vs FVM to identify root cause of velocity explosion.

Test Strategy:
1. Test ODE ONLY (disable hyperbolic step)
2. Test FVM ONLY (disable ODE step)  
3. Test FULL Strang splitting with detailed logging
4. Compare results to identify culprit

Expected Behavior:
- If ODE explodes alone → RK45 solver issue
- If FVM explodes alone → WENO5/Riemann/SSP-RK3 issue
- If both stable alone → Strang splitting coupling issue
"""
import pytest
import numpy as np
from arz_model.network.network_grid import NetworkGrid
from arz_model.core.traffic_lights import TrafficLightController, Phase
from arz_model.core.parameters import ModelParameters
from arz_model.core import physics
from arz_model.numerics import time_integration


@pytest.fixture
def diagnostic_params():
    """Minimal parameters for diagnostic tests."""
    params = ModelParameters()
    params.alpha = 0.5
    params.rho_jam = 1.0
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
    params.ode_solver = 'RK45'
    params.ode_rtol = 1e-6
    params.ode_atol = 1e-9
    params.Vmax_m = {0: 12.0, 1: 15.0, 2: 18.0}
    params.Vmax_c = {0: 10.0, 1: 13.0, 2: 16.0}
    params.red_light_factor = 0.05
    params.device = 'cpu'
    return params


def create_diagnostic_network(params):
    """Create simple 2-segment network with inflow BC and blocked junction."""
    network = NetworkGrid(params)
    
    network.add_segment('seg_0', xmin=0, xmax=100, N=50,
                        start_node=None, end_node='node_1')
    network.add_segment('seg_1', xmin=100, xmax=200, N=50,
                        start_node='node_1', end_node=None)
    
    phases = [Phase(duration=60.0, green_segments=['seg_1'])]  # seg_0 RED
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
    
    network.params.boundary_conditions = {
        'seg_0': {
            'left': {
                'type': 'inflow',
                'rho_m': 0.15,
                'v_m': 3.0
            }
        }
    }
    
    network.initialize()
    return network


def initialize_equilibrium(network):
    """Initialize network with equilibrium warm start."""
    for seg_id, segment in network.segments.items():
        grid = segment['grid']
        U = segment['U']
        
        rho_m_init = 0.12
        rho_c_init = 0.0
        
        rho_m_arr = np.full(grid.N_physical, rho_m_init)
        rho_c_arr = np.full(grid.N_physical, rho_c_init)
        R_local = grid.road_quality[grid.physical_cell_indices]
        
        v_m_eq, v_c_eq = physics.calculate_equilibrium_speed(
            rho_m_arr, rho_c_arr, R_local, network.params,
            V0_m_override=segment.get('V0_m'), V0_c_override=segment.get('V0_c')
        )
        
        p_m, p_c = physics.calculate_pressure(
            rho_m_arr, rho_c_arr,
            network.params.alpha, network.params.rho_jam, network.params.epsilon,
            network.params.K_m, network.params.gamma_m,
            network.params.K_c, network.params.gamma_c
        )
        w_m_eq = v_m_eq + p_m
        w_c_eq = v_c_eq + p_c
        
        U[0, grid.physical_cell_indices] = rho_m_init
        U[1, grid.physical_cell_indices] = w_m_eq
        U[2, grid.physical_cell_indices] = rho_c_init
        U[3, grid.physical_cell_indices] = w_c_eq


def get_velocity_stats(U, grid):
    """Extract velocity statistics from state array."""
    g = grid.num_ghost_cells
    N = grid.N_physical
    
    rho_m = U[0, g:g+N]
    w_m = U[1, g:g+N]
    
    # Calculate pressure
    rho_c = U[2, g:g+N]
    p_m, _ = physics.calculate_pressure(
        rho_m, rho_c,
        grid.params.alpha if hasattr(grid, 'params') else 0.5,
        grid.params.rho_jam if hasattr(grid, 'params') else 1.0,
        1e-10, 20.0, 2.0, 30.0, 2.0
    )
    
    v_m = w_m - p_m
    
    return {
        'v_mean': np.mean(v_m),
        'v_max': np.max(v_m),
        'v_min': np.min(v_m),
        'rho_mean': np.mean(rho_m),
        'rho_max': np.max(rho_m)
    }


@pytest.mark.diagnostic
def test_ode_only_no_hyperbolic(diagnostic_params):
    """
    TEST 1: ODE Step ONLY (disable hyperbolic transport)
    
    Tests if RK45 solver for source term S = (Ve - v)/τ is stable alone.
    
    Method: Apply only ODE step repeatedly, no spatial transport.
    Expected: Should converge to equilibrium v → Ve.
    """
    print("\n" + "="*80)
    print("TEST 1: ODE ONLY (No Hyperbolic Transport)")
    print("="*80)
    
    network = create_diagnostic_network(diagnostic_params)
    initialize_equilibrium(network)
    
    seg_0 = network.segments['seg_0']
    grid = seg_0['grid']
    U = seg_0['U']
    
    dt = 0.01  # 10ms timestep
    t_max = 15.0  # 15s simulation
    t = 0.0
    step = 0
    
    print(f"\nInitial state:")
    stats = get_velocity_stats(U, grid)
    print(f"  v_mean={stats['v_mean']:.2f} m/s, v_max={stats['v_max']:.2f}, rho_mean={stats['rho_mean']:.4f}")
    
    while t < t_max:
        # ONLY ODE step (no hyperbolic transport)
        U_new = time_integration.solve_ode_step_cpu(U, dt, grid, diagnostic_params)
        U[:] = U_new
        
        t += dt
        step += 1
        
        # Log every 1.0s
        if step % 100 == 0:
            stats = get_velocity_stats(U, grid)
            print(f"t={t:6.2f}s: v_mean={stats['v_mean']:8.2f} m/s, v_max={stats['v_max']:8.2f}, rho_mean={stats['rho_mean']:.4f}")
            
            # Check for explosion
            if stats['v_max'] > 100.0:
                print(f"\n⚠️ EXPLOSION DETECTED in ODE step at t={t:.2f}s!")
                print(f"   v_max = {stats['v_max']:.2f} m/s")
                pytest.fail(f"ODE solver exploded: v_max={stats['v_max']:.2f} m/s")
    
    # Final check
    stats = get_velocity_stats(U, grid)
    print(f"\nFinal state (t={t_max}s):")
    print(f"  v_mean={stats['v_mean']:.2f} m/s, v_max={stats['v_max']:.2f}, rho_mean={stats['rho_mean']:.4f}")
    
    assert stats['v_max'] < 50.0, f"ODE solver unstable: v_max={stats['v_max']:.2f} m/s"
    print("✅ ODE solver STABLE when isolated")


@pytest.mark.diagnostic
def test_fvm_only_no_ode(diagnostic_params):
    """
    TEST 2: FVM Hyperbolic ONLY (disable ODE relaxation)
    
    Tests if WENO5 + SSP-RK3 + Riemann solver is stable alone.
    
    Method: Apply only hyperbolic step (∂U/∂t + ∂F/∂x = 0), no source term.
    Expected: Should preserve initial state with boundary inflow.
    """
    print("\n" + "="*80)
    print("TEST 2: FVM ONLY (No ODE Relaxation)")
    print("="*80)
    
    network = create_diagnostic_network(diagnostic_params)
    initialize_equilibrium(network)
    
    seg_0 = network.segments['seg_0']
    grid = seg_0['grid']
    U = seg_0['U']
    
    dt = 0.01  # 10ms timestep
    t_max = 15.0  # 15s simulation
    t = 0.0
    step = 0
    
    print(f"\nInitial state:")
    stats = get_velocity_stats(U, grid)
    print(f"  v_mean={stats['v_mean']:.2f} m/s, v_max={stats['v_max']:.2f}, rho_mean={stats['rho_mean']:.4f}")
    
    while t < t_max:
        # ONLY Hyperbolic step (no ODE relaxation)
        U_new = time_integration.solve_hyperbolic_step_ssprk3(
            U, dt, grid, diagnostic_params, current_bc_params=None
        )
        U[:] = U_new
        
        t += dt
        step += 1
        
        # Log every 1.0s
        if step % 100 == 0:
            stats = get_velocity_stats(U, grid)
            print(f"t={t:6.2f}s: v_mean={stats['v_mean']:8.2f} m/s, v_max={stats['v_max']:8.2f}, rho_mean={stats['rho_mean']:.4f}")
            
            # Check for explosion
            if stats['v_max'] > 100.0:
                print(f"\n⚠️ EXPLOSION DETECTED in FVM step at t={t:.2f}s!")
                print(f"   v_max = {stats['v_max']:.2f} m/s")
                pytest.fail(f"FVM hyperbolic exploded: v_max={stats['v_max']:.2f} m/s")
    
    # Final check
    stats = get_velocity_stats(U, grid)
    print(f"\nFinal state (t={t_max}s):")
    print(f"  v_mean={stats['v_mean']:.2f} m/s, v_max={stats['v_max']:.2f}, rho_mean={stats['rho_mean']:.4f}")
    
    assert stats['v_max'] < 50.0, f"FVM hyperbolic unstable: v_max={stats['v_max']:.2f} m/s"
    print("✅ FVM hyperbolic STABLE when isolated")


@pytest.mark.diagnostic
def test_strang_splitting_with_detailed_logging(diagnostic_params):
    """
    TEST 3: Full Strang Splitting with Detailed Component Logging
    
    Tests complete ODE(dt/2) + Hyperbolic(dt) + ODE(dt/2) sequence.
    Logs velocity after EACH substep to pinpoint explosion moment.
    
    Method: Run full Strang splitting, log intermediate states.
    Expected: Identify which substep causes explosion.
    """
    print("\n" + "="*80)
    print("TEST 3: Full Strang Splitting with Detailed Logging")
    print("="*80)
    
    network = create_diagnostic_network(diagnostic_params)
    initialize_equilibrium(network)
    
    seg_0 = network.segments['seg_0']
    grid = seg_0['grid']
    U = seg_0['U'].copy()
    
    dt = 0.01  # 10ms timestep
    t_max = 15.0  # 15s simulation
    t = 0.0
    step = 0
    
    print(f"\nInitial state:")
    stats = get_velocity_stats(U, grid)
    print(f"  v_mean={stats['v_mean']:.2f} m/s, v_max={stats['v_max']:.2f}, rho_mean={stats['rho_mean']:.4f}")
    
    while t < t_max:
        step += 1
        U_start = U.copy()
        
        # --- Substep 1: ODE(dt/2) ---
        U_after_ode1 = time_integration.solve_ode_step_cpu(U, dt/2.0, grid, diagnostic_params)
        stats_ode1 = get_velocity_stats(U_after_ode1, grid)
        
        # --- Substep 2: Hyperbolic(dt) ---
        U_after_hyp = time_integration.solve_hyperbolic_step_ssprk3(
            U_after_ode1, dt, grid, diagnostic_params, current_bc_params=None
        )
        stats_hyp = get_velocity_stats(U_after_hyp, grid)
        
        # --- Substep 3: ODE(dt/2) ---
        U_after_ode2 = time_integration.solve_ode_step_cpu(U_after_hyp, dt/2.0, grid, diagnostic_params)
        stats_ode2 = get_velocity_stats(U_after_ode2, grid)
        
        U[:] = U_after_ode2
        t += dt
        
        # Log every 1.0s with substep details
        if step % 100 == 0:
            print(f"\nt={t:6.2f}s (step {step}):")
            print(f"  After ODE1:  v_max={stats_ode1['v_max']:8.2f} m/s, rho_mean={stats_ode1['rho_mean']:.4f}")
            print(f"  After HYP:   v_max={stats_hyp['v_max']:8.2f} m/s, rho_mean={stats_hyp['rho_mean']:.4f}")
            print(f"  After ODE2:  v_max={stats_ode2['v_max']:8.2f} m/s, rho_mean={stats_ode2['rho_mean']:.4f}")
            
            # Check which substep exploded
            if stats_ode1['v_max'] > 100.0:
                print(f"\n🔥 EXPLOSION in ODE1 substep!")
                pytest.fail(f"First ODE step exploded: v_max={stats_ode1['v_max']:.2f} m/s")
            elif stats_hyp['v_max'] > 100.0:
                print(f"\n🔥 EXPLOSION in HYPERBOLIC substep!")
                pytest.fail(f"Hyperbolic step exploded: v_max={stats_hyp['v_max']:.2f} m/s")
            elif stats_ode2['v_max'] > 100.0:
                print(f"\n🔥 EXPLOSION in ODE2 substep!")
                pytest.fail(f"Second ODE step exploded: v_max={stats_ode2['v_max']:.2f} m/s")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"Final state (t={t_max}s):")
    stats = get_velocity_stats(U, grid)
    print(f"  v_mean={stats['v_mean']:.2f} m/s, v_max={stats['v_max']:.2f}, rho_mean={stats['rho_mean']:.4f}")
    
    if stats['v_max'] > 50.0:
        print(f"⚠️ Strang splitting produced unrealistic velocities: v_max={stats['v_max']:.2f} m/s")
        pytest.fail(f"Strang splitting unstable: v_max={stats['v_max']:.2f} m/s")
    
    print("✅ Strang splitting completed without explosion")


@pytest.mark.diagnostic
def test_compare_tau_sensitivity(diagnostic_params):
    """
    TEST 4: Relaxation Time Sensitivity Analysis
    
    Tests if ODE stiffness (controlled by τ) affects stability.
    
    Method: Run Strang splitting with different τ_m values.
    Expected: Larger τ (slower relaxation) should be more stable.
    """
    print("\n" + "="*80)
    print("TEST 4: Relaxation Time (τ) Sensitivity Analysis")
    print("="*80)
    
    tau_values = [0.5, 1.0, 2.0, 5.0, 10.0]  # Different relaxation times
    results = {}
    
    for tau_m in tau_values:
        print(f"\n--- Testing τ_m = {tau_m:.1f}s ---")
        
        params = diagnostic_params
        params.tau_m = tau_m
        params.tau_c = tau_m
        
        network = create_diagnostic_network(params)
        initialize_equilibrium(network)
        
        seg_0 = network.segments['seg_0']
        grid = seg_0['grid']
        U = seg_0['U'].copy()
        
        dt = 0.01
        t_max = 10.0  # Shorter test
        t = 0.0
        exploded = False
        
        while t < t_max and not exploded:
            U_new = time_integration.strang_splitting_step(
                U, dt, grid, params, current_bc_params=None
            )
            U[:] = U_new
            t += dt
            
            stats = get_velocity_stats(U, grid)
            if stats['v_max'] > 100.0:
                exploded = True
                results[tau_m] = {'stable': False, 't_explosion': t, 'v_max': stats['v_max']}
                print(f"  ⚠️ EXPLODED at t={t:.2f}s, v_max={stats['v_max']:.2f} m/s")
                break
        
        if not exploded:
            stats = get_velocity_stats(U, grid)
            results[tau_m] = {'stable': True, 't_final': t_max, 'v_max': stats['v_max']}
            print(f"  ✅ STABLE: v_max={stats['v_max']:.2f} m/s at t={t_max}s")
    
    # Summary
    print(f"\n{'='*80}")
    print("Relaxation Time Sensitivity Summary:")
    print(f"{'='*80}")
    for tau_m, result in results.items():
        if result['stable']:
            print(f"τ={tau_m:5.1f}s: STABLE   (v_max={result['v_max']:6.2f} m/s)")
        else:
            print(f"τ={tau_m:5.1f}s: EXPLODED at t={result['t_explosion']:.2f}s (v_max={result['v_max']:.2f} m/s)")
    
    # Analysis
    stable_count = sum(1 for r in results.values() if r['stable'])
    print(f"\nStable configurations: {stable_count}/{len(tau_values)}")
    
    if stable_count == 0:
        print("❌ CRITICAL: ALL τ values explode → ODE solver fundamentally unstable")
    elif stable_count < len(tau_values):
        print("⚠️ PARTIAL: Some τ values stable → ODE stiffness is a factor")
    else:
        print("✅ ALL STABLE: τ does not affect stability in this range")


@pytest.mark.diagnostic
def test_option1_bc_timing_modification(diagnostic_params):
    """
    OPTION 1: BC Timing Modification Test
    
    Strategy: Apply inflow BC only AFTER ODE substeps, not during hyperbolic step.
    
    Hypothesis: If BC is applied after ODE relaxation, the discontinuity should
    be smoothed by source term before hyperbolic propagation.
    
    Implementation: Manually control BC application timing in Strang sequence.
    
    Success Criteria: v_max < 20 m/s for 15s with no explosions.
    """
    print("\n" + "="*80)
    print("OPTION 1: BC Timing Modification Test")
    print("="*80)
    print("\nStrategy: Apply BC after ODE steps, skip during hyperbolic step")
    print("Expected: Reduced BC-splitting coupling error\n")
    
    network = create_diagnostic_network(diagnostic_params)
    initialize_equilibrium(network)
    
    seg_0 = network.segments['seg_0']
    grid = seg_0['grid']
    U = seg_0['U'].copy()
    
    # Get BC parameters
    bc_left = diagnostic_params.boundary_conditions.get('seg_0', {}).get('left', {})
    rho_m_bc = bc_left.get('rho_m', 0.15)
    v_m_bc = bc_left.get('v_m', 3.0)
    
    dt = 0.01
    t_max = 15.0
    t = 0.0
    step = 0
    
    print(f"Initial state:")
    stats = get_velocity_stats(U, grid)
    print(f"  v_mean={stats['v_mean']:.2f} m/s, v_max={stats['v_max']:.2f}, rho_mean={stats['rho_mean']:.4f}")
    
    max_v_observed = 0.0
    
    while t < t_max:
        step += 1
        
        # --- Substep 1: ODE(dt/2) with BC ---
        U_after_ode1 = time_integration.solve_ode_step_cpu(U, dt/2.0, grid, diagnostic_params)
        
        # Apply BC AFTER ODE step
        g = grid.num_ghost_cells
        p_m_bc, _ = physics.calculate_pressure(
            np.array([rho_m_bc]), np.array([0.0]),
            diagnostic_params.alpha, diagnostic_params.rho_jam, diagnostic_params.epsilon,
            diagnostic_params.K_m, diagnostic_params.gamma_m,
            diagnostic_params.K_c, diagnostic_params.gamma_c
        )
        w_m_bc = v_m_bc + p_m_bc[0]
        
        U_after_ode1[0, :g] = rho_m_bc
        U_after_ode1[1, :g] = w_m_bc
        U_after_ode1[2, :g] = 0.0
        U_after_ode1[3, :g] = 0.0
        
        # --- Substep 2: Hyperbolic(dt) WITHOUT BC application ---
        # Temporarily disable BC for hyperbolic step
        original_bc = diagnostic_params.boundary_conditions
        diagnostic_params.boundary_conditions = {}
        
        U_after_hyp = time_integration.solve_hyperbolic_step_ssprk3(
            U_after_ode1, dt, grid, diagnostic_params, current_bc_params=None
        )
        
        # Restore BC
        diagnostic_params.boundary_conditions = original_bc
        
        # --- Substep 3: ODE(dt/2) with BC ---
        U_after_ode2 = time_integration.solve_ode_step_cpu(U_after_hyp, dt/2.0, grid, diagnostic_params)
        
        # Apply BC AFTER second ODE step
        U_after_ode2[0, :g] = rho_m_bc
        U_after_ode2[1, :g] = w_m_bc
        U_after_ode2[2, :g] = 0.0
        U_after_ode2[3, :g] = 0.0
        
        U[:] = U_after_ode2
        t += dt
        
        # Monitor velocity
        stats = get_velocity_stats(U, grid)
        max_v_observed = max(max_v_observed, stats['v_max'])
        
        # Log every 1.0s
        if step % 100 == 0:
            print(f"t={t:6.2f}s: v_mean={stats['v_mean']:8.2f} m/s, v_max={stats['v_max']:8.2f}, max_observed={max_v_observed:8.2f}")
            
            # Check for explosion
            if stats['v_max'] > 100.0:
                print(f"\n⚠️ OPTION 1 FAILED: Explosion at t={t:.2f}s, v_max={stats['v_max']:.2f} m/s")
                pytest.fail(f"Option 1 BC timing modification failed: v_max={stats['v_max']:.2f} m/s")
    
    # Final evaluation
    print(f"\n{'='*80}")
    print(f"OPTION 1 RESULTS:")
    print(f"{'='*80}")
    stats = get_velocity_stats(U, grid)
    print(f"Final state (t={t_max}s):")
    print(f"  v_mean={stats['v_mean']:.2f} m/s")
    print(f"  v_max={stats['v_max']:.2f} m/s")
    print(f"  max_v_observed={max_v_observed:.2f} m/s")
    print(f"  rho_mean={stats['rho_mean']:.4f}")
    
    # Success criteria: v_max < 20 m/s
    if stats['v_max'] < 20.0 and max_v_observed < 20.0:
        print(f"\n✅ OPTION 1 SUCCESS: BC timing modification stabilized simulation!")
        print(f"   v_max={stats['v_max']:.2f} m/s < 20 m/s threshold")
        print(f"   No explosions detected throughout 15s simulation")
        assert True
    else:
        print(f"\n❌ OPTION 1 FAILED: v_max={stats['v_max']:.2f} m/s exceeds 20 m/s threshold")
        print(f"   Proceed to OPTION 2: Boundary Correction Function")
        pytest.fail(f"Option 1 unsuccessful: v_max={stats['v_max']:.2f} m/s > 20 m/s")


if __name__ == "__main__":
    # Run diagnostic tests
    params = diagnostic_params()
    
    print("\n" + "🔬 COMPONENT ISOLATION DIAGNOSTIC SUITE 🔬".center(80))
    print("=" * 80)
    
    try:
        test_ode_only_no_hyperbolic(params)
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
    
    try:
        test_fvm_only_no_ode(params)
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
    
    try:
        test_strang_splitting_with_detailed_logging(params)
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
    
    try:
        test_compare_tau_sensitivity(params)
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUITE COMPLETE")
    print("=" * 80)
