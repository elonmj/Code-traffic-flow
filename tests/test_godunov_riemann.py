"""
Godunov Riemann Solver Tests

Validation tests for the Godunov first-order upwind Riemann solver.
Tests include:
- Sod shock tube (academic standard)
- Inflow BC with steep gradient
- Smooth case convergence
"""

import pytest
import numpy as np
from arz_model.core.parameters import ModelParameters
from arz_model.grid.grid1d import Grid1D
from arz_model.numerics.time_integration import strang_splitting_step


def test_godunov_sod_shock_tube():
    """
    Test Sod shock tube: high density → low density discontinuity for motorcycles.
    
    Verifies:
    - Positivity preservation (ρ ≥ 0)
    - Mass conservation
    - Shock formation (strong gradient detected)
    """
    print("\n" + "="*60)
    print("Godunov Test: Sod Shock Tube")
    print("="*60)
    
    # Setup params
    params = ModelParameters()
    params.alpha = 0.5
    params.rho_jam = 0.2  # veh/m
    params.epsilon = 1e-10
    params.K_m = 20.0 / 3.6  # Convert km/h to m/s
    params.gamma_m = 2.0
    params.K_c = 30.0 / 3.6
    params.gamma_c = 2.0
    params.tau_m = 1.0
    params.tau_c = 1.0
    params.V_creeping = 0.5  # m/s
    params.Vmax_m = {2: 25.0}  # Road quality 2 -> 25 m/s
    params.Vmax_c = {2: 30.0}  # Road quality 2 -> 30 m/s
    params.spatial_scheme = 'godunov'  # KEY: Use Godunov
    params.time_scheme = 'ssprk3'
    params.ode_solver = 'RK45'  # Valid scipy method
    params.ode_rtol = 1e-6
    params.ode_atol = 1e-9
    params.device = 'cpu'
    
    # Grid
    grid = Grid1D(xmin=0, xmax=100, N=100, num_ghost_cells=3)
    grid.road_quality = np.full(grid.N_physical, 2)  # Uniform road quality
    
    # Initial conditions: discontinuity at x=50
    U = np.zeros((4, grid.N_total))
    g = grid.num_ghost_cells
    N = grid.N_physical
    
    # Left: high density (ρ=0.5 veh/m)
    rho_L = 0.05  # Lower to avoid jam
    U[0, g:g+N//2] = rho_L
    # w = v + P, with v ~ 5 m/s at this density
    p_L = params.K_m * (rho_L / params.rho_jam) ** params.gamma_m
    U[1, g:g+N//2] = 5.0 + p_L
    
    # Right: low density (ρ=0.1 veh/m)
    rho_R = 0.01
    U[0, g+N//2:g+N] = rho_R
    p_R = params.K_m * (rho_R / params.rho_jam) ** params.gamma_m
    U[1, g+N//2:g+N] = 15.0 + p_R
    
    print(f"\nInitial conditions:")
    print(f"  Left (x<50):  rho_m={rho_L:.4f} veh/m")
    print(f"  Right (x>=50): rho_m={rho_R:.4f} veh/m")
    print(f"  Total mass: {np.sum(U[0, g:g+N]) * grid.dx:.4f} vehicles")
    
    # Simulate 50 timesteps
    dt = 0.05
    print(f"\nSimulating {50} steps with dt={dt}s...")
    for i in range(50):
        U = strang_splitting_step(U, dt, grid, params)
        if (i+1) % 10 == 0:
            rho_m = U[0, g:g+N]
            print(f"  Step {i+1}: rho_m range=[{rho_m.min():.6f}, {rho_m.max():.6f}]")
    
    # Verifications
    rho_m = U[0, g:g+N]
    
    # 1. Positivity
    print(f"\n✓ Checking positivity...")
    assert np.all(rho_m >= 0), f"Negative densities detected! min(rho_m)={rho_m.min():.6f}"
    print(f"  ✅ All densities non-negative (min={rho_m.min():.6f})")
    
    # 2. Mass conservation
    print(f"\n✓ Checking mass conservation...")
    masse_final = np.sum(rho_m) * grid.dx
    masse_init = (rho_L * 50 + rho_R * 50)
    conservation_error = abs(masse_final - masse_init) / masse_init
    print(f"  Initial mass: {masse_init:.4f} vehicles")
    print(f"  Final mass:   {masse_final:.4f} vehicles")
    print(f"  Relative error: {conservation_error*100:.2f}%")
    assert conservation_error < 0.05, f"Mass not conserved! Error={conservation_error*100:.2f}%"
    print(f"  ✅ Mass conserved (error < 5%)")
    
    # 3. Shock present (strong gradient)
    print(f"\n✓ Checking shock formation...")
    gradients = np.abs(np.diff(rho_m))
    max_gradient = np.max(gradients)
    print(f"  Max gradient: {max_gradient:.6f} veh/m²")
    assert max_gradient > 0.001, f"No shock detected! max_gradient={max_gradient:.6f}"
    print(f"  ✅ Shock formed (gradient > 0.001)")
    
    print(f"\n{'='*60}")
    print(f"✅ Sod shock tube test PASSED")
    print(f"{'='*60}\n")


def test_godunov_inflow_bc_steep_gradient():
    """
    Test Godunov stability with inflow BC creating steep gradient.
    
    This is a realistic scenario: empty road + sudden inflow boundary condition.
    
    Verifies:
    - Mass accumulation from inflow
    - No excessive accumulation (stability)
    - Positivity everywhere
    """
    print("\n" + "="*60)
    print("Godunov Test: Inflow BC Steep Gradient")
    print("="*60)
    
    # Setup
    params = ModelParameters()
    params.alpha = 0.5
    params.rho_jam = 0.2
    params.epsilon = 1e-10
    params.K_m = 20.0 / 3.6
    params.gamma_m = 2.0
    params.K_c = 30.0 / 3.6
    params.gamma_c = 2.0
    params.tau_m = 1.0
    params.tau_c = 1.0
    params.V_creeping = 0.5  # m/s
    params.Vmax_m = {2: 25.0}  # Road quality 2 -> 25 m/s
    params.Vmax_c = {2: 30.0}  # Road quality 2 -> 30 m/s
    params.spatial_scheme = 'godunov'  # KEY: Use Godunov
    params.time_scheme = 'ssprk3'
    params.ode_solver = 'RK45'  # Valid scipy method
    params.ode_rtol = 1e-6
    params.ode_atol = 1e-9
    params.device = 'cpu'
    
    grid = Grid1D(xmin=0, xmax=50, N=25, num_ghost_cells=3)
    grid.road_quality = np.full(grid.N_physical, 2)  # Uniform road quality
    
    # Initial state: nearly empty
    U = np.zeros((4, grid.N_total))
    U[0, :] = params.epsilon
    U[2, :] = params.epsilon
    
    print(f"\nInitial: nearly empty road (rho ~ {params.epsilon:.2e} veh/m)")
    
    # Inflow BC: inject motorcycles at moderate density
    rho_inflow = 0.05  # veh/m
    v_inflow = 10.0  # m/s
    p_inflow = params.K_m * (rho_inflow / params.rho_jam) ** params.gamma_m
    w_inflow = v_inflow + p_inflow
    
    bc_params = {
        'left': {
            'type': 0,  # inflow
            'state': np.array([rho_inflow, w_inflow, 0.0, 0.0])
        },
        'right': {
            'type': 1  # outflow
        }
    }
    
    print(f"Inflow BC: rho_m={rho_inflow:.4f} veh/m, v={v_inflow:.1f} m/s")
    
    # Simulate 100 timesteps
    dt = 0.1
    print(f"\nSimulating {100} steps with dt={dt}s...")
    for i in range(100):
        U = strang_splitting_step(U, dt, grid, params, bc_params)
        if (i+1) % 20 == 0:
            g = grid.num_ghost_cells
            rho_first = U[0, g]
            rho_max = U[0, g:g+grid.N_physical].max()
            print(f"  Step {i+1}: rho_first={rho_first:.6f}, rho_max={rho_max:.6f}")
    
    # Verifications
    g = grid.num_ghost_cells
    rho_m_first_cell = U[0, g]
    rho_m_physical = U[0, g:g+grid.N_physical]
    
    # 1. Mass accumulated from inflow
    print(f"\n✓ Checking mass accumulation...")
    print(f"  First cell density: {rho_m_first_cell:.6f} veh/m")
    assert rho_m_first_cell > 0.001, (
        f"No accumulation! rho_first={rho_m_first_cell:.6f}"
    )
    print(f"  ✅ Mass accumulated (rho > 0.001)")
    
    # 2. No excessive accumulation (stability check)
    print(f"\n✓ Checking stability...")
    print(f"  Max density: {rho_m_physical.max():.6f} veh/m")
    assert rho_m_physical.max() < params.rho_jam, (
        f"Excessive accumulation! rho_max={rho_m_physical.max():.6f} > rho_jam={params.rho_jam}"
    )
    print(f"  ✅ No blow-up (rho_max < rho_jam)")
    
    # 3. Positivity everywhere
    print(f"\n✓ Checking positivity...")
    assert np.all(rho_m_physical >= 0), "Negative densities!"
    print(f"  ✅ All densities non-negative (min={rho_m_physical.min():.6f})")
    
    print(f"\n{'='*60}")
    print(f"✅ Inflow BC test PASSED")
    print(f"{'='*60}\n")


@pytest.mark.parametrize("scheme", ["weno5", "godunov"])
def test_smooth_case_convergence(scheme):
    """
    Test that both WENO5 and Godunov give reasonable results on smooth case.
    
    On smooth Gaussian initial condition, both schemes should:
    - Conserve mass
    - Preserve positivity
    - Stay bounded
    
    Note: Godunov (1st order) will be more diffusive than WENO5 (5th order),
    but both should remain stable and physically correct.
    """
    print("\n" + "="*60)
    print(f"Smooth Case Test: {scheme.upper()}")
    print("="*60)
    
    # Setup
    params = ModelParameters()
    params.alpha = 0.5
    params.rho_jam = 0.2
    params.epsilon = 1e-10
    params.K_m = 20.0 / 3.6
    params.gamma_m = 2.0
    params.K_c = 30.0 / 3.6
    params.gamma_c = 2.0
    params.tau_m = 1.0
    params.tau_c = 1.0
    params.V_creeping = 0.5  # m/s
    params.Vmax_m = {2: 25.0}  # Road quality 2 -> 25 m/s
    params.Vmax_c = {2: 30.0}  # Road quality 2 -> 30 m/s
    params.spatial_scheme = scheme  # TEST BOTH
    params.time_scheme = 'ssprk3'
    params.ode_solver = 'RK45'  # Valid scipy method
    params.ode_rtol = 1e-6
    params.ode_atol = 1e-9
    params.device = 'cpu'
    
    grid = Grid1D(xmin=0, xmax=100, N=50, num_ghost_cells=3)
    grid.road_quality = np.full(grid.N_physical, 2)  # Uniform road quality
    
    # Smooth initial condition: Gaussian
    x_centers = grid._cell_centers[grid.num_ghost_cells:grid.num_ghost_cells+grid.N_physical]
    rho_smooth = 0.02 * np.exp(-((x_centers - 50)**2) / 100)
    
    U = np.zeros((4, grid.N_total))
    g = grid.num_ghost_cells
    U[0, g:g+grid.N_physical] = rho_smooth
    # w = v + p, with uniform velocity
    for j in range(g, g+grid.N_physical):
        rho = U[0, j]
        p = params.K_m * (rho / params.rho_jam) ** params.gamma_m
        U[1, j] = 10.0 + p  # v=10 m/s
    
    print(f"\nInitial: Smooth Gaussian")
    print(f"  Peak density: {rho_smooth.max():.6f} veh/m")
    print(f"  Total mass: {np.sum(rho_smooth) * grid.dx:.4f} vehicles")
    
    # Simulate
    dt = 0.1
    print(f"\nSimulating 50 steps with dt={dt}s...")
    masse_init = np.sum(rho_smooth) * grid.dx
    
    for i in range(50):
        U = strang_splitting_step(U, dt, grid, params)
        if (i+1) % 10 == 0:
            rho_now = U[0, g:g+grid.N_physical]
            print(f"  Step {i+1}: rho range=[{rho_now.min():.6f}, {rho_now.max():.6f}]")
    
    rho_final = U[0, g:g+grid.N_physical]
    
    # 1. Mass conservation
    print(f"\n✓ Checking mass conservation...")
    masse_final = np.sum(rho_final) * grid.dx
    conservation_error = abs(masse_final - masse_init) / masse_init
    print(f"  Initial mass: {masse_init:.4f} vehicles")
    print(f"  Final mass:   {masse_final:.4f} vehicles")
    print(f"  Relative error: {conservation_error*100:.2f}%")
    assert conservation_error < 0.05, f"Mass not conserved! Error={conservation_error*100:.2f}%"
    print(f"  ✅ Mass conserved")
    
    # 2. Positivity
    print(f"\n✓ Checking positivity...")
    assert np.all(rho_final >= 0), "Negative densities!"
    print(f"  ✅ All densities non-negative (min={rho_final.min():.6f})")
    
    # 3. Bounded (no blow-up)
    print(f"\n✓ Checking bounds...")
    assert np.max(rho_final) < params.rho_jam, f"Density > rho_jam!"
    print(f"  ✅ Densities bounded (max={rho_final.max():.6f} < {params.rho_jam})")
    
    print(f"\n{'='*60}")
    print(f"✅ Smooth case test PASSED for {scheme.upper()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Run tests individually for debugging
    test_godunov_sod_shock_tube()
    test_godunov_inflow_bc_steep_gradient()
    test_smooth_case_convergence("weno5")
    test_smooth_case_convergence("godunov")
