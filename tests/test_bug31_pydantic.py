"""
Bug 31 Regression Test - Pydantic Config Version

Tests that IC and BC remain independent (Bug 31 architectural fix).
This test verifies the fix is preserved with new Pydantic config system.
"""

import pytest
import numpy as np
from arz_model.config import ConfigBuilder, SimulationConfig, GridConfig, PhysicsConfig
from arz_model.config.ic_config import UniformEquilibriumIC
from arz_model.config.bc_config import BoundaryConditionsConfig, InflowBC, OutflowBC, BCState
from arz_model.simulation.runner import SimulationRunner


def test_bug31_ic_bc_independence():
    """
    Bug 31: IC and BC must be COMPLETELY INDEPENDENT.
    
    - IC defines domain state at t=0 ONLY
    - BC defines flux at boundaries for all t≥0
    - IC should NOT affect BC configuration
    - BC should NOT derive state from IC
    """
    
    # Create configuration with explicit IC and BC
    config = SimulationConfig(
        grid=GridConfig(N=100, xmin=0.0, xmax=10.0),
        
        # Initial condition: Equilibrium at specific densities
        initial_conditions=UniformEquilibriumIC(
            type="uniform_equilibrium",
            rho_m=0.1,  # 100 veh/km
            rho_c=0.05,  # 50 veh/km
            R_val=10
        ),
        
        # Boundary conditions: DIFFERENT from IC (proves independence)
        boundary_conditions=BoundaryConditionsConfig(
            left=InflowBC(
                type="inflow",
                state=BCState(
                    rho_m=0.15,  # DIFFERENT from IC
                    w_m=10.0,
                    rho_c=0.08,  # DIFFERENT from IC
                    w_c=8.0
                )
            ),
            right=OutflowBC(
                type="outflow",
                state=BCState(rho_m=0.1, w_m=15.0, rho_c=0.05, w_c=12.0)
            )
        ),
        
        physics=PhysicsConfig(),
        t_final=10.0,
        output_dt=1.0,
        device='cpu',
        quiet=True
    )
    
    # Create runner
    runner = SimulationRunner(config=config, quiet=True)
    
    # Verify IC was created correctly (equilibrium)
    U_initial = runner.U
    rho_m_initial = U_initial[0, 2:-2].mean()  # Physical cells only
    rho_c_initial = U_initial[2, 2:-2].mean()
    
    # IC should be equilibrium at specified densities
    assert np.abs(rho_m_initial - 0.1e-3) < 1e-5, "IC motorcycles density incorrect"
    assert np.abs(rho_c_initial - 0.05e-3) < 1e-5, "IC cars density incorrect"
    
    # Verify BC is DIFFERENT from IC (proves independence)
    bc_params = runner.current_bc_params
    assert 'left' in bc_params, "Left BC not configured"
    assert bc_params['left']['type'] == 'inflow', "Left BC type incorrect"
    
    left_bc_state = bc_params['left']['state']
    assert left_bc_state[0] == 0.15, "Left BC rho_m should be 0.15 (DIFFERENT from IC 0.1)"
    assert left_bc_state[2] == 0.08, "Left BC rho_c should be 0.08 (DIFFERENT from IC 0.05)"
    
    print("✅ Bug 31 test PASSED: IC and BC are independent")
    print(f"   IC: rho_m={rho_m_initial*1000:.1f} veh/km, rho_c={rho_c_initial*1000:.1f} veh/km")
    print(f"   BC: rho_m={left_bc_state[0]*1000:.1f} veh/km, rho_c={left_bc_state[2]*1000:.1f} veh/km")


def test_bug31_no_ic_to_bc_coupling():
    """
    Verify that IC equilibrium state is NOT stored for BC use.
    
    This was the root cause of Bug 31: runner stored initial_equilibrium_state
    and reused it for BC configuration, creating IC→BC coupling.
    """
    
    config = ConfigBuilder.simple_test()
    runner = SimulationRunner(config=config, quiet=True)
    
    # Verify the problematic variable is NOT present
    assert not hasattr(runner, 'initial_equilibrium_state'), \
        "ARCHITECTURAL ERROR: initial_equilibrium_state should NOT exist (causes IC→BC coupling)"
    
    print("✅ Bug 31 architectural fix verified: No IC→BC coupling variable")


def test_inflow_bc_requires_explicit_state():
    """
    Verify that inflow BC requires explicit state configuration.
    
    BC cannot derive state from IC - must be explicitly configured.
    Pydantic validates this at config creation time.
    """
    
    from pydantic_core import ValidationError
    
    # This should FAIL at config creation (Pydantic validation)
    with pytest.raises(ValidationError, match="Field required"):
        # Create config with inflow BC but NO state
        bad_config = SimulationConfig(
            grid=GridConfig(N=50, xmin=0.0, xmax=5.0),
            initial_conditions=UniformEquilibriumIC(
                type="uniform_equilibrium",
                rho_m=0.1,
                rho_c=0.05,
                R_val=10
            ),
            boundary_conditions=BoundaryConditionsConfig(
                left=InflowBC(
                    type="inflow"
                    # NO state specified - Pydantic will catch this
                ),
                right=OutflowBC(
                    type="outflow",
                    state=BCState(rho_m=0.1, w_m=15.0, rho_c=0.05, w_c=12.0)
                )
            ),
            physics=PhysicsConfig(),
            t_final=1.0
        )
    
    print("✅ Inflow BC validation works: Pydantic requires explicit state")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Bug 31 Regression Tests - Pydantic Config System")
    print("="*60 + "\n")
    
    test_bug31_ic_bc_independence()
    print()
    test_bug31_no_ic_to_bc_coupling()
    print()
    test_inflow_bc_requires_explicit_state()
    
    print("\n" + "="*60)
    print("✅ ALL BUG 31 TESTS PASSED")
    print("="*60)
