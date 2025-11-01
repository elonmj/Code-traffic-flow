"""
Test Phase 1: Junction Flux Blocking CPU Implementation

Simple test to verify junction-aware Riemann solver blocks traffic during RED signal.
"""
import pytest
import numpy as np
from arz_model.core.parameters import ModelParameters
from arz_model.grid.grid1d import Grid1D
from arz_model.network.junction_info import JunctionInfo
from arz_model.numerics import riemann_solvers


def test_junction_info_creation():
    """Test that JunctionInfo dataclass can be created correctly."""
    junction = JunctionInfo(
        is_junction=True,
        light_factor=0.01,
        node_id=1
    )
    
    assert junction.is_junction is True
    assert junction.light_factor == 0.01
    assert junction.node_id == 1
    assert "node=1" in str(junction) or "blocking=99%" in str(junction)
    print(f"✅ JunctionInfo creation OK: {junction}")


def test_grid1d_junction_attribute():
    """Test that Grid1D has junction_at_right attribute."""
    grid = Grid1D(N=10, xmin=0.0, xmax=100.0, num_ghost_cells=3)
    
    # Should start as None
    assert hasattr(grid, 'junction_at_right')
    assert grid.junction_at_right is None
    
    # Can be set
    junction = JunctionInfo(is_junction=True, light_factor=0.5, node_id=2)
    grid.junction_at_right = junction
    
    assert grid.junction_at_right is not None
    assert grid.junction_at_right.light_factor == 0.5
    print(f"✅ Grid1D junction attribute OK")


def test_central_upwind_flux_no_junction():
    """Test that flux works normally without junction_info (backward compatibility)."""
    # Create params with required attributes
    params = ModelParameters()
    params.alpha = 0.5
    params.rho_jam = 0.2
    params.epsilon = 1e-10
    params.K_m = 20.0
    params.gamma_m = 2.0
    params.K_c = 30.0
    params.gamma_c = 2.0
    params.device = 'cpu'
    
    # Simple test states
    U_L = np.array([0.08, 0.56, 0.0, 0.0])  # rho_m=0.08, w_m=0.56
    U_R = np.array([0.08, 0.56, 0.0, 0.0])
    
    # Compute flux without junction
    F = riemann_solvers.central_upwind_flux(U_L, U_R, params, junction_info=None)
    
    assert F.shape == (4,)
    # Flux might be zero if states are identical (no gradient)
    print(f"✅ Normal flux (no junction): F={F}")
    print(f"   (Flux might be ~0 for identical states)")
    return F


def test_central_upwind_flux_with_red_junction():
    """Test that flux is reduced when junction_info has RED signal (light_factor=0.01)."""
    params = ModelParameters()
    params.alpha = 0.5
    params.rho_jam = 0.2
    params.epsilon = 1e-10
    params.K_m = 20.0
    params.gamma_m = 2.0
    params.K_c = 30.0
    params.gamma_c = 2.0
    params.device = 'cpu'
    
    # Same test states
    U_L = np.array([0.08, 0.56, 0.0, 0.0])
    U_R = np.array([0.08, 0.56, 0.0, 0.0])
    
    # Compute normal flux
    F_normal = riemann_solvers.central_upwind_flux(U_L, U_R, params, junction_info=None)
    
    # Compute flux with RED signal
    junction_red = JunctionInfo(is_junction=True, light_factor=0.01, node_id=1)
    F_red = riemann_solvers.central_upwind_flux(U_L, U_R, params, junction_info=junction_red)
    
    # RED flux should be ~99% reduced in absolute value
    assert np.abs(F_red[0]) < np.abs(F_normal[0])  # Reduced magnitude
    assert np.abs(F_red[0] - 0.01 * F_normal[0]) < 1e-6  # Reduced by exactly 99%
    
    print(f"✅ RED signal blocks flux: F_normal[0]={F_normal[0]:.6f}, F_red[0]={F_red[0]:.6f}")
    reduction_pct = (1 - np.abs(F_red[0])/np.abs(F_normal[0]))*100 if F_normal[0] != 0 else 0
    print(f"   Reduction: {reduction_pct:.1f}%")


def test_central_upwind_flux_with_green_junction():
    """Test that flux is NOT reduced when junction_info has GREEN signal (light_factor=1.0)."""
    params = ModelParameters()
    params.alpha = 0.5
    params.rho_jam = 0.2
    params.epsilon = 1e-10
    params.K_m = 20.0
    params.gamma_m = 2.0
    params.K_c = 30.0
    params.gamma_c = 2.0
    params.device = 'cpu'
    
    U_L = np.array([0.08, 0.56, 0.0, 0.0])
    U_R = np.array([0.08, 0.56, 0.0, 0.0])
    
    # Normal flux
    F_normal = riemann_solvers.central_upwind_flux(U_L, U_R, params, junction_info=None)
    
    # GREEN signal (light_factor=1.0)
    junction_green = JunctionInfo(is_junction=True, light_factor=1.0, node_id=1)
    F_green = riemann_solvers.central_upwind_flux(U_L, U_R, params, junction_info=junction_green)
    
    # GREEN flux should equal normal flux
    assert np.allclose(F_green, F_normal, rtol=1e-10)
    
    print(f"✅ GREEN signal allows full flow: F_green[0]={F_green[0]:.6f} == F_normal[0]={F_normal[0]:.6f}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 1 TEST: Junction Flux Blocking CPU Implementation")
    print("="*70 + "\n")
    
    test_junction_info_creation()
    test_grid1d_junction_attribute()
    test_central_upwind_flux_no_junction()
    test_central_upwind_flux_with_red_junction()
    test_central_upwind_flux_with_green_junction()
    
    print("\n" + "="*70)
    print("✅ ALL PHASE 1 TESTS PASSED")
    print("="*70 + "\n")
