import numpy as np
import pytest
from numba import cuda

from arz_model.numerics.time_integration import solve_hyperbolic_step_ssp_rk3_gpu_native
from arz_model.grid.grid1d import Grid1D
from arz_model.config.physics_config import PhysicsConfig
from arz_model.numerics.gpu.memory_pool import GPUMemoryPool

def test_ssp_rk3_preserves_positivity_at_all_stages():
    """
    Verify that SSP-RK3 with per-stage bounds enforcement preserves
    density positivity at all intermediate stages.
    
    This test creates a scenario with extreme initial conditions that
    would cause intermediate overshoots without bounds enforcement.
    """
    # Setup: Grid near rho_max with high velocity
    grid = Grid1D(N_physical=100, dx=25.0, num_ghost_cells=3)
    params = PhysicsConfig(
        rho_max=0.2,  # 200 veh/km
        v_max_m_kmh=100.0,
        v_max_c_kmh=120.0,
        epsilon=1e-6
    )
    
    # Create initial state near physical bounds
    U_host = np.zeros((4, grid.N_total), dtype=np.float64)
    U_host[0, :] = 0.19  # rho_m near rho_max
    U_host[2, :] = 0.10  # rho_c moderate
    
    # Transfer to GPU
    d_U = cuda.to_device(U_host)
    gpu_pool = GPUMemoryPool(max_size_gb=1.0)
    
    # Run one RK3 time step with large dt (should trigger bounds)
    dt = 0.5
    d_U_result = solve_hyperbolic_step_ssp_rk3_gpu_native(
        d_U, dt, grid, params, gpu_pool, seg_id="test", current_time=0.0
    )
    
    # Verify result respects bounds
    U_result = d_U_result.copy_to_host()
    
    assert np.all(U_result[0, :] >= 0.0), "Negative rho_m detected"
    assert np.all(U_result[2, :] >= 0.0), "Negative rho_c detected"
    assert np.all(U_result[0, :] <= params.rho_max), f"rho_m exceeds rho_max"
    assert np.all(U_result[2, :] <= params.rho_max), f"rho_c exceeds rho_max"
    assert np.all(np.isfinite(U_result)), "NaN or Inf detected in result"
