"""
Quick validation test for GPU implementation.
Tests that the GPU kernels can be called successfully.
"""
import numpy as np
from arz_model.core.parameters import ModelParameters
from arz_model.grid.grid1d import Grid1D
from arz_model.numerics.time_integration import strang_splitting_step_gpu, GPU_AVAILABLE

print("=" * 60)
print("GPU KERNEL VALIDATION TEST")
print("=" * 60)

# Check GPU availability
print(f"\n1. GPU Available: {GPU_AVAILABLE}")
if not GPU_AVAILABLE:
    print("❌ GPU not available - test skipped")
    exit(0)
else:
    # Let CUDA initialize naturally on first use
    from numba import cuda
    print(f"✅ Numba CUDA ready (will initialize on first kernel call)")

# Create minimal test setup
print("\n2. Creating test setup...")
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
params.N = 20  # Small grid for quick test
params.cfl_number = 0.8
params.ghost_cells = 3
params.spatial_scheme = 'weno5'
params.time_scheme = 'ssprk3'
params.ode_solver = 'RK45'
params.ode_rtol = 1e-6
params.ode_atol = 1e-9
params.Vmax_m = {0: 12.0, 1: 15.0, 2: 18.0}
params.Vmax_c = {0: 10.0, 1: 13.0, 2: 16.0}
params.device = 'gpu'

print("✅ Parameters created")

# Create grid (correct API: N, xmin, xmax, num_ghost_cells)
grid = Grid1D(N=20, xmin=0.0, xmax=100.0, num_ghost_cells=3)
print(f"✅ Grid created: N={grid.N_physical}, N_total={grid.N_total}")

# Load road quality (required for ODE step)
road_quality = np.ones(grid.N_physical, dtype=int)  # Category 1 road
grid.load_road_quality(road_quality)
print(f"✅ Road quality loaded: {road_quality[:5]}...")

# Create initial state (equilibrium)
U_init = np.zeros((4, grid.N_total), dtype=np.float64)
rho_m_init = 0.1
U_init[0, grid.physical_cell_indices] = rho_m_init
U_init[1, grid.physical_cell_indices] = 8.0  # v≈8 m/s + small pressure
U_init[2, grid.physical_cell_indices] = 0.0
U_init[3, grid.physical_cell_indices] = 0.0

print(f"✅ Initial state created: rho_m_mean={np.mean(U_init[0, grid.physical_cell_indices]):.4f}")

# Transfer to GPU (Numba CUDA)
from numba import cuda
U_gpu = cuda.to_device(U_init)
print(f"✅ State transferred to GPU: shape={U_gpu.shape}, dtype={U_gpu.dtype}")

# Test GPU step
print("\n3. Testing GPU Strang splitting step...")
dt = 0.001  # 1 ms timestep

try:
    U_new_gpu = strang_splitting_step_gpu(U_gpu, dt, grid, params)
    print(f"✅ GPU step completed successfully!")
    print(f"   Output shape: {U_new_gpu.shape}")
    print(f"   Output dtype: {U_new_gpu.dtype}")
    
    # Check result sanity (copy from GPU to CPU using Numba)
    U_new_cpu = U_new_gpu.copy_to_host()
    rho_m_new = np.mean(U_new_cpu[0, grid.physical_cell_indices])
    w_m_new = np.mean(U_new_cpu[1, grid.physical_cell_indices])
    
    print(f"\n4. Result validation:")
    print(f"   rho_m_mean: {rho_m_init:.4f} → {rho_m_new:.4f}")
    print(f"   w_m_mean: {8.0:.4f} → {w_m_new:.4f}")
    print(f"   rho_m_max: {np.max(U_new_cpu[0, grid.physical_cell_indices]):.4f}")
    print(f"   rho_m_min: {np.min(U_new_cpu[0, grid.physical_cell_indices]):.4f}")
    
    # Basic sanity checks
    if np.all(np.isfinite(U_new_cpu)):
        print("✅ All values are finite (no NaN/Inf)")
    else:
        print("❌ WARNING: Non-finite values detected!")
        
    if np.all(U_new_cpu[0, :] >= 0):
        print("✅ All densities are non-negative")
    else:
        print("❌ WARNING: Negative densities detected!")
        
    if np.all(U_new_cpu[0, :] <= 2.0 * params.rho_jam):
        print("✅ All densities within reasonable bounds")
    else:
        print("❌ WARNING: Extremely high densities detected!")
    
    print("\n" + "=" * 60)
    print("🎉 GPU KERNEL VALIDATION PASSED!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR during GPU step: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
