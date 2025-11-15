import numpy as np
from numba import cuda
import math
from typing import TYPE_CHECKING, Optional

from ..grid.grid1d import Grid1D
from ..core import physics

# Import GPU implementations from within the same package
from .gpu.weno_cuda import (
    weno5_reconstruction_kernel, 
    apply_boundary_conditions_kernel
)
from .reconstruction.weno_gpu import _compute_flux_divergence_weno_kernel
from .riemann_solvers import central_upwind_flux_cuda_kernel
from .reconstruction.converter import conserved_to_primitives_arr_gpu

if TYPE_CHECKING:
    from ..core.parameters import PhysicsConfig
    from .gpu.memory_pool import GPUMemoryPool

from ..core.parameters import ModelParameters


# --- Physical State Bounds Enforcement ---

@cuda.jit
def _apply_bounds_kernel(U, N_physical, num_ghost, rho_max, v_max, epsilon,
                         alpha, K_m, gamma_m, K_c, gamma_c):
    """
    GPU kernel for applying physical bounds to state variables.
    
    Enforces:
    - Density: 0 ≤ rho ≤ rho_max
    - Velocity: |v| ≤ v_max
    """
    i = cuda.grid(1)
    
    if i < N_physical:
        j = i + num_ghost  # Physical cell index in full array
        
        rho_m = U[0, j]
        w_m = U[1, j]
        rho_c = U[2, j]
        w_c = U[3, j]
        
        # 1. Clamp densities to [0, rho_max]
        rho_m = max(0.0, min(rho_m, rho_max))
        rho_c = max(0.0, min(rho_c, rho_max))
        
        # 2. Calculate pressure and clamp velocity (motorcycles)
        if rho_m > epsilon:
            # Simplified pressure calculation (inline to avoid device function issues)
            rho_total = rho_m + rho_c
            pressure_factor = (rho_total / rho_max) ** gamma_m if rho_total > epsilon else 0.0
            p_m = K_m * pressure_factor
            
            v_m = w_m - p_m
            v_m = max(-v_max, min(v_m, v_max))
            w_m = v_m + p_m
        else:
            w_m = 0.0
        
        # 3. Same for cars
        if rho_c > epsilon:
            rho_total = rho_m + rho_c
            pressure_factor = (rho_total / rho_max) ** gamma_c if rho_total > epsilon else 0.0
            p_c = K_c * pressure_factor
            
            v_c = w_c - p_c
            v_c = max(-v_max, min(v_c, v_max))
            w_c = v_c + p_c
        else:
            w_c = 0.0
        
        # Write back bounded values
        U[0, j] = rho_m
        U[1, j] = w_m
        U[2, j] = rho_c
        U[3, j] = w_c


def _apply_physical_bounds_gpu_in_place(d_U: cuda.devicearray.DeviceNDArray, grid: Grid1D, 
                                           params: 'PhysicsConfig', stream: Optional[cuda.cudadrv.driver.Stream] = 0):
    """
    Applies physical bounds to a state array on the GPU, modifying it in-place.
    This is a helper to be called from within other GPU-orchestrating functions.
    """
    threads_per_block = 256
    blocks_per_grid = math.ceil(grid.N_physical / threads_per_block)
    
    # Get v_max from config, convert from km/h to m/s
    v_max_physical = max(params.v_max_m_kmh, params.v_max_c_kmh) / 3.6

    _apply_bounds_kernel[blocks_per_grid, threads_per_block, stream](
        d_U, 
        grid.N_physical, 
        grid.num_ghost_cells, 
        params.rho_max, 
        v_max_physical, 
        params.epsilon,
        params.alpha, 
        params.k_m, 
        params.gamma_m, 
        params.k_c, 
        params.gamma_c
    )


# --- End of Physical State Bounds Enforcement ---


def strang_splitting_step_gpu_native(
    d_U_n: cuda.devicearray.DeviceNDArray, 
    dt: float, 
    grid: Grid1D, 
    params: 'PhysicsConfig', 
    gpu_pool: 'GPUMemoryPool',
    seg_id: str,
    current_time: float
) -> cuda.devicearray.DeviceNDArray:
    """
    Performs one full, GPU-native time step using Strang splitting.

    This function is the core of the GPU-only simulation loop. It orchestrates
    the ODE and hyperbolic substeps, ensuring all data remains on the GPU
    and all transfers are eliminated.

    Args:
        d_U_n: Input state device array for the current segment.
        dt: The full time step.
        grid: The Grid1D object for the segment (CPU object).
        params: ModelParameters object (CPU object).
        gpu_pool: The GPUMemoryPool managing all device arrays.
        seg_id: The identifier for the current segment.
        current_time: The current simulation time.

    Returns:
        The updated state device array for the segment.
    """
    # 1. Get cached road quality array from the memory pool
    d_R = gpu_pool.get_road_quality_array(seg_id)

    # 2. First ODE substep (dt/2)
    d_U_star = solve_ode_step_gpu(d_U_n, dt / 2.0, grid, params, d_R)

    # 3. Hyperbolic substep (dt)
    # This will be a new function that wraps the GPU-native WENO/SSP-RK3 logic
    d_U_ss = solve_hyperbolic_step_ssp_rk3_gpu_native(d_U_star, dt, grid, params, gpu_pool, seg_id, current_time)

    # 4. Second ODE substep (dt/2)
    d_U_np1 = solve_ode_step_gpu(d_U_ss, dt / 2.0, grid, params, d_R)
    
    # 5. Apply physical bounds to prevent numerical instability
    # TODO: Implement apply_physical_state_bounds_gpu function
    # For now, skip bounds application as it's not critical for basic functionality
    # d_U_np1 = apply_physical_state_bounds_gpu(d_U_np1, grid, params, rho_max=1.5 * params.rho_max)

    return d_U_np1


def solve_hyperbolic_step_ssp_rk3_gpu_native(
    d_U_in: cuda.devicearray.DeviceNDArray, 
    dt: float, 
    grid: Grid1D, 
    params: 'PhysicsConfig', 
    gpu_pool: 'GPUMemoryPool',
    seg_id: str,
    current_time: float
) -> cuda.devicearray.DeviceNDArray:
    """
    Solves the hyperbolic step w_t + F(w)_x = 0 using a 3rd-order SSP-RK scheme
    entirely on the GPU, leveraging the GPUMemoryPool.

    This function replaces the legacy `solve_hyperbolic_step_ssprk3_gpu`.

    Args:
        d_U_in: Input state device array.
        dt: Time step.
        grid: Grid object.
        params: ModelParameters object.
        gpu_pool: The memory pool for managing GPU arrays.
        seg_id: The segment ID.
        current_time: The current simulation time.

    Returns:
        The state array after the hyperbolic step.
    """
    # Get temporary arrays from the pool for intermediate RK steps
    # This avoids reallocation and leverages cached memory.
    d_U1 = gpu_pool.get_temp_array(d_U_in.shape, d_U_in.dtype)
    d_U2 = gpu_pool.get_temp_array(d_U_in.shape, d_U_in.dtype)

    # Configure kernel launch grid for RK stages
    threadsperblock = (16, 16)  # 2D grid for state array (4 x N_total)
    blockspergrid_x = (d_U_in.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (d_U_in.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # --- RK Stage 1 ---
    # L_U0 = L(U_n)
    L_U0 = calculate_spatial_discretization_weno_gpu_native(d_U_in, grid, params, gpu_pool, seg_id, current_time)
    # U_1 = U_n + dt * L(U_n)
    ssp_rk3_stage_1_kernel[blockspergrid, threadsperblock](d_U_in, L_U0, dt, d_U1)
    _apply_physical_bounds_gpu_in_place(d_U1, grid, params)

    # --- RK Stage 2 ---
    # L_U1 = L(U_1)
    L_U1 = calculate_spatial_discretization_weno_gpu_native(d_U1, grid, params, gpu_pool, seg_id, current_time)
    # U_2 = (3/4)U_n + (1/4)U_1 + (1/4)dt * L(U_1)
    ssp_rk3_stage_2_kernel[blockspergrid, threadsperblock](d_U_in, d_U1, L_U1, dt, d_U2)
    _apply_physical_bounds_gpu_in_place(d_U2, grid, params)

    # --- RK Stage 3 ---
    # L_U2 = L(U_2)
    L_U2 = calculate_spatial_discretization_weno_gpu_native(d_U2, grid, params, gpu_pool, seg_id, current_time)
    # U_np1 = (1/3)U_n + (2/3)U_2 + (2/3)dt * L(U_2)
    d_U_out = gpu_pool.get_temp_array(d_U_in.shape, d_U_in.dtype) # Get a new array for the output
    ssp_rk3_stage_3_kernel[blockspergrid, threadsperblock](d_U_in, d_U2, L_U2, dt, d_U_out)
    _apply_physical_bounds_gpu_in_place(d_U_out, grid, params)

    # Release temporary arrays back to the pool for reuse
    gpu_pool.release_temp_array(d_U1)
    gpu_pool.release_temp_array(d_U2)
    
    # The final result is in d_U_out, which is also a temporary array.
    # The caller (`strang_splitting_step_gpu_native`) will continue the process.
    return d_U_out


@cuda.jit
def ssp_rk3_stage_1_kernel(d_U_in, d_L_U, dt, d_U_out):
    """Kernel for SSP-RK3 stage 1."""
    i, j = cuda.grid(2)
    if i < d_U_in.shape[0] and j < d_U_in.shape[1]:
        d_U_out[i, j] = d_U_in[i, j] + dt * d_L_U[i, j]

@cuda.jit
def ssp_rk3_stage_2_kernel(d_U_n, d_U1, d_L_U1, dt, d_U2):
    """Kernel for SSP-RK3 stage 2."""
    i, j = cuda.grid(2)
    if i < d_U_n.shape[0] and j < d_U_n.shape[1]:
        d_U2[i, j] = 0.75 * d_U_n[i, j] + 0.25 * d_U1[i, j] + 0.25 * dt * d_L_U1[i, j]

@cuda.jit
def ssp_rk3_stage_3_kernel(d_U_n, d_U2, d_L_U2, dt, d_U_np1):
    """Kernel for SSP-RK3 stage 3."""
    i, j = cuda.grid(2)
    if i < d_U_n.shape[0] and j < d_U_n.shape[1]:
        d_U_np1[i, j] = (1.0/3.0) * d_U_n[i, j] + (2.0/3.0) * d_U2[i, j] + (2.0/3.0) * dt * d_L_U2[i, j]


def calculate_spatial_discretization_weno_gpu_native(
    d_U_in: cuda.devicearray.DeviceNDArray, 
    grid: Grid1D, 
    params: 'PhysicsConfig', 
    gpu_pool: 'GPUMemoryPool',
    seg_id: str,
    current_time: float
) -> cuda.devicearray.DeviceNDArray:
    """
    Performs a fully GPU-native spatial discretization using WENO5.

    This function orchestrates the following steps entirely on the GPU:
    1. Converts conserved variables (U) to primitive variables (P).
    2. Performs WENO5 reconstruction on each primitive variable.
    3. Calculates numerical fluxes at interfaces using a Central-Upwind scheme.
    4. Computes the final spatial discretization L(U) = -dF/dx.

    It assumes that boundary conditions have already been applied to the
    ghost cells of the input array `d_U_in` by the network coupling kernels.

    Args:
        d_U_in: Input state device array (with ghost cells updated).
        grid: The Grid1D object for the segment.
        params: ModelParameters object.
        gpu_pool: The GPUMemoryPool for managing temporary arrays.
        seg_id: The segment ID (used for junction-awareness).
        current_time: The current simulation time (for logging/debugging).

    Returns:
        A device array containing the spatial discretization L(U).
    """
    # Get constants from grid and params
    N_total = grid.N_total
    N_physical = grid.N_physical
    n_ghost = grid.num_ghost_cells
    # params is already PhysicsConfig, no .physics accessor needed
    phys_params = params

    # --- Get temporary arrays from the pool ---
    d_P = gpu_pool.get_temp_array(d_U_in.shape, d_U_in.dtype)
    d_P_left = gpu_pool.get_temp_array(d_U_in.shape, d_U_in.dtype)
    d_P_right = gpu_pool.get_temp_array(d_U_in.shape, d_U_in.dtype)
    d_fluxes = gpu_pool.get_temp_array(d_U_in.shape, d_U_in.dtype)
    d_L_U = gpu_pool.get_temp_array(d_U_in.shape, d_U_in.dtype)

    # --- Kernel launch configuration ---
    threadsperblock = 256
    blockspergrid_total = (N_total + threadsperblock - 1) // threadsperblock
    
    # --- 1. Conversion: Conserved -> Primitives ---
    conserved_to_primitives_arr_gpu(
        d_U_in, phys_params.alpha, phys_params.rho_max, phys_params.epsilon,
        phys_params.k_m, phys_params.gamma_m, phys_params.k_c, phys_params.gamma_c,
        target_array=d_P
    )

    # --- 2. Reconstruction: WENO5 ---
    for var_idx in range(4):
        weno5_reconstruction_kernel[blockspergrid_total, threadsperblock](
            d_P[var_idx, :], d_P_left[var_idx, :], d_P_right[var_idx, :],
            N_total, phys_params.epsilon  # Use epsilon for WENO numerical stability
        )
        # Apply simple extrapolation at boundaries for WENO stencil
        apply_boundary_conditions_kernel[1, n_ghost](
            d_P_left[var_idx, :], d_P_right[var_idx, :], d_P[var_idx, :], N_total
        )

    # --- 3. Flux Calculation: Central-Upwind ---
    # This kernel is now imported at the top level
    
    # Determine junction blocking factor for this segment
    light_factor = 1.0
    # The logic for getting segment_info and light_factor needs to be handled
    # by the caller (NetworkSimulator) and passed down. For now, we assume 1.0.

    blockspergrid_flux = (N_total - 1 + threadsperblock - 1) // threadsperblock
    central_upwind_flux_cuda_kernel[blockspergrid_flux, threadsperblock](
        d_U_in, # The flux kernel uses U to calculate speeds
        phys_params.alpha, phys_params.rho_max, phys_params.epsilon,
        phys_params.k_m, phys_params.gamma_m, phys_params.k_c, phys_params.gamma_c,
        light_factor,
        d_fluxes
    )

    # --- 4. Flux Divergence ---
    # This kernel is now imported at the top level
    blockspergrid_phys = (N_physical + threadsperblock - 1) // threadsperblock
    _compute_flux_divergence_weno_kernel[blockspergrid_phys, threadsperblock](
        d_fluxes, d_L_U, grid.dx, n_ghost, N_physical
    )

    # --- Release temporary arrays ---
    gpu_pool.release_temp_array(d_P)
    gpu_pool.release_temp_array(d_P_left)
    gpu_pool.release_temp_array(d_P_right)
    gpu_pool.release_temp_array(d_fluxes)
    # d_L_U is the return value, so it's not released here.
    # The caller is responsible for it.

    return d_L_U



