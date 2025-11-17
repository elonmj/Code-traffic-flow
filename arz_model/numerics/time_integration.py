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

# Import optimized SSP-RK3 fused kernel (Phase 2.3+2.4)
from .gpu.ssp_rk3_cuda import ssp_rk3_fused_kernel

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
    d_R = gpu_pool.get_segment_road_quality(seg_id)

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

    **Phase 2.3+2.4 OPTIMIZED VERSION**: Uses fused SSP-RK3 kernel with integrated
    WENO5 reconstruction and Central-Upwind Riemann solver for maximum performance.

    Args:
        d_U_in: Input state device array.
        dt: Time step.
        grid: Grid object.
        params: PhysicsConfig object.
        gpu_pool: The memory pool for managing GPU arrays.
        seg_id: The segment ID.
        current_time: The current simulation time.

    Returns:
        The state array after the hyperbolic step.
    """
    # Allocate output array from pool
    d_U_out = gpu_pool.get_temp_array(d_U_in.shape, d_U_in.dtype)
    
    # CRITICAL: Fused kernel expects shape (N, num_vars), but time_integration uses (num_vars, N)
    # We need to transpose for compatibility
    N = d_U_in.shape[1]  # Number of spatial cells
    num_vars = d_U_in.shape[0]  # Number of conserved variables (4 for ARZ)
    
    # Transpose input: (num_vars, N) -> (N, num_vars)
    d_U_in_T = cuda.device_array((N, num_vars), dtype=d_U_in.dtype)
    d_U_out_T = cuda.device_array((N, num_vars), dtype=d_U_in.dtype)
    
    # Simple transpose kernel
    @cuda.jit
    def transpose_kernel(src, dst):
        i, j = cuda.grid(2)
        if i < src.shape[0] and j < src.shape[1]:
            dst[j, i] = src[i, j]
    
    threadsperblock_2d = (16, 16)
    blockspergrid_x = (num_vars + threadsperblock_2d[0] - 1) // threadsperblock_2d[0]
    blockspergrid_y = (N + threadsperblock_2d[1] - 1) // threadsperblock_2d[1]
    blockspergrid_2d = (blockspergrid_x, blockspergrid_y)
    
    transpose_kernel[blockspergrid_2d, threadsperblock_2d](d_U_in, d_U_in_T)
    
    # Extract spatial resolution from grid
    dx = grid.dx
    
    # Extract physics parameters for WENO+Riemann
    rho_max = params.rho_max
    alpha = params.alpha
    epsilon = params.epsilon
    k_m = params.k_m
    gamma_m = params.gamma_m
    k_c = params.k_c
    gamma_c = params.gamma_c
    weno_eps = 1e-6  # WENO smoothness indicator epsilon
    
    # Configure kernel launch: 1D grid over spatial cells
    # Each thread handles one spatial cell through all 3 RK stages
    threadsperblock = 256
    blockspergrid = (N + threadsperblock - 1) // threadsperblock
    
    # Launch FUSED kernel: Does WENO5 + Riemann + 3 RK stages in one go!
    # This eliminates intermediate global memory traffic (Phase 2.3)
    # and integrates high-order physics (Phase 2.4)
    ssp_rk3_fused_kernel[blockspergrid, threadsperblock](
        d_U_in_T,    # u_n: Input state (transposed to N, num_vars)
        d_U_out_T,   # u_np1: Output state (transposed to N, num_vars)
        dt,          # Time step
        dx,          # Spatial resolution
        N,           # Number of cells
        num_vars,    # Number of variables (4)
        alpha,       # Physics: anticipation parameter
        rho_max,     # Physics: jam density (rho_jam)
        epsilon,     # Numerical stability epsilon
        k_m,         # Physics: motorway capacity
        gamma_m,     # Physics: motorway exponent
        k_c,         # Physics: city capacity
        gamma_c,     # Physics: city exponent
        weno_eps     # WENO: smoothness epsilon
    )
    
    # Transpose output back: (N, num_vars) -> (num_vars, N)
    blockspergrid_x_out = (N + threadsperblock_2d[0] - 1) // threadsperblock_2d[0]
    blockspergrid_y_out = (num_vars + threadsperblock_2d[1] - 1) // threadsperblock_2d[1]
    blockspergrid_2d_out = (blockspergrid_x_out, blockspergrid_y_out)
    transpose_kernel[blockspergrid_2d_out, threadsperblock_2d](d_U_out_T, d_U_out)
    
    # Apply physical bounds to ensure positivity and max constraints
    _apply_physical_bounds_gpu_in_place(d_U_out, grid, params)
    
    # Return the output (caller will continue Strang splitting process)
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
            N_total  # epsilon is now module-level constant in WENO kernel
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


# --- New CUDA Kernel for ODE Step ---
@cuda.jit
def _ode_step_kernel(U_in, U_out, dt_ode, R_local_arr, N_physical, num_ghost_cells,
                     # Pass necessary parameters explicitly
                     alpha, rho_jam, K_m, gamma_m, K_c, gamma_c, # Pressure
                     rho_jam_eq, V_creeping, # Equilibrium Speed base params
                     v_max_m_cat1, v_max_m_cat2, v_max_m_cat3, # Motorcycle Vmax per category
                     v_max_c_cat1, v_max_c_cat2, v_max_c_cat3, # Car Vmax per category
                     tau_relax_m, tau_relax_c, # Relaxation times
                     epsilon):
    """
    CUDA kernel for explicit Euler step for the ODE source term.
    Updates U_out based on U_in and the source term S(U_in).
    Operates only on physical cells.
    """
    idx = cuda.grid(1) # Global thread index

    # Check if index is within the range of physical cells
    if idx < N_physical:
        j_phys = idx
        j_total = j_phys + num_ghost_cells # Index in the full U array (including ghosts)

        # --- 1. Get local state and road quality ---
        # Read state variables directly into scalars (potential register allocation)
        y0 = U_in[0, j_total]
        y1 = U_in[1, j_total]
        y2 = U_in[2, j_total]
        y3 = U_in[3, j_total]
        # Note: Access U_in[i, j_total] is likely non-coalesced. Consider transposing U_in/U_out later.

        # Road quality for this physical cell
        # Assumes R_local_arr is the array of road qualities for physical cells
        R_local = R_local_arr[j_phys]

        # --- 2. Calculate intermediate values (Equilibrium speeds, Relaxation times) ---
        # These calculations need to be done per-cell within the kernel
        rho_m_calc = max(y0, 0.0)
        rho_c_calc = max(y2, 0.0)

        # Assume physics functions have @cuda.jit(device=True) versions
        Ve_m, Ve_c = physics.calculate_equilibrium_speed_gpu(
            rho_m_calc, rho_c_calc, R_local,
            rho_jam_eq, V_creeping, # Pass base params for eq speed
            v_max_m_cat1, v_max_m_cat2, v_max_m_cat3, # Pass category-specific Vmax
            v_max_c_cat1, v_max_c_cat2, v_max_c_cat3
        )
        tau_m, tau_c = physics.calculate_relaxation_time_gpu(
            rho_m_calc, rho_c_calc, # Pass densities (might be used in future)
            tau_relax_m, tau_relax_c # Pass base relaxation times
        )

        # --- 3. Calculate source term S(U) ---
        # Assume physics.calculate_source_term_gpu has a @cuda.jit(device=True) version
        # Create a tuple or temporary array if the device function expects an array-like input
        # If it accepts scalars, pass them directly. Assuming it needs array-like:
        y_temp = (y0, y1, y2, y3) # Pass as a tuple
        source = physics.calculate_source_term_gpu(
            y_temp, alpha, rho_jam, K_m, gamma_m, K_c, gamma_c,
            Ve_m, Ve_c, tau_m, tau_c, epsilon
        )

        # --- 4. Apply Explicit Euler step ---
        # Update the output array directly at the correct total index
        # Note: Access U_out[i, j_total] is likely non-coalesced.
        U_out[0, j_total] = y0 + dt_ode * source[0]
        U_out[1, j_total] = y1 + dt_ode * source[1]
        U_out[2, j_total] = y2 + dt_ode * source[2]
        U_out[3, j_total] = y3 + dt_ode * source[3]

# --- New GPU Wrapper Function for ODE Step ---
def solve_ode_step_gpu(d_U_in: cuda.devicearray.DeviceNDArray, dt_ode: float, grid: Grid1D, params: 'PhysicsConfig', d_R: cuda.devicearray.DeviceNDArray) -> cuda.devicearray.DeviceNDArray:
    """
    Solves the ODE system dU/dt = S(U) using an explicit Euler step on the GPU.
    Operates entirely on GPU arrays.

    Args:
        d_U_in (cuda.devicearray.DeviceNDArray): Input state device array (including ghost cells). Shape (4, N_total).
        dt_ode (float): Time step for the ODE integration.
        grid (Grid1D): Grid object (used for N_physical, num_ghost_cells).
        params (ModelParameters): Model parameters.
        d_R (cuda.devicearray.DeviceNDArray): Road quality device array (physical cells only). Shape (N_physical,).

    Returns:
        cuda.devicearray.DeviceNDArray: Output state device array after the ODE step. Shape (4, N_total).
    """
    # Road quality check is implicitly handled by requiring d_R
    if d_R is None or not cuda.is_cuda_array(d_R):
         raise ValueError("Valid GPU road quality array d_R must be provided for GPU ODE step.")
    if not hasattr(physics, 'calculate_source_term_gpu') or \
       not hasattr(physics, 'calculate_equilibrium_speed_gpu') or \
       not hasattr(physics, 'calculate_relaxation_time_gpu'):
        raise NotImplementedError("GPU versions (_gpu suffix) of required physics functions are not available in the physics module.")

    # --- Extract max speed values ---
    # PhysicsConfig has single values for each vehicle type (no categories)
    # Convert from km/h to m/s (divide by 3.6)
    v_max_m_cat1 = params.v_max_m_kmh / 3.6
    v_max_m_cat2 = params.v_max_m_kmh / 3.6
    v_max_m_cat3 = params.v_max_m_kmh / 3.6
    
    v_max_c_cat1 = params.v_max_c_kmh / 3.6
    v_max_c_cat2 = params.v_max_c_kmh / 3.6
    v_max_c_cat3 = params.v_max_c_kmh / 3.6


    # --- 1. Allocate output array on GPU ---
    # Note: We don't need to initialize with d_U_in because the kernel only updates
    # physical cells. Ghost cells will be updated by the boundary condition kernel later.
    # However, allocating like d_U_in ensures the same shape and dtype.
    d_U_out = cuda.device_array_like(d_U_in)
    # Explicitly copy ghost cells from input to output *before* kernel launch
    # This ensures they are preserved if the kernel doesn't touch them (which it shouldn't)
    # and are correct if the subsequent hyperbolic step needs them.
    n_ghost = grid.num_ghost_cells
    n_phys = grid.N_physical
    d_U_out[:, :n_ghost] = d_U_in[:, :n_ghost]
    d_U_out[:, n_ghost+n_phys:] = d_U_in[:, n_ghost+n_phys:]


    # --- 2. Configure and launch kernel ---
    threadsperblock = 256 # Typical value, can be tuned
    blockspergrid = math.ceil(grid.N_physical / threadsperblock)

    _ode_step_kernel[blockspergrid, threadsperblock](
        d_U_in, d_U_out, dt_ode, d_R, grid.N_physical, grid.num_ghost_cells,
        # Pass all necessary parameters explicitly from the params object
        # Pressure params (params is already PhysicsConfig, no .physics accessor needed)
        params.alpha, params.rho_max, params.k_m, params.gamma_m, params.k_c, params.gamma_c,
        # Equilibrium speed params (base + extracted category Vmax)
        params.rho_max, params.v_creeping_kmh / 3.6, # Convert creeping speed from km/h to m/s
        v_max_m_cat1, v_max_m_cat2, v_max_m_cat3,
        v_max_c_cat1, v_max_c_cat2, v_max_c_cat3,
        # Relaxation times
        params.tau_m, params.tau_c,
        # Epsilon
        params.epsilon
    )
    # cuda.synchronize() # No sync needed here, let subsequent steps handle it

    # --- 3. Return GPU array ---
    # No copy back to host
    return d_U_out


# --- End of Physical State Bounds Enforcement ---



