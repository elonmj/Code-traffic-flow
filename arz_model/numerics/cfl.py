import numpy as np
from numba import cuda, float64, int32 # Import cuda and types
import math # For ceil
from ..grid.grid1d import Grid1D
from arz_model.core.physics import _calculate_pressure_cuda, _calculate_physical_velocity_cuda, _calculate_eigenvalues_cuda
from arz_model.core.parameters import ModelParameters
from arz_model.config.network_simulation_config import NetworkSimulationConfig

# Global counter for CFL correction messages
_cfl_correction_count = 0

# --- CUDA Kernel for Max Wavespeed Calculation ---

# Define block size for reduction (must be power of 2)
TPB_REDUCE = 256 # Threads per block for reduction kernel

@cuda.jit
def _calculate_max_wavespeed_kernel(d_U, n_ghost, n_phys,
                                    # Physics parameters needed for eigenvalues
                                    alpha, rho_max, epsilon, K_m, gamma_m, K_c, gamma_c,
                                    dx,
                                    # Output array (size 1) for the global max
                                    d_max_ratio_out):
    """
    Calculates the maximum ratio (lambda / dx) across all physical cells on the GPU
    and updates a global maximum.
    """
    # Shared memory for block-level reduction
    s_max_ratio = cuda.shared.array(shape=(TPB_REDUCE,), dtype=float64)

    # Global thread index and corresponding physical cell index
    idx_global = cuda.grid(1)
    phys_idx = idx_global

    # Local thread index within the block
    tx = cuda.threadIdx.x

    # Initialize shared memory for this thread
    s_max_ratio[tx] = 0.0

    # --- Calculate max ratio for cells handled by this thread ---
    max_ratio_thread = 0.0
    while phys_idx < n_phys:
        j_total = n_ghost + phys_idx # Index in the full d_U array

        # 1. Get state for the current physical cell
        rho_m, w_m, rho_c, w_c = d_U[0, j_total], d_U[1, j_total], d_U[2, j_total], d_U[3, j_total]

        # Ensure densities are non-negative
        rho_m_calc = max(rho_m, 0.0)
        rho_c_calc = max(rho_c, 0.0)

        # 2. Calculate intermediate values using device functions
        p_m, p_c = _calculate_pressure_cuda(rho_m_calc, rho_c_calc,
                                            alpha, rho_max, epsilon,
                                            K_m, gamma_m, K_c, gamma_c)
        v_m, v_c = _calculate_physical_velocity_cuda(w_m, w_c, p_m, p_c)

        # 3. Calculate eigenvalues using device function
        lambda1, lambda2, lambda3, lambda4 = _calculate_eigenvalues_cuda(
            rho_m_calc, v_m, rho_c_calc, v_c,
            alpha, rho_max, epsilon, K_m, gamma_m, K_c, gamma_c
        )

        # 4. Find max absolute eigenvalue for this cell
        max_lambda_cell = max(abs(lambda1), abs(lambda2), abs(lambda3), abs(lambda4))
        
        # 5. Calculate ratio for this cell
        ratio_cell = max_lambda_cell / dx if dx > 0 else 0.0

        # 6. Update the maximum for this thread
        max_ratio_thread = max(max_ratio_thread, ratio_cell)

        # Move to the next cell this thread is responsible for
        phys_idx += cuda.gridDim.x * cuda.blockDim.x

    # Store the thread's maximum in shared memory
    s_max_ratio[tx] = max_ratio_thread

    # Synchronize threads within the block
    cuda.syncthreads()

    # --- Perform reduction in shared memory ---
    stride = TPB_REDUCE // 2
    while stride > 0:
        if tx < stride:
            s_max_ratio[tx] = max(s_max_ratio[tx], s_max_ratio[tx + stride])
        cuda.syncthreads()
        stride //= 2

    # --- Write block's maximum to global memory ---
    if tx == 0:
        cuda.atomic.max(d_max_ratio_out, 0, s_max_ratio[0])



# --- Main Function ---

def calculate_cfl_dt(U, grid, params: 'NetworkSimulationConfig'):
    """
    Calculates the stable time step dt for a single segment based on the CFL condition.

    Args:
        U (np.ndarray): The state vector for the segment.
        grid (Grid1D): The grid for the segment.
        params (NetworkSimulationConfig): The simulation configuration object.

    Returns:
        float: The stable time step dt.
    """
    if U.shape[1] == 0:
        return float('inf')

    # Directly access physics parameters from the Pydantic config
    # No 'g' parameter in the current PhysicsConfig. The eigenvalue calculation
    # depends on other parameters like K_m, gamma_m, etc.
    # Let's pass the required ones to the eigenvalue function.
    alpha = params.physics.alpha
    rho_jam = params.physics.rho_jam / 1000.0
    epsilon = params.physics.epsilon
    K_m = params.physics.k_m
    gamma_m = params.physics.gamma_m
    K_c = params.physics.k_c
    gamma_c = params.physics.gamma_c

    rho_m, rho_c, v_m, v_c = U[0, :], U[1, :], U[2, :], U[3, :]
    dx = grid.dx
    
    # Calculate eigenvalues - pass the full params object
    phys = params.physics
    lambda1, lambda2, lambda3, lambda4 = calculate_eigenvalues(
        rho_m, v_m, rho_c, v_c, 
        phys.alpha, phys.rho_jam, phys.epsilon,
        phys.k_m, phys.gamma_m, phys.k_c, phys.gamma_c
    )
    
    max_abs_lambda = np.maximum(np.abs(lambda1), np.abs(lambda2))
    max_abs_lambda = np.maximum(max_abs_lambda, np.abs(lambda3))
    max_abs_lambda = np.maximum(max_abs_lambda, np.abs(lambda4))
    
    max_speed = np.max(max_abs_lambda)
    
    if max_speed == 0:
        return float('inf')
        
    return dx / max_speed


def cfl_condition(network: 'NetworkGrid') -> (float, str):
    """
    Calculates the CFL-stable time step for the entire network.

    Iterates over all segments, calculates the stable dt for each,
    and returns the minimum dt to ensure stability for the whole network.

    Args:
        network: The NetworkGrid object.

    Returns:
        A tuple containing:
        - The minimum stable dt for the network.
        - The ID of the segment that is limiting the time step.
    """
    min_dt = float('inf')
    limiting_segment = None

    for seg_id, segment_data in network.segments.items():
        U = segment_data['U']
        grid = segment_data['grid']
        
        # ARCHITECTURAL FIX (2025-11-04): The concept of per-segment 'params' is deprecated.
        # The simulation now operates with a single, unified configuration object.
        # Always use the config from the parent network object to ensure consistency.
        params = network.simulation_config
        if params is None:
            raise ValueError(f"FATAL: NetworkGrid.simulation_config is not set. Cannot run simulation.")

        # Get physical cells
        physical_U = U[:, grid.physical_cell_indices]
        
        # Calculate stable dt for this segment
        dt_segment = calculate_cfl_dt(physical_U, grid, params)
        
        if dt_segment < min_dt:
            min_dt = dt_segment
            limiting_segment = seg_id
            
    return min_dt, limiting_segment


def cfl_condition_gpu_native(gpu_pool: 'GPUMemoryPool', network: 'NetworkGrid', params: 'ModelParameters', cfl_max: float) -> float:
    """
    Calculates the maximum stable time step (dt) across all segments on the GPU.

    This function orchestrates the following steps:
    1. Launches a kernel on each segment to find its local maximum eigenvalue.
    2. Reduces the results on the GPU to find the global maximum eigenvalue.
    3. Calculates and returns the stable dt.

    Args:
        gpu_pool: The GPUMemoryPool containing the state of all segments.
        network: The NetworkGrid object to get grid information (dx).
        params: The model parameters.
        cfl_max: The maximum CFL number.

    Returns:
        The calculated stable time step.
    """
    # Create a single-element device array to store the global maximum of (lambda / dx)
    d_global_max_ratio = cuda.to_device(np.array([0.0], dtype=np.float64))
    
    # params is already the PhysicsConfig object, no need for .physics accessor
    phys_params = params

    for seg_id in gpu_pool.segment_ids:
        d_U = gpu_pool.get_segment_state(seg_id)
        grid = network.segments[seg_id]['grid']
        
        threadsperblock = TPB_REDUCE
        blockspergrid = (grid.N_physical + (threadsperblock - 1)) // threadsperblock

        # Launch kernel for each segment
        _calculate_max_wavespeed_kernel[blockspergrid, threadsperblock](
            d_U, grid.num_ghost_cells, grid.N_physical,
            phys_params.alpha, phys_params.rho_jam, phys_params.epsilon,
            phys_params.k_m, phys_params.gamma_m, phys_params.k_c, phys_params.gamma_c,
            grid.dx,
            d_global_max_ratio
        )

    # Copy the final result back to the host
    global_max_ratio = d_global_max_ratio.copy_to_host()[0]

    if global_max_ratio < 1e-9:
        # If max speed is near zero, can use a large dt, but not infinite
        return 1.0
    
    # dt = CFL / max(lambda/dx)
    stable_dt = cfl_max / global_max_ratio
    
    return stable_dt
