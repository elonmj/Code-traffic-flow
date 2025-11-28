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
                                    alpha, rho_max, epsilon, v_max, 
                                    K_m, gamma_m, K_c, gamma_c,
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

        # Robustly clamp densities to physical bounds [epsilon, rho_max]
        rho_m_calc = min(max(rho_m, epsilon), rho_max)
        rho_c_calc = min(max(rho_c, epsilon), rho_max)

        # Handle potential NaN/inf values from previous steps
        if not math.isfinite(rho_m_calc):
            rho_m_calc = 0.0  # Fallback to vacuum state
        if not math.isfinite(rho_c_calc):
            rho_c_calc = 0.0

        # 2. Calculate intermediate values using device functions
        p_m, p_c = _calculate_pressure_cuda(rho_m_calc, rho_c_calc,
                                            alpha, rho_max, epsilon,
                                            K_m, gamma_m, K_c, gamma_c)
        v_m, v_c = _calculate_physical_velocity_cuda(w_m, w_c, p_m, p_c)

        # Clamp physical velocities
        v_m = min(max(v_m, -v_max), v_max)
        v_c = min(max(v_c, -v_max), v_max)

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


# --- Batched CFL Kernel ---

@cuda.jit
def _calculate_max_wavespeed_batched_kernel(d_U_batched, d_segment_lengths, d_batched_offsets,
                                             num_segments,
                                             # Physics parameters
                                             alpha, rho_max, epsilon, v_max,
                                             K_m, gamma_m, K_c, gamma_c,
                                             dx,
                                             # Per-segment output array (size: num_segments)
                                             d_segment_max_ratios):
    """
    Batched kernel: Each block processes one segment, calculates max(lambda/dx) for that segment.
    
    Grid: num_segments blocks
    Block: TPB_REDUCE threads
    
    Each block writes its segment's max ratio to d_segment_max_ratios[blockIdx.x].
    CPU-side reduction then finds global max.
    """
    # Shared memory for block-level reduction
    s_max_ratio = cuda.shared.array(shape=(TPB_REDUCE,), dtype=float64)
    
    # Block ID = segment index
    seg_idx = cuda.blockIdx.x
    if seg_idx >= num_segments:
        return
    
    # Thread ID within block
    tx = cuda.threadIdx.x
    
    # Get segment bounds
    start_idx = d_batched_offsets[seg_idx]
    seg_length = d_segment_lengths[seg_idx]
    
    # Initialize shared memory
    s_max_ratio[tx] = 0.0
    
    # Each thread processes multiple cells (grid-stride loop)
    max_ratio_thread = 0.0
    cell_idx = tx
    
    while cell_idx < seg_length:
        global_idx = start_idx + cell_idx
        
        # Get state (batched layout: [total_cells, 4])
        rho_m = d_U_batched[global_idx, 0]
        w_m = d_U_batched[global_idx, 1]
        rho_c = d_U_batched[global_idx, 2]
        w_c = d_U_batched[global_idx, 3]
        
        # Clamp densities
        rho_m_calc = min(max(rho_m, epsilon), rho_max)
        rho_c_calc = min(max(rho_c, epsilon), rho_max)
        
        # Handle NaN/inf
        if not math.isfinite(rho_m_calc):
            rho_m_calc = 0.0
        if not math.isfinite(rho_c_calc):
            rho_c_calc = 0.0
        
        # Calculate pressure
        p_m, p_c = _calculate_pressure_cuda(rho_m_calc, rho_c_calc,
                                             alpha, rho_max, epsilon,
                                             K_m, gamma_m, K_c, gamma_c)
        
        # Calculate velocities
        v_m, v_c = _calculate_physical_velocity_cuda(w_m, w_c, p_m, p_c)
        
        # Clamp velocities
        v_m = min(max(v_m, -v_max), v_max)
        v_c = min(max(v_c, -v_max), v_max)
        
        # Calculate eigenvalues
        lambda1, lambda2, lambda3, lambda4 = _calculate_eigenvalues_cuda(
            rho_m_calc, v_m, rho_c_calc, v_c,
            alpha, rho_max, epsilon, K_m, gamma_m, K_c, gamma_c
        )
        
        # Max absolute eigenvalue
        max_lambda_cell = max(abs(lambda1), abs(lambda2), abs(lambda3), abs(lambda4))
        
        # Ratio
        ratio_cell = max_lambda_cell / dx if dx > 0 else 0.0
        
        # Update thread max
        max_ratio_thread = max(max_ratio_thread, ratio_cell)
        
        # Next cell for this thread
        cell_idx += cuda.blockDim.x
    
    # Store thread max in shared memory
    s_max_ratio[tx] = max_ratio_thread
    cuda.syncthreads()
    
    # Reduction within block
    stride = TPB_REDUCE // 2
    while stride > 0:
        if tx < stride:
            s_max_ratio[tx] = max(s_max_ratio[tx], s_max_ratio[tx + stride])
        cuda.syncthreads()
        stride //= 2
    
    # Block leader writes segment max to output
    if tx == 0:
        d_segment_max_ratios[seg_idx] = s_max_ratio[0]


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

    # State vector U = [rho_moto, v_moto, rho_car, v_car]
    rho_m = U[0, :]  # Motos density
    v_m = U[1, :]    # Motos velocity
    rho_c = U[2, :]  # Cars density
    v_c = U[3, :]    # Cars velocity
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


def cfl_condition_gpu_native(gpu_pool: 'GPUMemoryPool', network: 'NetworkGrid', params: 'ModelParameters', cfl_max: float, return_diagnostics: bool = False):
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
        return_diagnostics: If True, returns (dt, diagnostics_dict) instead of just dt.

    Returns:
        The calculated stable time step, or (dt, diagnostics) if return_diagnostics=True.
    """
    # Create a single-element device array to store the global maximum of (lambda / dx)
    d_global_max_ratio = cuda.to_device(np.array([0.0], dtype=np.float64))
    
    # params is already the PhysicsConfig object, no need for .physics accessor
    phys_params = params
    v_max_physical = max(phys_params.v_max_m_kmh, phys_params.v_max_c_kmh) / 3.6
    
    # For diagnostics: track per-segment ratios
    segment_diagnostics = {}

    for seg_id in gpu_pool.segment_ids:
        d_U = gpu_pool.get_segment_state(seg_id)
        grid = network.segments[seg_id]['grid']
        
        threadsperblock = TPB_REDUCE
        blockspergrid = (grid.N_physical + (threadsperblock - 1)) // threadsperblock

        # If diagnostics requested, compute per-segment max_ratio
        if return_diagnostics:
            d_seg_max_ratio = cuda.to_device(np.array([0.0], dtype=np.float64))
            
            _calculate_max_wavespeed_kernel[blockspergrid, threadsperblock](
                d_U, grid.num_ghost_cells, grid.N_physical,
                phys_params.alpha, phys_params.rho_max, phys_params.epsilon, v_max_physical,
                phys_params.k_m, phys_params.gamma_m, phys_params.k_c, phys_params.gamma_c,
                grid.dx,
                d_seg_max_ratio
            )
            
            seg_max_ratio = d_seg_max_ratio.copy_to_host()[0]
            
            # Also get state statistics for this segment
            U_cpu = gpu_pool.checkpoint_to_cpu(seg_id)
            phys_indices = grid.physical_cell_indices
            
            rho_m = U_cpu[0, phys_indices]
            w_m = U_cpu[1, phys_indices]
            rho_c = U_cpu[2, phys_indices]
            w_c = U_cpu[3, phys_indices]
            
            # Calculate velocities (simplified, just for diagnostics)
            # v ≈ w for now (ignoring pressure correction for speed)
            
            segment_diagnostics[seg_id] = {
                'max_ratio': seg_max_ratio,
                'dx': grid.dx,
                'rho_m': {'min': np.min(rho_m), 'max': np.max(rho_m), 'mean': np.mean(rho_m)},
                'rho_c': {'min': np.min(rho_c), 'max': np.max(rho_c), 'mean': np.mean(rho_c)},
                'w_m': {'min': np.min(w_m), 'max': np.max(w_m), 'mean': np.mean(w_m)},
                'w_c': {'min': np.min(w_c), 'max': np.max(w_c), 'mean': np.mean(w_c)},
                'dt_seg': cfl_max / seg_max_ratio if seg_max_ratio > 1e-9 else 1.0
            }
            
            # Update global max using host-side comparison
            # (cuda.atomic.max can only be called from within a kernel)
            current_global = d_global_max_ratio.copy_to_host()[0]
            if seg_max_ratio > current_global:
                d_global_max_ratio[:] = cuda.to_device(np.array([seg_max_ratio], dtype=np.float64))
        else:
            # Normal operation without diagnostics
            _calculate_max_wavespeed_kernel[blockspergrid, threadsperblock](
                d_U, grid.num_ghost_cells, grid.N_physical,
                phys_params.alpha, phys_params.rho_max, phys_params.epsilon, v_max_physical,
                phys_params.k_m, phys_params.gamma_m, phys_params.k_c, phys_params.gamma_c,
                grid.dx,
                d_global_max_ratio
            )

    # Copy the final result back to the host
    global_max_ratio = d_global_max_ratio.copy_to_host()[0]

    if global_max_ratio < 1e-9:
        # If max speed is near zero, can use a large dt, but not infinite
        stable_dt = 1.0
    else:
        # dt = CFL / max(lambda/dx)
        stable_dt = cfl_max / global_max_ratio
    
    if return_diagnostics:
        diagnostics = {
            'global_max_ratio': global_max_ratio,
            'stable_dt': stable_dt,
            'segments': segment_diagnostics
        }
        return stable_dt, diagnostics
    else:
        return stable_dt


def cfl_condition_gpu_batched(gpu_pool: 'GPUMemoryPool', dx: float, params: 'ModelParameters', cfl_max: float):
    """
    Calculates the maximum stable time step (dt) for batched GPU architecture.
    
    This function uses the batched arrays and launches ONE kernel with num_segments blocks,
    where each block processes one segment in parallel. This eliminates the per-segment loop.
    
    Args:
        gpu_pool: The GPUMemoryPool containing batched state arrays.
        dx: Uniform grid spacing (Victoria Island: 25.0m).
        params: The model parameters (PhysicsConfig).
        cfl_max: The maximum CFL number.
    
    Returns:
        The calculated stable time step.
    
    Performance:
        - Single kernel launch vs 70 sequential launches
        - Each block computes max(lambda/dx) for its segment
        - CPU-side reduction finds global max
    
    References:
        - .copilot-tracking/research/20251117-gpu-occupancy-warning-research.md
        - .copilot-tracking/plans/20251117-gpu-batching-architecture-plan.instructions.md
    """
    # Get batched arrays (order: U, R, offsets, lengths, light_factors)
    d_U_batched, _, d_batched_offsets, d_segment_lengths, _ = gpu_pool.get_batched_arrays()
    num_segments = len(gpu_pool.segment_ids)
    
    # Allocate output array for per-segment max ratios
    d_segment_max_ratios = cuda.device_array(num_segments, dtype=np.float64)
    
    # Physics parameters
    phys_params = params
    v_max_physical = max(phys_params.v_max_m_kmh, phys_params.v_max_c_kmh) / 3.6
    
    # Launch batched kernel: grid=num_segments, block=TPB_REDUCE
    # Each block processes one segment
    _calculate_max_wavespeed_batched_kernel[num_segments, TPB_REDUCE](
        d_U_batched, d_segment_lengths, d_batched_offsets, num_segments,
        phys_params.alpha, phys_params.rho_max, phys_params.epsilon, v_max_physical,
        phys_params.k_m, phys_params.gamma_m, phys_params.k_c, phys_params.gamma_c,
        dx,
        d_segment_max_ratios
    )
    
    # CPU-side reduction: find max across all segments
    segment_ratios_host = d_segment_max_ratios.copy_to_host()
    global_max_ratio = np.max(segment_ratios_host)
    
    if global_max_ratio < 1e-9:
        # Near-zero max speed
        stable_dt = 1.0
    else:
        # dt = CFL / max(lambda/dx)
        stable_dt = cfl_max / global_max_ratio
    
    return stable_dt


def compute_adaptive_cfl_with_history(
    dt_history: list[float], 
    base_cfl: float = 0.8, 
    n_window: int = 10, 
    threshold: float = 0.05
) -> float:
    """
    Compute adaptive CFL factor based on recent dt history.
    
    Reduces CFL from base_cfl to 0.5 if dt has been persistently small,
    indicating numerical stress. This gives more stability margin.
    
    Args:
        dt_history: List of recent dt values (most recent last)
        base_cfl: Nominal CFL factor when system is stable (default 0.8)
        n_window: Number of recent steps to check (default 10)
        threshold: dt value considered 'collapsed' (default 0.05s)
    
    Returns:
        Adapted CFL factor: 0.5 if unstable, else base_cfl
        
    Example:
        >>> dt_hist = [0.02, 0.03, 0.01, 0.02, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02]
        >>> compute_adaptive_cfl_with_history(dt_hist, base_cfl=0.8, threshold=0.05)
        0.5  # All values < 0.05, so reduce CFL
        
    References:
        - #file:../research/20251116-numerical-stability-arz-dt-collapse-research.md
    """
    # Need enough history to make decision
    if len(dt_history) < n_window:
        return base_cfl
    
    # Check last n_window steps
    recent_dts = dt_history[-n_window:]
    collapsed_count = sum(1 for dt in recent_dts if dt < threshold)
    
    # If 50%+ of recent steps are collapsed, reduce CFL
    if collapsed_count >= n_window // 2:
        return 0.5  # Conservative CFL for stressed systems
    
    return base_cfl  # Normal CFL
