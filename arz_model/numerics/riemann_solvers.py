import numpy as np
import numba # Import numba itself
from numba import cuda, float64 # Keep float64 import for convenience elsewhere? Or remove if not used.
import math # Import math for CUDA device functions
from typing import Optional
from ..core.parameters import ModelParameters
from ..core import physics # Import the physics module itself
# Import specific CUDA device functions from physics
from ..core.physics import (
    _calculate_pressure_cuda,
    _calculate_physical_velocity_cuda,
    _calculate_eigenvalues_cuda
)

# Import for junction-aware flux blocking
try:
    from ..network.junction_info import JunctionInfo
except ImportError:
    JunctionInfo = None

def central_upwind_flux(
    U_L: np.ndarray, 
    U_R: np.ndarray, 
    params: ModelParameters,
    junction_info: Optional['JunctionInfo'] = None
) -> np.ndarray:
    """
    Calculates the numerical flux at the interface between states U_L and U_R
    using the first-order Central-Upwind scheme (Kurganov-Tadmor type).

    Handles the non-conservative form of the w_i equations approximately
    by defining a flux F(U) = (rho_m*v_m, w_m, rho_c*v_c, w_c)^T for the
    calculation within the CU formula.
    
    Junction-aware flux blocking: When junction_info is provided with a traffic signal,
    the computed flux is reduced by light_factor to physically block flow during RED signals.
    Based on Daganzo (1995) supply-demand junction model adapted to numerical flux calculation.

    Args:
        U_L (np.ndarray): State vector [rho_m, w_m, rho_c, w_c] to the left of the interface (SI units).
        U_R (np.ndarray): State vector [rho_m, w_m, rho_c, w_c] to the right of the interface (SI units).
        params (ModelParameters): Model parameters object.
        junction_info (Optional[JunctionInfo]): Junction metadata for flux blocking.
            If None, normal flux calculation (backward compatible).
            If provided with is_junction=True, flux is multiplied by light_factor:
                - RED signal: light_factor ≈ 0.01 (1% flow, 99% blocked)
                - GREEN signal: light_factor = 1.0 (100% flow, no blocking)

    Returns:
        np.ndarray: The numerical flux vector F_CU at the interface. Shape (4,).
        
    References:
        - Kurganov & Tadmor (2000): New high-resolution central schemes for nonlinear conservation laws
        - Daganzo (1995): The cell transmission model, part II: Network traffic
        
    Example:
        >>> # Normal flux calculation (no junction)
        >>> F = central_upwind_flux(U_L, U_R, params)
        >>> 
        >>> # Junction with RED signal (99% blocked)
        >>> junction = JunctionInfo(is_junction=True, light_factor=0.01, node_id=1)
        >>> F_red = central_upwind_flux(U_L, U_R, params, junction)
        >>> # F_red ≈ 0.01 * F (flux reduced by 99%)
    """
    # Ensure inputs are numpy arrays
    U_L = np.asarray(U_L)
    U_R = np.asarray(U_R)

    # Extract states
    rho_m_L, w_m_L, rho_c_L, w_c_L = U_L
    rho_m_R, w_m_R, rho_c_R, w_c_R = U_R

    # Ensure densities are non-negative for calculations
    rho_m_L_calc = max(rho_m_L, 0.0)
    rho_c_L_calc = max(rho_c_L, 0.0)
    rho_m_R_calc = max(rho_m_R, 0.0)
    rho_c_R_calc = max(rho_c_R, 0.0)

    # Calculate pressures and velocities for L and R states
    p_m_L, p_c_L = physics.calculate_pressure(rho_m_L_calc, rho_c_L_calc,
                                              params.alpha, params.rho_jam, params.epsilon,
                                              params.K_m, params.gamma_m,
                                              params.K_c, params.gamma_c)
    v_m_L, v_c_L = physics.calculate_physical_velocity(w_m_L, w_c_L, p_m_L, p_c_L)

    p_m_R, p_c_R = physics.calculate_pressure(rho_m_R_calc, rho_c_R_calc,
                                              params.alpha, params.rho_jam, params.epsilon,
                                              params.K_m, params.gamma_m,
                                              params.K_c, params.gamma_c)
    v_m_R, v_c_R = physics.calculate_physical_velocity(w_m_R, w_c_R, p_m_R, p_c_R)

    # Calculate eigenvalues for L and R states
    # Note: physics.calculate_eigenvalues expects arrays, so pass scalars wrapped
    lambda_L_list = physics.calculate_eigenvalues(np.array([rho_m_L_calc]), np.array([v_m_L]),
                                                 np.array([rho_c_L_calc]), np.array([v_c_L]), params)
    lambda_R_list = physics.calculate_eigenvalues(np.array([rho_m_R_calc]), np.array([v_m_R]),
                                                 np.array([rho_c_R_calc]), np.array([v_c_R]), params)
    # Flatten the list of single-element arrays back to scalars for max/min
    lambda_L = [l[0] for l in lambda_L_list]
    lambda_R = [l[0] for l in lambda_R_list]


    # Calculate local one-sided wave speeds (a+ and a-)
    a_plus = max(max(lambda_L, default=0), max(lambda_R, default=0), 0.0)
    a_minus = min(min(lambda_L, default=0), min(lambda_R, default=0), 0.0)

    # Define the approximate physical flux F(U) = (rho_m*v_m, w_m, rho_c*v_c, w_c)^T
    # Note: This treats w_m and w_c as if they were part of a conserved quantity flux.
    # This is an approximation necessary for applying the CU formula directly.
    F_L = np.array([rho_m_L_calc * v_m_L, w_m_L, rho_c_L_calc * v_c_L, w_c_L])
    F_R = np.array([rho_m_R_calc * v_m_R, w_m_R, rho_c_R_calc * v_c_R, w_c_R])

    # Calculate the Central-Upwind numerical flux
    denominator = a_plus - a_minus
    if abs(denominator) < params.epsilon:
        # Handle case where a+ approx equals a- (e.g., vacuum state or zero speeds)
        # In this case, the flux is often taken as the average or simply F(U_L) or F(U_R).
        # Let's use the average as a reasonable default.
        F_CU = 0.5 * (F_L + F_R)
    else:
        term1 = (a_plus * F_L - a_minus * F_R) / denominator
        term2 = (a_plus * a_minus / denominator) * (U_R - U_L)
        F_CU = term1 + term2

    # Apply junction-aware flux blocking if at junction interface
    if junction_info is not None and junction_info.is_junction:
        # [PHASE 3 DEBUG - Junction flux blocking verification]
        if junction_info.light_factor < 0.5:  # RED or YELLOW signal
            print("[JUNCTION FLUX BLOCKING]")
            print("  light_factor =", junction_info.light_factor)
            print("  F_before (momentum) =", F_CU[1])
        
        F_CU = F_CU * junction_info.light_factor
        
        if junction_info.light_factor < 0.5:
            print("  F_after (momentum, blocked) =", F_CU[1])
    
    return F_CU


def godunov_flux_upwind(
    U_L: np.ndarray,
    U_R: np.ndarray,
    params: ModelParameters,
    junction_info: Optional['JunctionInfo'] = None
) -> np.ndarray:
    """
    Godunov flux via upwind selection (Phase 1 - simplified).
    
    This is a robust first-order Riemann solver that selects the flux based on
    wave speeds (eigenvalues). For the ARZ multi-class traffic model:
    
    - If all waves move right (λ_min ≥ 0): Use left flux F(U_L)
    - If all waves move left (λ_max ≤ 0): Use right flux F(U_R)
    - Mixed waves: Fallback to central_upwind_flux
    
    Advantages:
    - Positivity-preserving (monotone scheme)
    - Robust with sharp discontinuities (BCs, shocks)
    - No Gibbs oscillations
    
    Disadvantages:
    - First-order accuracy (more diffusive than WENO5)
    
    Junction-aware flux blocking: Same as central_upwind_flux, the computed
    flux is reduced by light_factor when junction_info indicates a traffic signal.
    
    Args:
        U_L (np.ndarray): State vector [rho_m, w_m, rho_c, w_c] left of interface (SI units).
        U_R (np.ndarray): State vector [rho_m, w_m, rho_c, w_c] right of interface (SI units).
        params (ModelParameters): Model parameters object.
        junction_info (Optional[JunctionInfo]): Junction metadata for flux blocking.
    
    Returns:
        np.ndarray: The numerical flux vector F at the interface. Shape (4,).
        
    References:
        - Godunov (1959): A difference method for numerical calculation of discontinuous solutions
        - Mammar et al. (2009): Riemann solver for ARZ model
        - Villa (2016): ARZ with traffic lights (arXiv:1605.00632)
    
    Example:
        >>> # Normal flux calculation
        >>> F = godunov_flux_upwind(U_L, U_R, params)
        >>> 
        >>> # Junction with RED signal (99% blocked)
        >>> junction = JunctionInfo(is_junction=True, light_factor=0.01, node_id=1)
        >>> F_red = godunov_flux_upwind(U_L, U_R, params, junction)
    """
    # Ensure inputs are numpy arrays
    U_L = np.asarray(U_L)
    U_R = np.asarray(U_R)
    
    # 1. Extract and clamp densities
    rho_m_L = max(U_L[0], 0.0)
    rho_c_L = max(U_L[2], 0.0)
    rho_m_R = max(U_R[0], 0.0)
    rho_c_R = max(U_R[2], 0.0)
    
    # 2. Calculate pressures (REUSE physics.py)
    p_m_L, p_c_L = physics.calculate_pressure(
        np.array([rho_m_L]), np.array([rho_c_L]),
        params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c
    )
    p_m_R, p_c_R = physics.calculate_pressure(
        np.array([rho_m_R]), np.array([rho_c_R]),
        params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c
    )
    
    # 3. Calculate velocities
    v_m_L, v_c_L = physics.calculate_physical_velocity(U_L[1], U_L[3], p_m_L[0], p_c_L[0])
    v_m_R, v_c_R = physics.calculate_physical_velocity(U_R[1], U_R[3], p_m_R[0], p_c_R[0])
    
    # 4. Calculate eigenvalues (REUSE physics.py)
    lambda_L_list = physics.calculate_eigenvalues(
        np.array([rho_m_L]), np.array([v_m_L]),
        np.array([rho_c_L]), np.array([v_c_L]),
        params
    )
    lambda_R_list = physics.calculate_eigenvalues(
        np.array([rho_m_R]), np.array([v_m_R]),
        np.array([rho_c_R]), np.array([v_c_R]),
        params
    )
    
    # 5. Flatten (calculate_eigenvalues returns list of arrays)
    all_lambda_L = [float(l[0]) for l in lambda_L_list]
    all_lambda_R = [float(l[0]) for l in lambda_R_list]
    
    # 6. Upwind selection
    lambda_min = min(min(all_lambda_L), min(all_lambda_R))
    lambda_max = max(max(all_lambda_L), max(all_lambda_R))
    
    if lambda_min >= 0.0:
        # All waves move right → flux = F(U_L)
        F = np.array([
            rho_m_L * v_m_L,
            U_L[1],
            rho_c_L * v_c_L,
            U_L[3]
        ])
    elif lambda_max <= 0.0:
        # All waves move left → flux = F(U_R)
        F = np.array([
            rho_m_R * v_m_R,
            U_R[1],
            rho_c_R * v_c_R,
            U_R[3]
        ])
    else:
        # Mixed waves → fallback to Central-Upwind (handles complex wave interactions)
        F = central_upwind_flux(U_L, U_R, params)
    
    # 7. Junction blocking (REUSE logic)
    if junction_info is not None and junction_info.is_junction:
        F = F * junction_info.light_factor
    
    return F


# --- CUDA Device Function for Central-Upwind Flux ---

@cuda.jit(device=True)
def _central_upwind_flux_cuda(U_L_i, U_R_i,
                              alpha, rho_jam, epsilon,
                              K_m, gamma_m, K_c, gamma_c):
    """
    CUDA device function to calculate the numerical flux at a single interface
    using the first-order Central-Upwind scheme. Returns the flux components as a tuple.

    Args:
        U_L_i (tuple/array-like): State vector [rho_m, w_m, rho_c, w_c] left of interface.
        U_R_i (tuple/array-like): State vector [rho_m, w_m, rho_c, w_c] right of interface.
        alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c (float): Model parameters.

    Returns:
        tuple[float, float, float, float]: The four components of the numerical flux vector F_CU.
    """
    # Extract states (assuming U_L_i, U_R_i are tuples or array-like)
    rho_m_L, w_m_L, rho_c_L, w_c_L = U_L_i[0], U_L_i[1], U_L_i[2], U_L_i[3]
    rho_m_R, w_m_R, rho_c_R, w_c_R = U_R_i[0], U_R_i[1], U_R_i[2], U_R_i[3]

    # Ensure densities are non-negative for calculations
    rho_m_L_calc = max(rho_m_L, 0.0)
    rho_c_L_calc = max(rho_c_L, 0.0)
    rho_m_R_calc = max(rho_m_R, 0.0)
    rho_c_R_calc = max(rho_c_R, 0.0)

    # Calculate pressures and velocities for L and R states using CUDA device functions
    p_m_L, p_c_L = _calculate_pressure_cuda(rho_m_L_calc, rho_c_L_calc,
                                            alpha, rho_jam, epsilon,
                                            K_m, gamma_m, K_c, gamma_c)
    v_m_L, v_c_L = _calculate_physical_velocity_cuda(w_m_L, w_c_L, p_m_L, p_c_L)

    p_m_R, p_c_R = _calculate_pressure_cuda(rho_m_R_calc, rho_c_R_calc,
                                            alpha, rho_jam, epsilon,
                                            K_m, gamma_m, K_c, gamma_c)
    v_m_R, v_c_R = _calculate_physical_velocity_cuda(w_m_R, w_c_R, p_m_R, p_c_R)

    # Calculate eigenvalues for L and R states using CUDA device function
    lambda1_L, lambda2_L, lambda3_L, lambda4_L = _calculate_eigenvalues_cuda(
        rho_m_L_calc, v_m_L, rho_c_L_calc, v_c_L,
        alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c
    )
    lambda1_R, lambda2_R, lambda3_R, lambda4_R = _calculate_eigenvalues_cuda(
        rho_m_R_calc, v_m_R, rho_c_R_calc, v_c_R,
        alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c
    )

    # Calculate local one-sided wave speeds (a+ and a-)
    # Note: Numba CUDA device functions don't have default arguments for max/min
    max_lambda_L = max(lambda1_L, max(lambda2_L, max(lambda3_L, lambda4_L)))
    max_lambda_R = max(lambda1_R, max(lambda2_R, max(lambda3_R, lambda4_R)))
    min_lambda_L = min(lambda1_L, min(lambda2_L, min(lambda3_L, lambda4_L)))
    min_lambda_R = min(lambda1_R, min(lambda2_R, min(lambda3_R, lambda4_R)))

    a_plus = max(max_lambda_L, max_lambda_R, 0.0)
    a_minus = min(min_lambda_L, min_lambda_R, 0.0)

    # Define the approximate physical flux F(U) = (rho_m*v_m, w_m, rho_c*v_c, w_c)^T
    # Cannot create numpy arrays inside device function, use local variables or tuples if needed
    F_L_0 = rho_m_L_calc * v_m_L
    F_L_1 = w_m_L
    F_L_2 = rho_c_L_calc * v_c_L
    F_L_3 = w_c_L

    F_R_0 = rho_m_R_calc * v_m_R
    F_R_1 = w_m_R
    F_R_2 = rho_c_R_calc * v_c_R
    F_R_3 = w_c_R

    # Calculate the Central-Upwind numerical flux
    denominator = a_plus - a_minus
    # Declare local variables for flux components
    f0, f1, f2, f3 = 0.0, 0.0, 0.0, 0.0
    if abs(denominator) < epsilon:
        # Handle case where a+ approx equals a-
        f0 = 0.5 * (F_L_0 + F_R_0)
        f1 = 0.5 * (F_L_1 + F_R_1)
        f2 = 0.5 * (F_L_2 + F_R_2)
        f3 = 0.5 * (F_L_3 + F_R_3)
    else:
        inv_denominator = 1.0 / denominator
        factor = a_plus * a_minus * inv_denominator

        f0 = (a_plus * F_L_0 - a_minus * F_R_0) * inv_denominator + factor * (U_R_i[0] - U_L_i[0])
        f1 = (a_plus * F_L_1 - a_minus * F_R_1) * inv_denominator + factor * (U_R_i[1] - U_L_i[1])
        f2 = (a_plus * F_L_2 - a_minus * F_R_2) * inv_denominator + factor * (U_R_i[2] - U_L_i[2])
        f3 = (a_plus * F_L_3 - a_minus * F_R_3) * inv_denominator + factor * (U_R_i[3] - U_L_i[3])

    return f0, f1, f2, f3


# --- CUDA Kernel Wrapper for Central-Upwind Flux ---

# Define constants for shared memory allocation
NUM_VARS = 4   # Number of state variables (must be known at compile time)
TPB_FLUX = 256 # Threads per block for flux kernel (must be known at compile time)
SHARED_MEM_COLS = TPB_FLUX + 1 # Calculate derived constant

@cuda.jit
def central_upwind_flux_cuda_kernel(d_U_in,
                                    alpha, rho_jam, epsilon,
                                    K_m, gamma_m, K_c, gamma_c,
                                    light_factor,
                                    d_F_CU_out):
    """
    CUDA kernel to calculate the Central-Upwind flux for all interfaces using shared memory.
    Each thread calculates the flux for one interface idx (between cell idx and idx+1).
    
    Junction-aware: light_factor parameter enables traffic signal flux blocking.
    - light_factor = 1.0: GREEN signal (normal flow)
    - light_factor ≈ 0.01: RED signal (99% blocked flow)
    """
    # Shared memory for U state: NUM_VARS variables, TPB threads + 1 extra cell for right neighbor
    s_U = cuda.shared.array(shape=(NUM_VARS, SHARED_MEM_COLS), dtype=numba.float64) # Use constants for shape

    # Global thread index (corresponds to the *left* cell index for the interface)
    idx = cuda.grid(1)
    # Local thread index
    tx = cuda.threadIdx.x
    # Block ID
    bx = cuda.blockIdx.x
    # Block width
    bw = cuda.blockDim.x # Should be TPB_FLUX

    N_total = d_U_in.shape[1]

    # --- Load data into shared memory ---
    # Each thread loads its corresponding cell state U[:, idx] into s_U[:, tx]
    if idx < N_total:
        s_U[0, tx] = d_U_in[0, idx]
        s_U[1, tx] = d_U_in[1, idx]
        s_U[2, tx] = d_U_in[2, idx]
        s_U[3, tx] = d_U_in[3, idx]

    # The last thread in the block needs to load the state for the cell to its right
    # This cell's global index is blockDim.x * (blockIdx.x + 1)
    # Or simply idx + 1 for the last thread if idx = blockDim.x * (blockIdx.x + 1) - 1
    if tx == bw - 1:
        idx_right_neighbor = idx + 1
        if idx_right_neighbor < N_total:
            s_U[0, tx + 1] = d_U_in[0, idx_right_neighbor]
            s_U[1, tx + 1] = d_U_in[1, idx_right_neighbor]
            s_U[2, tx + 1] = d_U_in[2, idx_right_neighbor]
            s_U[3, tx + 1] = d_U_in[3, idx_right_neighbor]
        # else: # Handle boundary case if needed (e.g., load zeros or extrapolate)
            # For now, assume subsequent steps handle boundary fluxes correctly
            # and we don't need to explicitly load beyond N_total-1 here.
            # If idx_right_neighbor == N_total, s_U[:, tx+1] remains uninitialized.

    # Synchronize threads within the block to ensure all shared memory is loaded
    cuda.syncthreads()

    # --- Calculate flux using shared memory ---
    # Each thread calculates flux at interface 'idx' (between cell idx and idx+1)
    # Check bounds: We need U_L=s_U[:,tx] and U_R=s_U[:,tx+1]
    # The kernel calculates fluxes for interfaces 0 to N_total-2
    if idx < N_total - 1:
        # Get U_L and U_R from shared memory
        # Need temporary arrays or tuples to pass to device function if it expects array-like
        U_L_s = (s_U[0, tx], s_U[1, tx], s_U[2, tx], s_U[3, tx])
        U_R_s = (s_U[0, tx + 1], s_U[1, tx + 1], s_U[2, tx + 1], s_U[3, tx + 1])

        # Call the device function, which now returns a tuple
        f0, f1, f2, f3 = _central_upwind_flux_cuda(U_L_s, U_R_s, # Pass tuples
                                                   alpha, rho_jam, epsilon,
                                                   K_m, gamma_m, K_c, gamma_c)

        # Apply junction blocking (same as CPU version)
        f0 *= light_factor
        f1 *= light_factor
        f2 *= light_factor
        f3 *= light_factor

        # Write the result components directly to global memory
        d_F_CU_out[0, idx] = f0
        d_F_CU_out[1, idx] = f1
        d_F_CU_out[2, idx] = f2
        d_F_CU_out[3, idx] = f3

    # Note: Flux at interface N_total-1 is not calculated.


# --- Wrapper function to call the CUDA kernel ---

def central_upwind_flux_gpu(d_U_in: cuda.devicearray.DeviceNDArray, params: ModelParameters, light_factor: float = 1.0) -> cuda.devicearray.DeviceNDArray:
    """
    Calculates the numerical flux at all interfaces using the Central-Upwind scheme on the GPU.
    Operates entirely on GPU arrays.

    Args:
        d_U_in (cuda.devicearray.DeviceNDArray): Input state device array (including ghost cells). Shape (4, N_total).
        params (ModelParameters): Model parameters object.
        light_factor (float): Junction blocking factor (1.0 = GREEN/normal, 0.01 = RED/blocked). Default 1.0.

    Returns:
        cuda.devicearray.DeviceNDArray: The numerical flux vectors F_CU at all interfaces. Shape (4, N_total) on the GPU.
                                         The flux at index j corresponds to the interface between cell j and j+1.
                                         The last column (interface N_total-1) is not calculated by the kernel.
    """
    if not cuda.is_cuda_array(d_U_in):
        raise TypeError("Input d_U_in must be a Numba CUDA device array.")

    N_total = d_U_in.shape[1]

    # Allocate output array for fluxes on the GPU.
    # Size N_total to match CPU version's expectation, even though the
    # kernel currently calculates N_total-1 fluxes. The last column remains uninitialized.
    d_F_CU = cuda.device_array((4, N_total), dtype=d_U_in.dtype)

    # Configure the kernel launch
    # Launch threads for N_total-1 interfaces (from j=0 to j=N_total-2)
    threadsperblock = TPB_FLUX # Use the constant defined above
    blockspergrid = ( (N_total - 1) + (threadsperblock - 1)) // threadsperblock

    # Launch the kernel
    central_upwind_flux_cuda_kernel[blockspergrid, threadsperblock](
        d_U_in, # Pass the input GPU array directly
        params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c,
        light_factor,  # Junction blocking factor
        d_F_CU
    )

    # Return the fluxes directly on the GPU device
    # The last column (interface N_total-1) is not calculated by the kernel.
    # The consuming function (solve_hyperbolic_step_gpu) needs to be aware of this
    # or handle the boundary flux appropriately if needed.
    return d_F_CU

