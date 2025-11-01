import numpy as np
from numba import njit, cuda # Import cuda
from .parameters import ModelParameters # Use absolute import


# --- Physical Constants and Conversions ---
KM_TO_M = 1000.0  # meters per kilometer
H_TO_S = 3600.0   # seconds per hour
M_TO_KM = 1.0 / KM_TO_M
S_TO_H = 1.0 / H_TO_S

# Derived conversion factors
KMH_TO_MS = KM_TO_M / H_TO_S  # km/h to m/s
MS_TO_KMH = H_TO_S / KM_TO_M  # m/s to km/h

# Vehicle density conversions
VEH_KM_TO_VEH_M = 1.0 / KM_TO_M # veh/km to veh/m
VEH_M_TO_VEH_KM = KM_TO_M       # veh/m to veh/km
# ----------------------------------------

@njit
def calculate_pressure(rho_m: np.ndarray, rho_c: np.ndarray,
                       alpha: float, rho_jam: float, epsilon: float,
                       K_m: float, gamma_m: float,
                       K_c: float, gamma_c: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the pressure terms for motorcycles (m) and cars (c).
    (Numba-optimized version)

    Args:
        rho_m: Density of motorcycles (veh/m).
        rho_c: Density of cars (veh/m).
        alpha: Interaction parameter.
        rho_jam: Jam density (veh/m).
        epsilon: Small number for numerical stability.
        K_m: Pressure coefficient for motorcycles (m/s).
        gamma_m: Pressure exponent for motorcycles.
        K_c: Pressure coefficient for cars (m/s).
        gamma_c: Pressure exponent for cars.


    Returns:
        A tuple (p_m, p_c) containing pressure terms (m/s).
    """
    # Numba doesn't support raising ValueError with strings easily in nopython mode
    # Validation should happen before calling this njit function
    # if rho_jam <= 0:
    #     raise ValueError("Jam density rho_jam must be positive.")

    # Ensure densities are non-negative
    rho_m = np.maximum(rho_m, 0.0)
    rho_c = np.maximum(rho_c, 0.0)

    rho_eff_m = rho_m + alpha * rho_c
    rho_total = rho_m + rho_c

    # Avoid division by zero or issues near rho_jam
    # Calculate normalized densities.
    norm_rho_eff_m = rho_eff_m / rho_jam
    norm_rho_total = rho_total / rho_jam

    # Ensure base of power is non-negative
    norm_rho_eff_m = np.maximum(norm_rho_eff_m, 0.0)
    norm_rho_total = np.maximum(norm_rho_total, 0.0)

    p_m = K_m * (norm_rho_eff_m ** gamma_m)
    p_c = K_c * (norm_rho_total ** gamma_c)

    # Ensure pressure is zero if respective density is zero
    p_m = np.where(rho_m <= epsilon, 0.0, p_m)
    p_c = np.where(rho_c <= epsilon, 0.0, p_c)
    # Also ensure p_m is zero if rho_eff_m is zero (can happen if rho_m=0 and rho_c=0)
    p_m = np.where(rho_eff_m <= epsilon, 0.0, p_m)


    return p_m, p_c

# --- CUDA Kernel for Pressure Calculation ---
# This kernel calculates pressure for a single element (thread)
@cuda.jit(device=True) # Use device=True for functions called from other kernels
def _calculate_pressure_cuda(rho_m_i, rho_c_i, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c):
    """CUDA device function to calculate pressure for a single cell."""
    # Ensure densities are non-negative
    rho_m_i = max(rho_m_i, 0.0)
    rho_c_i = max(rho_c_i, 0.0)

    rho_eff_m_i = rho_m_i + alpha * rho_c_i
    rho_total_i = rho_m_i + rho_c_i

    # Calculate normalized densities.
    norm_rho_eff_m_i = rho_eff_m_i / rho_jam
    norm_rho_total_i = rho_total_i / rho_jam

    # Ensure base of power is non-negative
    norm_rho_eff_m_i = max(norm_rho_eff_m_i, 0.0)
    norm_rho_total_i = max(norm_rho_total_i, 0.0)

    p_m_i = K_m * (norm_rho_eff_m_i ** gamma_m)
    p_c_i = K_c * (norm_rho_total_i ** gamma_c)

    # Ensure pressure is zero if respective density is zero
    if rho_m_i <= epsilon:
        p_m_i = 0.0
    if rho_c_i <= epsilon:
        p_c_i = 0.0
    # Also ensure p_m is zero if rho_eff_m is zero
    if rho_eff_m_i <= epsilon:
        p_m_i = 0.0

    return p_m_i, p_c_i

# --- Removed calculate_pressure_cuda_kernel and calculate_pressure_gpu ---
# These were wrappers performing CPU->GPU->CPU transfers and are superseded
# by direct calls to the _calculate_pressure_cuda device function within
# other kernels.


def calculate_equilibrium_speed(rho_m: np.ndarray, rho_c: np.ndarray, R_local: np.ndarray, params: ModelParameters, 
                              V0_m_override: float = None, V0_c_override: float = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the equilibrium speeds for motorcycles (m) and cars (c).
    
    ARCHITECTURAL NOTE (2025-10-24):
        This function now supports segment-specific speed overrides via V0_m_override and V0_c_override.
        When provided, these values REPLACE the Vmax[R] lookup, enabling heterogeneous networks
        where different segments have different speed limits (e.g., Lagos arterial = 32 km/h,
        highway = 80 km/h) regardless of road quality R.
        
        Use case: NetworkGrid with ParameterManager for per-segment parameters.

    Args:
        rho_m: Density of motorcycles (veh/m).
        rho_c: Density of cars (veh/m).
        R_local: Array of local road quality indices for each cell.
        params: ModelParameters object.
        V0_m_override: Optional override for motorcycle free-flow speed (m/s).
                       If provided, replaces Vmax_m[R_local] calculation.
        V0_c_override: Optional override for car free-flow speed (m/s).
                       If provided, replaces Vmax_c[R_local] calculation.

    Returns:
        A tuple (Ve_m, Ve_c) containing equilibrium speeds (m/s).
    """
    if params.rho_jam <= 0:
        raise ValueError("Jam density rho_jam must be positive.")

    # Ensure densities are non-negative
    rho_m = np.maximum(rho_m, 0.0)
    rho_c = np.maximum(rho_c, 0.0)

    rho_total = rho_m + rho_c

    # Calculate reduction factor g, ensuring it's between 0 and 1
    g = np.maximum(0.0, 1.0 - rho_total / params.rho_jam)

    # Determine Vmax values: Use overrides if provided, else fall back to Vmax[R]
    if V0_m_override is not None:
        # Use segment-specific override (scalar, broadcast to array if needed)
        Vmax_m_local = V0_m_override if np.isscalar(rho_m) else np.full_like(rho_m, V0_m_override)
    else:
        # Get Vmax based on local road quality R_local
        # Use np.vectorize or direct array indexing if R_local is an array of indices
        try:
            Vmax_m_local = np.array([params.Vmax_m[int(r)] for r in R_local])
        except KeyError as e:
            raise ValueError(f"Invalid road category index found in R_local: {e}. Valid keys: {list(params.Vmax_m.keys())}") from e
        except TypeError: # Handle scalar R_local
             Vmax_m_local = params.Vmax_m[int(R_local)]
    
    if V0_c_override is not None:
        # Use segment-specific override (scalar, broadcast to array if needed)
        Vmax_c_local = V0_c_override if np.isscalar(rho_c) else np.full_like(rho_c, V0_c_override)
    else:
        # Get Vmax based on local road quality R_local
        try:
            Vmax_c_local = np.array([params.Vmax_c[int(r)] for r in R_local])
        except KeyError as e:
            raise ValueError(f"Invalid road category index found in R_local: {e}. Valid keys: {list(params.Vmax_c.keys())}") from e
        except TypeError: # Handle scalar R_local
             Vmax_c_local = params.Vmax_c[int(R_local)]


    Ve_m = params.V_creeping + (Vmax_m_local - params.V_creeping) * g
    Ve_c = Vmax_c_local * g

    # Ensure speeds are non-negative
    Ve_m = np.maximum(Ve_m, 0.0)
    Ve_c = np.maximum(Ve_c, 0.0)

    return Ve_m, Ve_c
# --- CUDA Device Function for Equilibrium Speed ---
@cuda.jit(device=True)
def calculate_equilibrium_speed_gpu(rho_m_i: float, rho_c_i: float, R_local_i: int,
                                    # Pass relevant scalar parameters explicitly
                                    rho_jam: float, V_creeping: float,
                                    # Vmax values for different road categories
                                    # Assuming max 3 categories for simplicity in if/elif
                                    v_max_m_cat1: float, v_max_m_cat2: float, v_max_m_cat3: float,
                                    v_max_c_cat1: float, v_max_c_cat2: float, v_max_c_cat3: float
                                    ) -> tuple[float, float]:
    """
    Calculates the equilibrium speeds for a single cell on the GPU.
    Uses if/elif for Vmax lookup based on R_local_i.
    """
    if rho_jam <= 0:
        # Cannot raise errors in device code easily, return 0 or handle upstream
        return 0.0, 0.0

    # Ensure densities are non-negative
    rho_m_calc = max(rho_m_i, 0.0)
    rho_c_calc = max(rho_c_i, 0.0)

    rho_total = rho_m_calc + rho_c_calc

    # Calculate reduction factor g, ensuring it's between 0 and 1
    g = max(0.0, 1.0 - rho_total / rho_jam)

    # Get Vmax based on local road quality R_local_i using if/elif
    # --- This section MUST be adapted based on your actual road categories ---
    Vmax_m_local_i = 0.0
    Vmax_c_local_i = 0.0
    if R_local_i == 1:
        Vmax_m_local_i = v_max_m_cat1
        Vmax_c_local_i = v_max_c_cat1
    elif R_local_i == 2:
         Vmax_m_local_i = v_max_m_cat2 # Assuming category 2 exists
         Vmax_c_local_i = v_max_c_cat2
    elif R_local_i == 3:
        Vmax_m_local_i = v_max_m_cat3
        Vmax_c_local_i = v_max_c_cat3
    # Add more elif conditions if you have more categories
    # else:
        # Handle unknown category? Default to a known one or lowest speed?
        # Vmax_m_local_i = v_max_m_cat3 # Example: Default to category 3
        # Vmax_c_local_i = v_max_c_cat3

    # Calculate equilibrium speeds
    Ve_m_i = V_creeping + (Vmax_m_local_i - V_creeping) * g
    Ve_c_i = Vmax_c_local_i * g

    # Ensure speeds are non-negative
    Ve_m_i = max(Ve_m_i, 0.0)
    Ve_c_i = max(Ve_c_i, 0.0)

    return Ve_m_i, Ve_c_i

# --- CUDA Device Function for Relaxation Time ---
@cuda.jit(device=True)
def calculate_relaxation_time_gpu(rho_m_i: float, rho_c_i: float,
                                  # Pass relevant scalar parameters explicitly
                                  tau_m: float, tau_c: float
                                  ) -> tuple[float, float]:
    """
    Calculates the relaxation times for a single cell on the GPU.
    Currently returns constant values based on params.
    """
    # rho_m_i and rho_c_i are unused for now, but kept for signature consistency
    # Future: Could implement density-dependent relaxation times here
    return tau_m, tau_c
def calculate_relaxation_time(rho_m: np.ndarray, rho_c: np.ndarray, params: ModelParameters) -> tuple[float, float]:
    """
    Calculates the relaxation times for motorcycles (m) and cars (c).
    Currently returns constant values based on params.

    Args:
        rho_m: Density of motorcycles (veh/m). (Currently unused)
        rho_c: Density of cars (veh/m). (Currently unused)
        params: ModelParameters object.

    Returns:
        A tuple (tau_m, tau_c) containing relaxation times (s).
    """
    # Future: Could implement density-dependent relaxation times here
    return params.tau_m, params.tau_c

@njit
def calculate_physical_velocity(w_m: np.ndarray, w_c: np.ndarray, p_m: np.ndarray, p_c: np.ndarray, v_max: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the physical velocities from Lagrangian variables and pressure.
    
    Applies bounds to prevent numerical instability from extreme velocity values
    that can arise when density approaches zero (velocity explosion).

    Args:
        w_m: Lagrangian variable for motorcycles (m/s).
        w_c: Lagrangian variable for cars (m/s).
        p_m: Pressure term for motorcycles (m/s).
        p_c: Pressure term for cars (m/s).
        v_max: Maximum realistic velocity magnitude (m/s). Default 50 m/s = 180 km/h.

    Returns:
        A tuple (v_m, v_c) containing physical velocities (m/s), bounded to [-v_max, v_max].
    """
    v_m = w_m - p_m
    v_c = w_c - p_c
    # Apply physical bounds to prevent numerical explosion
    # Numba-compatible element-wise clipping using maximum/minimum
    v_m = np.maximum(np.minimum(v_m, v_max), -v_max)
    v_c = np.maximum(np.minimum(v_c, v_max), -v_max)
    return v_m, v_c

# --- CUDA Kernel for Physical Velocity Calculation ---
# This kernel calculates physical velocity for a single element (thread)
@cuda.jit(device=True) # Use device=True for functions called from other kernels
def _calculate_physical_velocity_cuda(w_m_i, w_c_i, p_m_i, p_c_i):
    """CUDA device function to calculate physical velocity for a single cell."""
    v_m_i = w_m_i - p_m_i
    v_c_i = w_c_i - p_c_i
    return v_m_i, v_c_i

# --- Removed calculate_physical_velocity_cuda_kernel and calculate_physical_velocity_gpu ---
# These were wrappers performing CPU->GPU->CPU transfers and are superseded
# by direct calls to the _calculate_physical_velocity_cuda device function within
# other kernels.


@njit
def _calculate_pressure_derivative(rho_vals, K, gamma, rho_jam, epsilon):
    """ Helper to calculate dP/d(rho_eff) or dP/d(rho_total) for an array of densities. """
    if rho_jam <= 0 or gamma <= 0:
        return np.zeros_like(rho_vals)

    # Ensure non-negative before processing
    rho_vals_safe = np.maximum(rho_vals, 0.0)

    # Calculate normalized density
    norm_rho = rho_vals_safe / rho_jam

    # Derivative calculation
    derivative = (K * gamma / rho_jam) * (norm_rho**(gamma - 1.0))

    # Set derivative to zero where density is close to zero
    derivative = np.where(rho_vals_safe <= epsilon, 0.0, derivative)

    return np.maximum(derivative, 0.0) # Ensure non-negative derivative

def calculate_eigenvalues(rho_m: np.ndarray, v_m: np.ndarray, rho_c: np.ndarray, v_c: np.ndarray, params: ModelParameters) -> list[np.ndarray]:
    """
    Calculates the four eigenvalues (characteristic speeds) of the system.

    Args:
        rho_m: Density of motorcycles (veh/m).
        v_m: Velocity of motorcycles (m/s).
        rho_c: Density of cars (veh/m).
        v_c: Velocity of cars (m/s).
        params: ModelParameters object.

    Returns:
        A list containing four numpy arrays: [lambda1, lambda2, lambda3, lambda4] (m/s).
        Each array has the same shape as the input density/velocity arrays.
    """
    if params.rho_jam <= 0:
        raise ValueError("Jam density rho_jam must be positive.")

    # Ensure densities are non-negative
    rho_m = np.maximum(rho_m, params.epsilon) # Use epsilon to avoid issues in derivative calc
    rho_c = np.maximum(rho_c, params.epsilon)

    rho_eff_m = rho_m + params.alpha * rho_c
    rho_total = rho_m + rho_c

    # Calculate pressure derivatives dP/d(arg) where arg is rho_eff_m or rho_total
    # Need to handle scalar vs array inputs if vectorizing
    P_prime_m = _calculate_pressure_derivative(rho_eff_m, params.K_m, params.gamma_m, params.rho_jam, params.epsilon)
    P_prime_c = _calculate_pressure_derivative(rho_total, params.K_c, params.gamma_c, params.rho_jam, params.epsilon)


    lambda1 = v_m
    lambda2 = v_m - rho_m * P_prime_m
    lambda3 = v_c
    lambda4 = v_c - rho_c * P_prime_c

    return [lambda1, lambda2, lambda3, lambda4]

# --- CUDA Device Functions for Eigenvalue Calculation ---
import math # Needed for CUDA device functions

@cuda.jit(device=True)
def _calculate_pressure_derivative_cuda(rho_val, K, gamma, rho_jam, epsilon):
    """ CUDA device helper to calculate dP/d(rho_eff) or dP/d(rho_total). """
    if rho_jam <= 0 or gamma <= 0:
        return 0.0
    if rho_val <= epsilon:
        return 0.0 # Derivative is zero at zero density

    # Calculate normalized density (without capping)
    norm_rho = rho_val / rho_jam
    # Derivative of K * (x/rho_jam)^gamma = K * gamma * x^(gamma-1) / rho_jam^gamma
    # Use math.pow for CUDA device code
    derivative = K * gamma * (math.pow(norm_rho, gamma - 1.0)) / rho_jam
    return max(derivative, 0.0) # Ensure non-negative derivative

@cuda.jit(device=True)
def _calculate_eigenvalues_cuda(rho_m_i, v_m_i, rho_c_i, v_c_i,
                                alpha, rho_jam, epsilon,
                                K_m, gamma_m, K_c, gamma_c):
    """
    CUDA device function to calculate the four eigenvalues for a single cell.
    """
    # Ensure densities are non-negative (use epsilon for stability in derivative)
    rho_m_calc = max(rho_m_i, epsilon)
    rho_c_calc = max(rho_c_i, epsilon)

    rho_eff_m_i = rho_m_calc + alpha * rho_c_calc
    rho_total_i = rho_m_calc + rho_c_calc

    # Calculate pressure derivatives using the CUDA device function
    P_prime_m_i = _calculate_pressure_derivative_cuda(rho_eff_m_i, K_m, gamma_m, rho_jam, epsilon)
    P_prime_c_i = _calculate_pressure_derivative_cuda(rho_total_i, K_c, gamma_c, rho_jam, epsilon)

    lambda1 = v_m_i
    lambda2 = v_m_i - rho_m_calc * P_prime_m_i # Use rho_m_calc here
    lambda3 = v_c_i
    lambda4 = v_c_i - rho_c_calc * P_prime_c_i # Use rho_c_calc here

    return lambda1, lambda2, lambda3, lambda4


@njit
def calculate_source_term(U: np.ndarray,
                          # Pressure params
                          alpha: float, rho_jam: float, K_m: float, gamma_m: float, K_c: float, gamma_c: float,
                          # Equilibrium speeds (pre-calculated)
                          Ve_m: np.ndarray, Ve_c: np.ndarray,
                          # Relaxation times (pre-calculated)
                          tau_m: float, tau_c: float,
                          # Epsilon
                          epsilon: float) -> np.ndarray:
    """
    Calculates the source term vector S = (0, Sm, 0, Sc) for the ODE step.
    (Numba-optimized version)

    Args:
        U: State vector (or array of state vectors) [rho_m, w_m, rho_c, w_c].
           Shape (4,) or (4, N). Assumes SI units.
        alpha, rho_jam, K_m, gamma_m, K_c, gamma_c: Parameters for pressure calculation.
        Ve_m, Ve_c: Pre-calculated equilibrium speeds (m/s).
        tau_m, tau_c: Pre-calculated relaxation times (s).
        epsilon: Small number for numerical stability.


    Returns:
        Source term vector S (or array of source vectors). Shape (4,) or (4, N).
    """
    rho_m = U[0]
    w_m = U[1]
    rho_c = U[2]
    w_c = U[3]

    # Ensure densities are non-negative for calculations
    rho_m_calc = np.maximum(rho_m, 0.0)
    rho_c_calc = np.maximum(rho_c, 0.0)

    # Calculate pressure using the Numba-fied function
    p_m, p_c = calculate_pressure(rho_m_calc, rho_c_calc,
                                  alpha, rho_jam, epsilon,
                                  K_m, gamma_m, K_c, gamma_c)

    # Calculate physical velocity (this function is simple NumPy, likely okay for Numba)
    v_m, v_c = calculate_physical_velocity(w_m, w_c, p_m, p_c)

    # Equilibrium speeds (Ve_m, Ve_c) and relaxation times (tau_m, tau_c) are now inputs

    # Avoid division by zero if relaxation times are zero
    Sm = (Ve_m - v_m) / (tau_m + epsilon)
    Sc = (Ve_c - v_c) / (tau_c + epsilon)

    # Source term is zero if density is zero
    Sm = np.where(rho_m_calc <= epsilon, 0.0, Sm)
    Sc = np.where(rho_c_calc <= epsilon, 0.0, Sc)

    # Construct the full source vector S = [0, Sm, 0, Sc]
    # Need to handle scalar vs array inputs for U
    if U.ndim == 1: # Scalar input (single cell)
        # Initialize S and assign values (more Numba-friendly)
        S = np.zeros(4, dtype=U.dtype)
        S[1] = Sm # Sm and Sc should be scalars here if U.ndim == 1
        S[3] = Sc
    else: # Array input (multiple cells)
        S = np.zeros_like(U)
        S[1, :] = Sm
        S[3, :] = Sc
    return S

# --- CUDA Device Function for Source Term Calculation ---
@cuda.jit(device=True)
def calculate_source_term_gpu(y, # Local state vector [rho_m, w_m, rho_c, w_c]
                              # Pressure params
                              alpha: float, rho_jam: float, K_m: float, gamma_m: float, K_c: float, gamma_c: float,
                              # Equilibrium speeds (pre-calculated for this cell)
                              Ve_m_i: float, Ve_c_i: float,
                              # Relaxation times (pre-calculated for this cell)
                              tau_m_i: float, tau_c_i: float,
                              # Epsilon
                              epsilon: float) -> tuple[float, float, float, float]:
    """
    Calculates the source term vector S = (0, Sm, 0, Sc) for a single cell on the GPU.
    Calls other CUDA device functions for pressure and velocity.
    """
    rho_m_i = y[0]
    w_m_i = y[1]
    rho_c_i = y[2]
    w_c_i = y[3]

    # Ensure densities are non-negative for calculations
    rho_m_calc = max(rho_m_i, 0.0)
    rho_c_calc = max(rho_c_i, 0.0)

    # Calculate pressure using the CUDA device function
    p_m_i, p_c_i = _calculate_pressure_cuda(rho_m_calc, rho_c_calc,
                                            alpha, rho_jam, epsilon,
                                            K_m, gamma_m, K_c, gamma_c)

    # Calculate physical velocity using the CUDA device function
    v_m_i, v_c_i = _calculate_physical_velocity_cuda(w_m_i, w_c_i, p_m_i, p_c_i)

    # Equilibrium speeds (Ve_m_i, Ve_c_i) and relaxation times (tau_m_i, tau_c_i) are inputs

    # Avoid division by zero if relaxation times are zero
    Sm_i = 0.0
    if tau_m_i > epsilon and rho_m_calc > epsilon: # Only calculate if density > 0 and tau > 0
        Sm_i = (Ve_m_i - v_m_i) / tau_m_i

    Sc_i = 0.0
    if tau_c_i > epsilon and rho_c_calc > epsilon: # Only calculate if density > 0 and tau > 0
        Sc_i = (Ve_c_i - v_c_i) / tau_c_i

    # Source term vector S = (0, Sm, 0, Sc)
    return 0.0, Sm_i, 0.0, Sc_i
# Removed dead code block from CPU version after the correct return statement

# Removed redundant CUDA source term functions (_calculate_source_term_cuda,
# calculate_source_term_cuda_kernel, and the wrapper calculate_source_term_gpu)
# as they are not used by the current _ode_step_kernel approach.