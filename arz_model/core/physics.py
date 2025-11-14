import numpy as np
from numba import njit, cuda # Import cuda
import math # Needed for CUDA device functions

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
                       K_c: float, gamma_c: float) -> tuple:
    """
    Calculates the pressure terms for motorcycles (m) and cars (c).
    (Numba-optimized CPU version for initial condition setup)

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
    # Ensure densities are non-negative
    rho_m = np.maximum(rho_m, 0.0)
    rho_c = np.maximum(rho_c, 0.0)

    rho_eff_m = rho_m + alpha * rho_c
    rho_total = rho_m + rho_c

    # Calculate normalized densities
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
    # Also ensure p_m is zero if rho_eff_m is zero
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


# --- CUDA Device Functions for Eigenvalue Calculation ---

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

# --- CUDA Device Function for Physical Flux Calculation ---
@cuda.jit(device=True)
def _calculate_physical_flux_cuda(rho_m_i, w_m_i, rho_c_i, w_c_i, p_m_i, p_c_i):
    """CUDA device function to calculate the physical flux F(U) for a single cell."""
    v_m_i, v_c_i = _calculate_physical_velocity_cuda(w_m_i, w_c_i, p_m_i, p_c_i)
    
    # F(U) = [rho_m * v_m, w_m, rho_c * v_c, w_c]
    # Note: The flux for the w components is just w itself, which is an
    # approximation for the non-conservative part of the system.
    flux_rho_m = rho_m_i * v_m_i
    flux_w_m = w_m_i
    flux_rho_c = rho_c_i * v_c_i
    flux_w_c = w_c_i
    
    return flux_rho_m, flux_w_m, flux_rho_c, flux_w_c

@cuda.jit(device=True)
def _calculate_demand_flux_cuda(rho_m_i, w_m_i, rho_c_i, w_c_i,
                                alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c):
    """
    CUDA device function to calculate the demand flux for a single cell state.
    Demand is the physical flux F(U) for a given state U.
    """
    rho_m_calc = max(rho_m_i, 0.0)
    rho_c_calc = max(rho_c_i, 0.0)
    
    p_m_i, p_c_i = _calculate_pressure_cuda(rho_m_calc, rho_c_calc,
                                            alpha, rho_jam, epsilon,
                                            K_m, gamma_m, K_c, gamma_c)
                                            
    flux_rho_m, flux_w_m, flux_rho_c, flux_w_c = _calculate_physical_flux_cuda(
        rho_m_calc, w_m_i, rho_c_calc, w_c_i, p_m_i, p_c_i
    )
    
    return flux_rho_m, flux_rho_c

@cuda.jit(device=True)
def _calculate_supply_flux_cuda(rho_jam, K_m, gamma_m, K_c, gamma_c):
    """
    CUDA device function to calculate the supply (capacity) of a link.
    This is the maximum possible physical flux, which occurs at the critical density.
    """
    # This is a simplification. A true supply function depends on the downstream state.
    # Here, we approximate it with the maximum possible flux (capacity).
    # The critical density rho_crit where flux is maximum is found by solving d(F)/d(rho) = 0.
    # For the ARZ model, this is complex. We use an approximation.
    # For a single class model rho*v(rho), where v(rho) = Vmax(1-rho/rho_jam), the max
    # flux is at rho_crit = rho_jam/2.
    # Let's assume a similar behavior and calculate flux at a fraction of rho_jam.
    
    rho_crit_m = rho_jam / 2.0  # Approximation
    rho_crit_c = rho_jam / 2.0  # Approximation

    # We need to find the state U that corresponds to this.
    # Assume w = v, so p=0. This is another simplification.
    v_crit_m = 0.0 # Placeholder
    v_crit_c = 0.0 # Placeholder

    # A simpler, more robust approach is to define capacity directly.
    # For now, returning a high, constant value is a placeholder.
    # Let's use a value based on typical highway capacity (e.g., 2000 veh/hr/lane)
    # 2000 veh/hr -> 0.55 veh/s.
    # This is a placeholder until a better supply function is derived.
    supply_m = 0.55 
    supply_c = 0.55
    
    return supply_m, supply_c

@cuda.jit(device=True)
def _invert_flux_function_cuda(flux, rho_jam, Vmax, epsilon):
    """
    Inverts the flux function F(rho) = rho * V_e(rho) to find rho for a given flux.
    This assumes a simplified equilibrium velocity V_e(rho) = Vmax * (1 - rho/rho_jam).
    The flux function is F(rho) = Vmax * rho * (1 - rho/rho_jam), which is a quadratic.
    
    flux = -Vmax/rho_jam * rho^2 + Vmax * rho
    => Vmax/rho_jam * rho^2 - Vmax * rho + flux = 0
    
    This is a quadratic equation of the form a*x^2 + b*x + c = 0, where:
    x = rho
    a = Vmax / rho_jam
    b = -Vmax
    c = flux
    
    The solutions are rho = (-b ± sqrt(b^2 - 4ac)) / 2a.
    We choose the solution that is less than the critical density rho_jam/2,
    as this corresponds to the free-flow branch of the fundamental diagram.
    """
    a = Vmax / rho_jam
    b = -Vmax
    c = flux
    
    discriminant = b*b - 4*a*c
    
    if discriminant < 0:
        # No real solution, implies flux is greater than capacity.
        # Return critical density, where capacity is reached.
        return rho_jam / 2.0
        
    sqrt_discriminant = math.sqrt(discriminant)
    
    # Two possible solutions for rho
    rho1 = (-b + sqrt_discriminant) / (2 * a)
    rho2 = (-b - sqrt_discriminant) / (2 * a)
    
    # The physically correct state for the ghost cell of an outgoing link
    # corresponds to the free-flow condition, which is the lower density.
    return min(rho1, rho2)