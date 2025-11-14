import numpy as np
from numba import cuda # Import cuda
from scipy.integrate import solve_ivp
from typing import Optional
from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters
from ..core import physics
import math # Import math for ceil
import warnings
from . import riemann_solvers # Import the riemann solver module
from . import boundary_conditions
from .reconstruction.weno import reconstruct_weno5
from .reconstruction.converter import conserved_to_primitives_arr, primitives_to_conserved_arr
from ..config.debug_config import DEBUG_LOGS_ENABLED
from .logging_utils import should_log

# Import GPU implementations
try:
    from .gpu.weno_cuda import reconstruct_weno5_gpu_naive, reconstruct_weno5_gpu_optimized
    from .gpu.ssp_rk3_cuda import SSP_RK3_GPU, integrate_ssp_rk3_gpu
    GPU_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GPU implementations not available: {e}")
    GPU_AVAILABLE = False


# --- Physical State Bounds Enforcement ---

@cuda.jit
def _apply_bounds_kernel(U, N_physical, num_ghost, rho_max, v_max, epsilon,
                         alpha, rho_jam, K_m, gamma_m, K_c, gamma_c):
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
            pressure_factor = (rho_total / rho_jam) ** gamma_m if rho_total > epsilon else 0.0
            p_m = K_m * pressure_factor
            
            v_m = w_m - p_m
            v_m = max(-v_max, min(v_m, v_max))
            w_m = v_m + p_m
        else:
            w_m = 0.0
        
        # 3. Same for cars
        if rho_c > epsilon:
            rho_total = rho_m + rho_c
            pressure_factor = (rho_total / rho_jam) ** gamma_c if rho_total > epsilon else 0.0
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


def apply_physical_state_bounds_gpu(d_U: cuda.devicearray.DeviceNDArray, grid: Grid1D, 
                                    params: ModelParameters, rho_max: float = 1.0, 
                                    v_max: float = 50.0) -> cuda.devicearray.DeviceNDArray:
    """
    GPU version of apply_physical_state_bounds.
    
    Enforces physical bounds on GPU without CPU transfer.
    
    Args:
        d_U: Numba device array (4, N_total) on GPU
        grid: Grid object
        params: Model parameters
        rho_max: Maximum density (veh/m)
        v_max: Maximum velocity (m/s)
        
    Returns:
        Numba device array with bounds applied (same object, modified in-place)
    """
    threadsperblock = 256
    blockspergrid = math.ceil(grid.N_physical / threadsperblock)
    
    _apply_bounds_kernel[blockspergrid, threadsperblock](
        d_U, grid.N_physical, grid.num_ghost_cells, rho_max, v_max, params.physics.epsilon,
        params.physics.alpha, params.physics.rho_jam, params.physics.k_m, params.physics.gamma_m, params.physics.k_c, params.physics.gamma_c
    )
    
    return d_U


def apply_physical_state_bounds(U: np.ndarray, grid: Grid1D, params: ModelParameters, rho_max: float = 1.0, v_max: float = 50.0) -> np.ndarray:
    """
    Enforces physical bounds on the state vector to prevent numerical explosion.
    
    This is a safety net applied after time integration to catch any extreme values
    that might arise from numerical instabilities (WENO oscillations, CFL violations, etc.).
    
    Bounds applied:
    - Density: 0 ≤ rho ≤ rho_max (vehicles/m)
    - Velocity magnitude: |v| ≤ v_max (m/s)
    - Momentum w is adjusted to maintain bounded velocity: w = rho*v + p
    
    Args:
        U: State array (4, N_total)
        grid: Grid object
        params: Model parameters
        rho_max: Maximum density (veh/m). Default 1.0 = jam density
        v_max: Maximum velocity magnitude (m/s). Default 50.0 m/s = 180 km/h
        
    Returns:
        Bounded state array
    """
    U_bounded = U.copy()
    g = grid.num_ghost_cells
    N = grid.N_physical
    
    # Process physical cells only (ghost cells handled by BC)
    for j in range(g, g + N):
        rho_m = U_bounded[0, j]
        w_m = U_bounded[1, j]
        rho_c = U_bounded[2, j]
        w_c = U_bounded[3, j]
        
        # 1. Clamp densities to [0, rho_max]
        rho_m = max(0.0, min(rho_m, rho_max))
        rho_c = max(0.0, min(rho_c, rho_max))
        
        # 2. Calculate pressures and velocities
        if rho_m > params.physics.epsilon:
            p_m, _ = physics.calculate_pressure(
                np.array([rho_m]), np.array([rho_c]),
                params.physics.alpha, params.physics.rho_jam, params.physics.epsilon,
                params.physics.k_m, params.physics.gamma_m, params.physics.k_c, params.physics.gamma_c
            )
            p_m = p_m[0]
            v_m = w_m - p_m
            
            # 3. Clamp velocity and reconstruct momentum
            v_m = max(-v_max, min(v_m, v_max))
            w_m = v_m + p_m
        else:
            # Near-vacuum state: set velocity to zero
            w_m = 0.0
        
        # Same for cars
        if rho_c > params.physics.epsilon:
            _, p_c = physics.calculate_pressure(
                np.array([rho_m]), np.array([rho_c]),
                params.physics.alpha, params.physics.rho_jam, params.physics.epsilon,
                params.physics.k_m, params.physics.gamma_m, params.physics.k_c, params.physics.gamma_c
            )
            p_c = p_c[0]
            v_c = w_c - p_c
            v_c = max(-v_max, min(v_c, v_max))
            w_c = v_c + p_c
        else:
            w_c = 0.0
        
        # Write back bounded values
        U_bounded[0, j] = rho_m
        U_bounded[1, j] = w_m
        U_bounded[2, j] = rho_c
        U_bounded[3, j] = w_c
    
    return U_bounded


# --- CFL Condition Check ---

def check_cfl_condition(U: np.ndarray, grid: Grid1D, params: ModelParameters, dt: float, CFL_max: float = 0.9) -> tuple[bool, float]:
    """
    Checks if the CFL (Courant-Friedrichs-Lewy) condition is satisfied for stability.
    
    CFL condition: dt ≤ CFL_max * dx / λ_max
    where λ_max is the maximum wave speed (eigenvalue magnitude).
    
    Args:
        U: State array (4, N_total)
        grid: Grid object
        params: Model parameters
        dt: Current timestep
        CFL_max: Maximum allowed CFL number (default 0.9 for SSP-RK3)
        
    Returns:
        (is_stable, CFL_number) - Boolean indicating stability and actual CFL number
    """
    g = grid.num_ghost_cells
    N = grid.N_physical
    
    # Extract physical cells only
    U_phys = U[:, g:g+N]
    rho_m = U_phys[0, :]
    w_m = U_phys[1, :]
    rho_c = U_phys[2, :]
    w_c = U_phys[3, :]
    
    # Calculate pressures and velocities
    p_m, p_c = physics.calculate_pressure(
        rho_m, rho_c, params.physics.alpha, params.physics.rho_jam, params.physics.epsilon,
        params.physics.k_m, params.physics.gamma_m, params.physics.k_c, params.physics.gamma_c
    )
    v_m, v_c = physics.calculate_physical_velocity(w_m, w_c, p_m, p_c)
    
    # Calculate all eigenvalues
    eigenvalues = physics.calculate_eigenvalues(
        rho_m, v_m, rho_c, v_c,
        params.physics.alpha, params.physics.rho_jam, params.physics.epsilon,
        params.physics.k_m, params.physics.gamma_m, params.physics.k_c, params.physics.gamma_c
    )
    
    # Find maximum absolute eigenvalue (wave speed)
    lambda_max = 0.0
    for lambda_arr in eigenvalues:
        lambda_max = max(lambda_max, np.max(np.abs(lambda_arr)))
    
    # CFL number: dt * λ_max / dx
    CFL = dt * lambda_max / grid.dx if lambda_max > 0 else 0.0
    
    is_stable = CFL <= CFL_max
    
    return is_stable, CFL

# --- WENO-Based Spatial Discretization ---

def calculate_spatial_discretization_weno(U: np.ndarray, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None, apply_bc: bool = True) -> np.ndarray:
    """
    Calcule la discrétisation spatiale L(U) = -dF/dx en utilisant la reconstruction WENO5.
    
    Cette fonction orchestre :
    1. La conversion des variables conservées vers les variables primitives
    2. La reconstruction WENO5 des variables primitives aux interfaces
    3. Le calcul des flux via le solveur de Riemann Central-Upwind
    4. Le calcul de la dérivée spatiale du flux
    
    Junction-aware: If grid.junction_at_right is set, the flux at the rightmost 
    cell interface (exit boundary) is computed with junction blocking metadata,
    enabling traffic signal control in multi-segment networks.
    
    Args:
        U (np.ndarray): État conservé (4, N_total) incluant les cellules fantômes
        grid (Grid1D): Objet grille (may have junction_at_right attribute set)
        params (ModelParameters): Paramètres du modèle
        current_bc_params (dict | None): Mise à jour des paramètres BC (pour inflow dynamique). Defaults to None.
        
    Returns:
        np.ndarray: Discrétisation spatiale L(U) = -dF/dx (4, N_total)
    """
    # DEBUG: Track call count and confirm we reach BC call
    if not hasattr(calculate_spatial_discretization_weno, '_call_count'):
        calculate_spatial_discretization_weno._call_count = 0
    calculate_spatial_discretization_weno._call_count += 1
    call_count = calculate_spatial_discretization_weno._call_count
    
    if DEBUG_LOGS_ENABLED and call_count <= 5:
        print(f"[WENO #{call_count}] About to call apply_BC: current_bc_params={current_bc_params is not None}, params.BC exists={hasattr(params, 'boundary_conditions')}")
    
    # 0. Application des conditions aux limites sur l'état d'entrée
    U_bc = np.copy(U)
    if apply_bc:
        boundary_conditions.apply_boundary_conditions(U_bc, grid, params, current_bc_params)
    
    # DEBUG: Check ghost cell values after BC application
    if DEBUG_LOGS_ENABLED and call_count <= 3:
        g = grid.num_ghost_cells
        print(f"[WENO INPUT #{call_count}] Ghost cells U_bc[:, 0:{g}]:")
        print(f"  {U_bc[:, :g]}")
    
    # 1. Conversion vers les variables primitives
    P = conserved_to_primitives_arr(
        U_bc, params.physics.alpha, params.physics.rho_jam, params.physics.epsilon,
        params.physics.k_m, params.physics.gamma_m, params.physics.k_c, params.physics.gamma_c
    )
    
    # DEBUG: Check primitive variables after conversion
    if DEBUG_LOGS_ENABLED and call_count <= 3:
        g = grid.num_ghost_cells
        print(f"[PRIMITIVES #{call_count}] Ghost cells P[:, 0:{g}]:")
        print(f"  {P[:, :g]}")
        print(f"  Physical cell P[:, {g}] = {P[:, g]}")
    
    # 2. Reconstruction WENO5 pour chaque variable primitive
    N_total = U.shape[1]
    P_left = np.zeros_like(P)   # Variables primitives reconstruites à gauche des interfaces
    P_right = np.zeros_like(P)  # Variables primitives reconstruites à droite des interfaces
    
    for var_idx in range(4):  # Pour chaque variable (rho_m, v_m, rho_c, v_c)
        P_left[var_idx, :], P_right[var_idx, :] = reconstruct_weno5(P[var_idx, :])
    
    # DEBUG: Check WENO reconstruction at boundary
    if DEBUG_LOGS_ENABLED and call_count <= 3:
        print(f"[WENO RECON #{call_count}] At interface {g-0.5} (between ghost j={g-1} and ghost j={g}):")
        print(f"  P_left[0, {g}] (rho_m, from ghost {g})={P_left[0, g]:.6f}")
        print(f"  P_right[0, {g-1}] (rho_m, from ghost {g-1})={P_right[0, g-1]:.6f}")
        print(f"[WENO RECON #{call_count}] At interface {g+0.5} (between ghost j={g} and physical j={g+1}):")
        print(f"  P_left[0, {g+1}] (rho_m, from physical)={P_left[0, g+1]:.6f}")
        print(f"  P_right[0, {g}] (rho_m, from ghost)={P_right[0, g]:.6f}")
    
    # 3. Conversion des reconstructions primitives vers conservées pour le calcul de flux
    # Note: Dans notre sémantique WENO, P_left[i+1] et P_right[i] correspondent à l'interface i+1/2
    fluxes = np.zeros((4, N_total))
    g = grid.num_ghost_cells
    N = grid.N_physical
    
    for j in range(g - 1, g + N):  # Calculer les flux F_{j+1/2} pour j=g-1..g+N-1
        # Pour l'interface j+1/2, utiliser P_left[j+1] et P_right[j]
        if j + 1 < N_total:
            # Reconstruction gauche à l'interface j+1/2
            P_L = P_left[:, j + 1]
            # Reconstruction droite à l'interface j+1/2  
            P_R = P_right[:, j]
            
            # Apply positivity limiter with momentum consistency
            # WENO reconstruction can produce negative densities at sharp gradients
            # Critical: When clamping density, must also adjust momentum to prevent velocity explosion
            v_max_physical = 50.0  # Maximum realistic traffic velocity (m/s)
            
            # Left state consistency
            if P_L[0] < params.physics.epsilon:
                # Density too small - clamp and scale momentum
                rho_old = P_L[0]
                P_L[0] = params.physics.epsilon
                # Cap velocity to prevent explosion: v = w - p
                # Approximate: w_safe = rho * v_max + p ≈ rho * v_max (p << rho*v for small rho)
                w_m_max = P_L[0] * v_max_physical
                w_c_max = P_L[0] * v_max_physical
                P_L[1] = np.clip(P_L[1], -w_m_max, w_m_max)  # v_m bounded
            if P_L[2] < params.physics.epsilon:
                P_L[2] = params.physics.epsilon
                w_c_max = P_L[2] * v_max_physical
                P_L[3] = np.clip(P_L[3], -w_c_max, w_c_max)  # v_c bounded
            
            # Right state consistency
            if P_R[0] < params.physics.epsilon:
                rho_old = P_R[0]
                P_R[0] = params.physics.epsilon
                w_m_max = P_R[0] * v_max_physical
                w_c_max = P_R[0] * v_max_physical
                P_R[1] = np.clip(P_R[1], -w_m_max, w_m_max)
            if P_R[2] < params.physics.epsilon:
                P_R[2] = params.physics.epsilon
                w_c_max = P_R[2] * v_max_physical
                P_R[3] = np.clip(P_R[3], -w_c_max, w_c_max)
            
            # Conversion vers variables conservées pour le flux
            U_L = primitives_to_conserved_single(P_L, params)
            U_R = primitives_to_conserved_single(P_R, params)
            
            # Check if this is the junction interface (rightmost physical cell)
            junction_info = None
            if j == g + N - 1 and hasattr(grid, 'junction_at_right') and grid.junction_at_right is not None:
                junction_info = grid.junction_at_right
            
            # DEBUG: Log Riemann states at left boundary
            if DEBUG_LOGS_ENABLED and j == g and call_count <= 3:
                print(f"[RIEMANN #{call_count}] j={j} (interface {j+0.5}):")
                print(f"  U_L={U_L}")
                print(f"  U_R={U_R}")
            
            # Calcul du flux Central-Upwind (junction-aware if at exit boundary)
            phys = params.physics
            fluxes[:, j] = riemann_solvers.central_upwind_flux(
                U_L, U_R, 
                phys.alpha, phys.rho_jam, phys.epsilon,
                phys.k_m, phys.gamma_m, phys.k_c, phys.gamma_c,
                junction_info
            )
            
            # DEBUG: Log computed flux
            if DEBUG_LOGS_ENABLED and j == g and call_count <= 3:
                print(f"  flux={fluxes[:, j]}")
    
    # 4. Calcul de la discrétisation spatiale L(U) = -dF/dx
    L_U = np.zeros_like(U)
    
    # DEBUG: Log first physical cell flux for boundary analysis
    if not hasattr(calculate_spatial_discretization_weno, '_flux_log_count'):
        calculate_spatial_discretization_weno._flux_log_count = 0
    calculate_spatial_discretization_weno._flux_log_count += 1
    
    for j in range(g, g + N):  # Cellules physiques seulement
        flux_right = fluxes[:, j]      # F_{j+1/2}
        flux_left = fluxes[:, j - 1]   # F_{j-1/2}
        L_U[:, j] = -(flux_right - flux_left) / grid.dx
        
        # Log first physical cell
        if DEBUG_LOGS_ENABLED and j == g and calculate_spatial_discretization_weno._flux_log_count <= 5:
            print(f"[FLUX #{calculate_spatial_discretization_weno._flux_log_count}] First physical cell j={j}:")
            print(f"  flux_left[0] (rho_m flux from ghost)={flux_left[0]:.6f}")
            print(f"  flux_right[0] (rho_m flux to next)={flux_right[0]:.6f}")
            print(f"  L_U[0,{j}] (rho_m rate)={L_U[0,j]:.6f}")
    
    return L_U


def calculate_spatial_discretization_godunov(
    U: np.ndarray, 
    grid: Grid1D, 
    params: ModelParameters,
    current_bc_params: Optional[dict] = None
) -> np.ndarray:
    """
    Godunov spatial discretization (first-order upwind).
    
    Differences vs WENO5:
    - No reconstruction (piecewise constant)
    - Direct cell-to-cell flux calculation
    - Robust with sharp discontinuities
    
    This is the spatial discretization component of the Godunov method for
    hyperbolic conservation laws. It computes L(U) = -dF/dx where F are
    the numerical fluxes at cell interfaces.
    
    Args:
        U (np.ndarray): State array (4, N_total) = [rho_m, w_m, rho_c, w_c]
        grid (Grid1D): Spatial grid object
        params (ModelParameters): Model parameters
        current_bc_params (Optional[dict]): Dynamic boundary conditions (for time-varying inflow)
    
    Returns:
        np.ndarray: Spatial discretization L(U) = -dF/dx. Shape (4, N_total).
        
    References:
        - Godunov (1959): A difference method for numerical calculation of discontinuous solutions
        - LeVeque (2002): Finite Volume Methods for Hyperbolic Problems
        - Mammar et al. (2009): Riemann solver for ARZ traffic model
        
    Example:
        >>> L_U = calculate_spatial_discretization_godunov(U, grid, params)
        >>> # Use in time integration (Euler or SSP-RK3)
        >>> U_new = U + dt * L_U
    """
    # 1. Apply boundary conditions
    U_bc_result = boundary_conditions.apply_boundary_conditions(
        U, grid, params, current_bc_params
    )
    
    # Handle case where BC returns None (NetworkGrid interior segments)
    # In this case, ghost cells are already set by junction coupling
    U_bc = U if U_bc_result is None else U_bc_result
    
    # 2. Calculate fluxes at all interfaces
    g = grid.num_ghost_cells
    N = grid.N_physical
    fluxes = np.zeros((4, N + 2*g))
    
    # Loop over interfaces (j-1/2) from ghost to physical
    for j in range(g-1, g+N):
        # Left and right states (NO reconstruction - piecewise constant!)
        U_L = U_bc[:, j]
        U_R = U_bc[:, j+1]
        
        # Check if at junction interface (same logic as WENO5)
        junction_info = None
        if j == g + N - 1 and hasattr(grid, 'junction_at_right'):
            if grid.junction_at_right is not None:
                junction_info = grid.junction_at_right
        
        # Godunov flux (upwind selection with Central-Upwind fallback)
        fluxes[:, j] = riemann_solvers.godunov_flux_upwind(
            U_L, U_R, params, junction_info
        )
    
    # 3. Calculate L(U) = -dF/dx (conservative form)
    L_U = np.zeros_like(U)
    for j in range(g, g + N):
        # Finite volume: dU/dt = -(F_{j+1/2} - F_{j-1/2}) / dx
        L_U[:, j] = -(fluxes[:, j] - fluxes[:, j-1]) / grid.dx
    
    return L_U


def primitives_to_conserved_single(P_single, params):
    """
    Convertit un vecteur de variables primitives en variables conservées.
    
    Args:
        P_single (np.ndarray): Vecteur primitif (4,) = [rho_m, v_m, rho_c, v_c]
        params: Paramètres du modèle
        
    Returns:
        np.ndarray: Vecteur conservé (4,) = [rho_m, w_m, rho_c, w_c]
    """
    rho_m, v_m, rho_c, v_c = P_single
    
    # Calcul de la pression
    p_m, p_c = physics.calculate_pressure(
        np.array([rho_m]), np.array([rho_c]), 
        params.physics.alpha, params.physics.rho_jam, params.physics.epsilon,
        params.physics.k_m, params.physics.gamma_m, params.physics.k_c, params.physics.gamma_c
    )
    
    # Variables conservées w = v + p
    w_m = v_m + p_m[0]
    w_c = v_c + p_c[0]
    
    return np.array([rho_m, w_m, rho_c, w_c])


# --- Helper for ODE Step ---

def _ode_rhs(t: float, y: np.ndarray, cell_index: int, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    """
    Right-hand side function for the ODE solver (source term calculation).
    Calculates S(U) for a single cell j.

    Args:
        t (float): Current time (often unused in source term if not time-dependent).
        y (np.ndarray): State vector [rho_m, w_m, rho_c, w_c] for the current cell.
        cell_index (int): The index of the cell (including ghost cells) in the full U array.
        grid (Grid1D): Grid object to access road quality.
        params (ModelParameters): Model parameters.

    Returns:
        np.ndarray: The source term vector dU/dt = S(U) for this cell.
    """
    # Determine the corresponding physical cell index to get R(x)
    # If it's a ghost cell, we might assume a default R or extrapolate,
    # but often the source term is effectively zero in ghost cells anyway
    # unless specific BCs require source terms there.
    # For simplicity, let's use the nearest physical cell's R for ghost cells,
    # or handle based on BC type if needed later.
    physical_idx = max(0, min(cell_index - grid.num_ghost_cells, grid.N_physical - 1))

    if grid.road_quality is None:
         # ✅ BUG #35 FIX: Raise error instead of silent fallback
         # Silent fallback to R=3 was masking initialization failures
         # Equilibrium speed depends CRITICALLY on road quality → must be loaded
         raise ValueError(
             "❌ BUG #35: Road quality array not loaded before ODE solver! "
             "Equilibrium speed Ve calculation requires grid.road_quality. "
             "Fix: Ensure scenario config has 'road: {quality_type: uniform, quality_value: 2}' "
             "and runner._load_road_quality() is called during initialization."
         )
    else:
        R_local = grid.road_quality[physical_idx]

    # Calculate intermediate values needed for the Numba-fied source term
    rho_m = y[0]
    rho_c = y[2]
    rho_m_calc = np.maximum(rho_m, 0.0)
    rho_c_calc = np.maximum(rho_c, 0.0)

    # Check for segment-specific V0 overrides (set by NetworkGrid for heterogeneous networks)
    V0_m_override = getattr(params, '_V0_m_override', None)
    V0_c_override = getattr(params, '_V0_c_override', None)

    # Calculate equilibrium speeds and relaxation times (these are not Numba-fied yet)
    Ve_m, Ve_c = physics.calculate_equilibrium_speed(
        rho_m_calc, rho_c_calc, R_local, params,
        V0_m_override=V0_m_override,
        V0_c_override=V0_c_override
    )
    tau_m, tau_c = physics.calculate_relaxation_time(rho_m_calc, rho_c_calc, params)
    
    # [PHASE 2 DEBUG - Hypothesis C: tau_m numerical issue] - TEMPORARILY DISABLED
    # if cell_index == 5:  # Debug cell
    #     print("[TAU DEBUG cell=", cell_index, "]")
    #     print("  tau_m =", tau_m, ", tau_c =", tau_c)
    #     assert tau_m > 0, "Invalid tau_m"
    #     assert tau_c > 0, "Invalid tau_c"
    #     assert not np.isinf(tau_m), "tau_m is infinite"
    #     assert not np.isinf(tau_c), "tau_c is infinite"
    #     print("  tau values valid")
    
    # DEBUG: Print equilibrium speed calculation
    # DEBUG: Temporarily disabled to reduce output noise
    # if cell_index == 5:  # Only print for one cell
    #     rho_total = rho_m_calc + rho_c_calc
    #     g_factor = max(0.0, 1.0 - rho_total / params.rho_jam)
    #     Vmax_m_R = params.Vmax_m.get(int(R_local), 'MISSING')
    #     print(f"[DEBUG ODE cell={cell_index}] rho_m={rho_m_calc:.4f}, rho_c={rho_c_calc:.4f}, rho_total={rho_total:.4f}")
    #     print(f"  R={R_local}, Vmax_m[R]={Vmax_m_R}, g={g_factor:.4f}, Ve_m={Ve_m:.4f}, V0_override={V0_m_override}")
    #     print(f"  w_m={y[1]:.4f}, IC was w_m=0.7112")

    # Calculate the source term.
    # Note: This function (_ode_rhs) is called by scipy.integrate.solve_ivp
    # for each cell individually. This structure is inherently CPU-based
    # and not suitable for direct GPU acceleration using Numba CUDA kernels,
    # which operate on arrays.
    # For now, the source term calculation within the ODE solver remains CPU-based.

    # [PHASE 2 DEBUG - Pre source term calculation] - TEMPORARILY DISABLED
    # if cell_index == 5:
    #     print("[PRE-SOURCE DEBUG cell=", cell_index, "]")
    #     print("  y[0] (rho_m) =", y[0])
    #     print("  rho_m_calc =", rho_m_calc)
    #     print("  epsilon =", params.epsilon)
    #     print("  rho_m_calc <= epsilon?", rho_m_calc <= params.epsilon)
    
    source = physics.calculate_source_term( # This is the Numba-optimized CPU version
        y,
        # Pressure params
        params.physics.alpha, params.physics.rho_jam, params.physics.k_m, params.physics.gamma_m, params.physics.k_c, params.physics.gamma_c,
        # Equilibrium speeds
        Ve_m, Ve_c,
        # Relaxation times
        tau_m, tau_c,
        # Epsilon
        params.physics.epsilon
    )
    
    # [PHASE 2 DEBUG - Post source term] - TEMPORARILY DISABLED
    # if cell_index == 5:
    #     # Calculate pressure and velocity manually to debug
    #     p_m_calc, p_c_calc = physics.calculate_pressure(
    #         rho_m_calc, rho_c_calc,
    #         params.alpha, params.rho_jam, params.epsilon,
    #         params.K_m, params.gamma_m, params.K_c, params.gamma_c
    #     )
    #     v_m_calc = y[1] - p_m_calc  # v_m = w_m - p_m
    #     print(f"  p_m={p_m_calc:.4f}, v_m_calc={v_m_calc:.4f}")
    #     print(f"  Sm_expected = (Ve_m - v_m)/tau = ({Ve_m:.4f} - {v_m_calc:.4f})/{tau_m:.4f} = {(Ve_m - v_m_calc)/tau_m:.4f}")
    #     print(f"  source[1]={source[1]:.4f} (Sm actual)")
    
    return source


def _ode_rhs_corrected(t: float, y: np.ndarray, cell_index: int, grid: Grid1D, params: ModelParameters, q_correction: float) -> np.ndarray:
    """
    Right-hand side function for ODE solver with boundary correction.
    
    Implements corrected source term: S_corrected = S_original - q_correction
    Based on Einkemmer et al. (2018) for Strang splitting with inflow BC.
    
    Args:
        t: Current time
        y: State vector [rho_m, w_m, rho_c, w_c] for the current cell
        cell_index: Cell index (including ghost cells)
        grid: Grid object
        params: Model parameters
        q_correction: Boundary correction value for this cell
        
    Returns:
        Corrected source term vector dU/dt = S(U) - q
    """
    # Get original source term
    source_original = _ode_rhs(t, y, cell_index, grid, params)
    
    # Apply correction to momentum equation
    # 
    # In ARZ model, the source term for momentum is: d(w)/dt = (Ve - v)/τ
    # where w = v + p is the Lagrangian momentum
    # 
    # The boundary correction modifies this to: d(w)/dt = (Ve - v)/τ - q
    # where q = (Ve_BC - v_BC)/τ is the boundary correction function
    # 
    # This ensures that the source term at the boundary is compatible with
    # the inflow BC during intermediate Strang splitting substeps.
    # 
    # Reference: Einkemmer et al. (2018), Eq. (2.6)-(2.7)
    
    source_corrected = source_original.copy()
    source_corrected[1] -= q_correction  # Modify motorcycle momentum
    source_corrected[3] -= q_correction  # Also apply to car momentum for consistency
    
    return source_corrected


def solve_ode_step_cpu(U_in: np.ndarray, dt_ode: float, grid: Grid1D, params: ModelParameters, 
                      correction_term: np.ndarray | None = None) -> np.ndarray:
    """
    Solves the ODE system dU/dt = S(U) for each cell over a time step dt_ode using the CPU.
    
    Optionally applies boundary correction for Strang splitting with inflow BC.

    Args:
        U_in (np.ndarray): Input state array (including ghost cells). Shape (4, N_total).
        dt_ode (float): Time step for the ODE integration.
        grid (Grid1D): Grid object.
        params (ModelParameters): Model parameters.
        correction_term (np.ndarray | None): Optional boundary correction term q for each cell.
                                             Shape (N_total,). Only affects velocity relaxation.

    Returns:
        np.ndarray: Output state array after the ODE step. Shape (4, N_total).
    """
    U_out = np.copy(U_in) # Start with the input state (preserves ghost cells)

    # ODE solver should only operate on PHYSICAL cells, not ghost cells
    # Ghost cells are managed by boundary conditions
    for j in range(grid.num_ghost_cells, grid.num_ghost_cells + grid.N_physical):
        # Define the RHS function specific to this cell index j
        if correction_term is not None:
            # Use corrected RHS function
            rhs_func = lambda t, y: _ode_rhs_corrected(t, y, j, grid, params, correction_term[j])
        else:
            # Use standard RHS function
            rhs_func = lambda t, y: _ode_rhs(t, y, j, grid, params)

        # Initial state for this cell
        y0 = U_in[:, j]

        # Solve the ODE for this cell
        sol = solve_ivp(
            fun=rhs_func,
            t_span=[0, dt_ode],
            y0=y0,
            method=params.time.ode_solver,
            rtol=params.time.ode_rtol,
            atol=params.time.ode_atol,
            dense_output=False # We only need the final time point
        )

        if not sol.success:
            # Handle solver failure (e.g., log warning, raise error)
            # Might indicate stiffness or issues with parameters/state
            print(f"Warning: ODE solver failed for cell {j} at t={sol.t[-1]}. Status: {sol.status}, Message: {sol.message}")
            # Use the last successful state or initial state as fallback?
            U_out[:, j] = sol.y[:, -1] if sol.y.shape[1] > 0 else y0 # Fallback
        else:
            # Store the solution at the end of the time step
            U_out[:, j] = sol.y[:, -1]
            # Ensure densities remain non-negative after ODE step
            U_out[0, j] = np.maximum(U_out[0, j], params.physics.epsilon) # rho_m
            U_out[2, j] = np.maximum(U_out[2, j], params.physics.epsilon) # rho_c

    return U_out # Return the updated state array

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
def solve_ode_step_gpu(d_U_in: cuda.devicearray.DeviceNDArray, dt_ode: float, grid: Grid1D, params: ModelParameters, d_R: cuda.devicearray.DeviceNDArray) -> cuda.devicearray.DeviceNDArray:
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

    # --- Extract category-specific Vmax values ---
    # Assuming categories 1, 2, 3 exist. Add error handling or defaults if needed.
    try:
        v_max_m_cat1 = params.physics.Vmax_m[1]
        v_max_m_cat2 = params.physics.Vmax_m.get(2, params.physics.Vmax_m[1]) # Default cat 2 to 1 if missing
        v_max_m_cat3 = params.physics.Vmax_m.get(3, params.physics.Vmax_m[1]) # Default cat 3 to 1 if missing

        v_max_c_cat1 = params.physics.Vmax_c[1]
        v_max_c_cat2 = params.physics.Vmax_c.get(2, params.physics.Vmax_c[1]) # Default cat 2 to 1 if missing
        v_max_c_cat3 = params.physics.Vmax_c.get(3, params.physics.Vmax_c[1]) # Default cat 3 to 1 if missing
    except KeyError as e:
        raise ValueError(f"Missing required Vmax for category {e} in parameters (Vmax_m/Vmax_c dictionaries)") from e
    except AttributeError as e:
         raise AttributeError(f"Could not find Vmax_m or Vmax_c dictionaries in parameters object: {e}") from e


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
        # Pressure params
        params.physics.alpha, params.physics.rho_jam, params.physics.K_m, params.physics.gamma_m, params.physics.K_c, params.physics.gamma_c,
        # Equilibrium speed params (base + extracted category Vmax)
        params.physics.rho_jam, params.physics.V_creeping, # Note: rho_jam passed twice, once for pressure, once for eq speed
        v_max_m_cat1, v_max_m_cat2, v_max_m_cat3,
        v_max_c_cat1, v_max_c_cat2, v_max_c_cat3,
        # Relaxation times
        params.physics.tau_m, params.physics.tau_c,
        # Epsilon
        params.physics.epsilon
    )
    # cuda.synchronize() # No sync needed here, let subsequent steps handle it

    # --- 3. Return GPU array ---
    # No copy back to host
    return d_U_out


# --- Boundary Correction Functions (Strang Splitting Fix - Option 2) ---

def compute_boundary_correction(grid: Grid1D, params: ModelParameters, seg_id: str = 'seg_0') -> tuple:
    """
    Computes boundary correction function q for Strang splitting with inflow BC.
    
    Based on Einkemmer et al. (2018) "Efficient boundary corrected Strang splitting".
    The correction function q is defined as: q|boundary = Source(b(t))
    For ARZ model: q = (Ve - v_BC) / τ
    
    Args:
        grid: Grid object
        params: Model parameters with boundary_conditions dict
        seg_id: Segment identifier
        
    Returns:
        (q_left, q_right): Correction values at left and right boundaries
    """
    # Get BC configuration
    bc_params = params.boundary_conditions
    
    # Initialize correction values to zero (no correction for outflow/periodic)
    q_left = 0.0
    q_right = 0.0
    
    left_bc_obj = None
    right_bc_obj = None

    # Handle both segment-level and network-level BC structures
    if isinstance(bc_params, dict):
        # Network-level: bc_params is a dict like {'seg_0': bc_config_obj, ...}
        bc_config_for_segment = bc_params.get(seg_id)
        if bc_config_for_segment:
            left_bc_obj = bc_config_for_segment.left
            right_bc_obj = bc_config_for_segment.right
    elif hasattr(bc_params, 'left'):
        # Segment-level: bc_params is a BoundaryConditionsConfig object itself
        left_bc_obj = bc_params.left
        right_bc_obj = bc_params.right

    # Compute left boundary correction
    if left_bc_obj and left_bc_obj.type == 'inflow':
        # The state is a BCState Pydantic model, not a list
        state = left_bc_obj.state
        rho_m_bc = state.rho_m
        w_m_bc = state.w_m
        rho_c_bc = state.rho_c
        
        # Calculate pressure to convert momentum back to velocity
        p_m_bc, _ = physics.calculate_pressure(
            np.array([rho_m_bc]), np.array([rho_c_bc]),
            params.physics.alpha, params.physics.rho_jam, params.physics.epsilon,
            params.physics.k_m, params.physics.gamma_m, params.physics.k_c, params.physics.gamma_c
        )
        
        if rho_m_bc > params.physics.epsilon:
            v_m_bc = (w_m_bc - p_m_bc[0]) / rho_m_bc
        else:
            v_m_bc = 0.0
        
        q_left = rho_m_bc * v_m_bc

    # Compute right boundary correction (if needed)
    if right_bc_obj and right_bc_obj.type == 'inflow':
        state = right_bc_obj.state
        rho_m_bc = state.rho_m
        w_m_bc = state.w_m
        rho_c_bc = state.rho_c
        
        p_m_bc, _ = physics.calculate_pressure(
            np.array([rho_m_bc]), np.array([rho_c_bc]),
            params.physics.alpha, params.physics.rho_jam, params.physics.epsilon,
            params.physics.k_m, params.physics.gamma_m, params.physics.k_c, params.physics.gamma_c
        )
        
        if rho_m_bc > params.physics.epsilon:
            v_m_bc = (w_m_bc - p_m_bc[0]) / rho_m_bc
        else:
            v_m_bc = 0.0
            
        q_right = rho_m_bc * v_m_bc
    
    return q_left, q_right


def compute_boundary_weight(grid: Grid1D, side: str) -> np.ndarray:
    """
    Computes spatial weight function for boundary correction.
    
    The weight decays exponentially from boundary into domain interior,
    ensuring smooth transition and avoiding artificial discontinuities.
    
    Args:
        grid: Grid object
        side: 'left' or 'right' boundary
        
    Returns:
        weight: Array of weights for all cells (including ghost cells)
    """
    g = grid.num_ghost_cells
    N_total = grid.N_total
    dx = grid.dx
    
    # Decay length: correction extends ~3-5 cells from boundary
    # This is a tunable parameter - can adjust based on stability tests
    decay_length = 3.0 * dx
    
    # Create weight array
    weight = np.zeros(N_total)
    
    if side == 'left':
        # Distance from left boundary for all cells
        x_cells = np.arange(N_total) * dx
        x_boundary = g * dx  # Left boundary position
        distance = x_cells - x_boundary
        
        # Exponential decay from boundary
        # Weight = 1 at boundary, decays to ~0 at distance = 3*decay_length
        weight = np.exp(-np.maximum(distance, 0.0) / decay_length)
        
        # Cut off far from boundary (optional, for efficiency)
        weight[distance > 5 * decay_length] = 0.0
        
    elif side == 'right':
        # Distance from right boundary for all cells
        x_cells = np.arange(N_total) * dx
        x_boundary = (N_total - g) * dx  # Right boundary position
        distance = x_boundary - x_cells
        
        weight = np.exp(-np.maximum(distance, 0.0) / decay_length)
        weight[distance > 5 * decay_length] = 0.0
    
    return weight


# --- Manual Inflow BC Application (Strang Splitting Fix) ---

def apply_inflow_bc_manually(U: np.ndarray, grid: Grid1D, params: ModelParameters, seg_id: str = 'seg_0') -> np.ndarray:
    """
    Manually applies inflow boundary conditions to ghost cells.
    
    This function is used in the Strang splitting BC timing fix to apply
    boundary conditions AFTER ODE substeps instead of during hyperbolic transport.
    
    **CONSERVATIVE MODE**: When BC_APPLICATION_MODE='CONSERVATIVE', this function
    does MINIMAL prescription to avoid creating large gradients. The Riemann solver
    will handle the actual BC enforcement via characteristic decomposition.
    
    Args:
        U: State array (4, N_total)
        grid: Grid object
        params: Model parameters with boundary_conditions dict
        seg_id: Segment identifier to look up BC (default 'seg_0')
        
    Returns:
        State array with BC applied to ghost cells
    """
    from ..config.debug_config import BC_APPLICATION_MODE
    
    # CONSERVATIVE MODE: Skip aggressive BC prescription
    # Let Riemann solver handle BC naturally via flux computation
    if BC_APPLICATION_MODE == 'CONSERVATIVE':
        if DEBUG_LOGS_ENABLED:
            print(f"[MANUAL_BC] CONSERVATIVE mode - skipping aggressive ghost cell prescription")
        return U  # Return unmodified - Riemann solver will handle it
    
    # AGGRESSIVE MODE (original behavior - may cause instability)
    U_bc = U.copy()
    g = grid.num_ghost_cells
    
    if DEBUG_LOGS_ENABLED:
        print(f"[MANUAL_BC_DEBUG] Entry: g={g}, U_bc.shape={U_bc.shape}, U_bc[:, 0:3]={U_bc[:, 0:3]}")
    
    # Get BC configuration - handle both segment-level and network-level structures
    bc_dict = params.boundary_conditions
    
    if DEBUG_LOGS_ENABLED:
        print(f"[MANUAL_BC_DEBUG] bc_dict keys: {list(bc_dict.keys()) if bc_dict else None}")
    
    # Check if it's segment-level (has 'left'/'right' keys) or network-level (has seg_id keys)
    if 'left' in bc_dict or 'right' in bc_dict:
        # Segment-level BC
        left_bc = bc_dict.get('left', {})
        if DEBUG_LOGS_ENABLED:
            print(f"[MANUAL_BC_DEBUG] Segment-level BC: left_bc={left_bc}")
    else:
        # Network-level BC
        bc_config = bc_dict.get(seg_id, {})
        left_bc = bc_config.get('left', {})
        if DEBUG_LOGS_ENABLED:
            print(f"[MANUAL_BC_DEBUG] Network-level BC: bc_config={bc_config}, left_bc={left_bc}")
    
    # Only apply if it's an inflow BC
    if left_bc.get('type') == 'inflow':
        if DEBUG_LOGS_ENABLED:
            print(f"[MANUAL_BC_DEBUG] INFLOW DETECTED! left_bc={left_bc}")
        # Extract BC values - handle both direct values and 'state' array format
        if 'state' in left_bc:
            # Format: {'type': 'inflow', 'state': [rho_m, w_m, rho_c, w_c]}
            # state already contains momentum (w_m, w_c), so use it directly
            state = left_bc['state']
            if DEBUG_LOGS_ENABLED:
                print(f"[MANUAL_BC_APPLY] Using state format: {state}")
                print(f"[MANUAL_BC_APPLY] BEFORE: U_bc[:, 0:3]={U_bc[:, 0:3]}")
            U_bc[0, :g] = state[0]  # rho_m
            U_bc[1, :g] = state[1]  # w_m (momentum)
            U_bc[2, :g] = state[2]  # rho_c
            U_bc[3, :g] = state[3]  # w_c
            if DEBUG_LOGS_ENABLED:
                print(f"[MANUAL_BC_APPLY] AFTER: U_bc[:, 0:3]={U_bc[:, 0:3]}")
        else:
            # Format: {'type': 'inflow', 'rho_m': ..., 'v_m': ..., ...}
            # Need to convert velocity to momentum
            rho_m_bc = left_bc.get('rho_m', 0.15)
            v_m_bc = left_bc.get('v_m', 3.0)
            rho_c_bc = left_bc.get('rho_c', 0.0)
            v_c_bc = left_bc.get('v_c', 0.0)
            
            # Calculate pressure and momentum
            p_m_bc, p_c_bc = physics.calculate_pressure(
                np.array([rho_m_bc]), np.array([rho_c_bc]),
                params.alpha, params.rho_jam, params.epsilon,
                params.K_m, params.gamma_m, params.K_c, params.gamma_c
            )
            
            # ✅ FIX: w = ρ*v + p (pas v + p!)
            # Momentum généralisé ARZ = flux + pression
            w_m_bc = rho_m_bc * v_m_bc + p_m_bc[0]
            w_c_bc = rho_c_bc * v_c_bc + p_c_bc[0]
            
            if DEBUG_LOGS_ENABLED:
                print(f"[MANUAL_BC_APPLY] Using velocity format: rho_m={rho_m_bc}, w_m={w_m_bc}")
                print(f"[MANUAL_BC_APPLY] BEFORE: U_bc[:, 0:3]={U_bc[:, 0:3]}")
            
            # Apply to left ghost cells
            U_bc[0, :g] = rho_m_bc
            U_bc[1, :g] = w_m_bc
            U_bc[2, :g] = rho_c_bc
            U_bc[3, :g] = w_c_bc
            
            if DEBUG_LOGS_ENABLED:
                print(f"[MANUAL_BC_APPLY] AFTER: U_bc[:, 0:3]={U_bc[:, 0:3]}")
    else:
        if DEBUG_LOGS_ENABLED:
            print(f"[MANUAL_BC_DEBUG] NOT INFLOW! left_bc type={left_bc.get('type', 'MISSING')}")
    
    if DEBUG_LOGS_ENABLED:
        print(f"[MANUAL_BC_DEBUG] EXIT: U_bc[:, 0:3]={U_bc[:, 0:3]}")
    return U_bc


# --- Strang Splitting Step ---

# --- Strang Splitting Step ---

def strang_splitting_step(U_or_d_U_n, dt: float, grid: Grid1D, params: ModelParameters, d_R=None, current_bc_params: dict | None = None, seg_id: str = None, apply_bc: bool = True, current_time: float = 0.0):
    """
    Performs one full time step using Strang splitting.
    Handles both CPU and GPU arrays based on params.device.

    Args:
        U_or_d_U_n (np.ndarray or cuda.devicearray.DeviceNDArray): State array at time n.
        dt (float): The full time step.
        grid (Grid1D): Grid object.
        params (ModelParameters): Model parameters (including device).
        d_R (cuda.devicearray.DeviceNDArray, optional): GPU road quality array.
                                                        Required if params.device == 'gpu'. Defaults to None.
        current_bc_params (dict | None): Mise à jour des paramètres BC (pour inflow dynamique). Defaults to None.
        seg_id (str, optional): Segment identifier for BC application in network context. Defaults to None.
        apply_bc (bool): If True, applies boundary conditions within this function. Defaults to True.
        current_time (float): Current simulation time (for logging control). Defaults to 0.0.

    Returns:
        np.ndarray or cuda.devicearray.DeviceNDArray: State array at time n+1 (same type as input).
    """
    # Update Riemann solver's current time for frequency-based logging
    riemann_solvers.set_current_time(current_time)
    
    # DEBUG: Confirm Strang splitting is called
    if DEBUG_LOGS_ENABLED and hasattr(params, 'boundary_conditions') and params.boundary_conditions is not None:
        bc_keys = list(params.boundary_conditions.keys()) if isinstance(params.boundary_conditions, dict) else "NON-DICT"
        # Show full BC for left if it exists
        left_bc = params.boundary_conditions.get('left') if isinstance(params.boundary_conditions, dict) else None
        print(f"[STRANG] BC keys: {bc_keys}, left={left_bc}")
    elif DEBUG_LOGS_ENABLED:
        print(f"[STRANG] BC=None")
    
    # CFL stability check (CPU path only, for now)
    if params.device == 'cpu' and not cuda.is_cuda_array(U_or_d_U_n):
        is_stable, CFL = check_cfl_condition(U_or_d_U_n, grid, params, dt, CFL_max=0.9)
        if not is_stable:
            warnings.warn(
                f"CFL condition violated! CFL={CFL:.3f} > 0.9. "
                f"Consider reducing timestep dt={dt:.6f} or increasing grid resolution dx={grid.dx:.3f}.",
                RuntimeWarning,
                stacklevel=2
            )
    
    if params.device == 'gpu':
        # --- GPU Path ---
        if not cuda.is_cuda_array(U_or_d_U_n):
            raise TypeError("Device is 'gpu' but input U_or_d_U_n is not a GPU array.")
        if d_R is None or not cuda.is_cuda_array(d_R):
             raise ValueError("GPU road quality array d_R must be provided for GPU Strang splitting.")

        d_U_n = U_or_d_U_n # Rename for clarity

        # Step 1: Solve ODEs for dt/2
        d_U_star = solve_ode_step_gpu(d_U_n, dt / 2.0, grid, params, d_R)

        # Step 2: Solve Hyperbolic part for full dt (with current_bc_params for dynamic BC)
        # Dynamic solver selection
        if params.grid.spatial_scheme == 'first_order' and params.time.time_scheme == 'euler':
            # Use SSP-RK3 as fallback for simple first-order Euler GPU
            d_U_ss = solve_hyperbolic_step_ssprk3_gpu(d_U_star, dt, grid, params, current_bc_params)
        elif params.grid.spatial_scheme == 'first_order' and params.time.time_scheme == 'ssprk3':
            d_U_ss = solve_hyperbolic_step_ssprk3_gpu(d_U_star, dt, grid, params, current_bc_params)
        elif params.grid.spatial_scheme == 'weno5' and params.time.time_scheme == 'euler':
            d_U_ss = solve_hyperbolic_step_weno_gpu(d_U_star, dt, grid, params, current_bc_params)
        elif params.grid.spatial_scheme == 'weno5' and params.time.time_scheme == 'ssprk3':
            d_U_ss = solve_hyperbolic_step_ssprk3_gpu(d_U_star, dt, grid, params, current_bc_params)
        else:
            raise ValueError(f"GPU device currently supports: "
                           f"('first_order', 'euler'), ('first_order', 'ssprk3'), "
                           f"('weno5', 'euler'), ('weno5', 'ssprk3'). "
                           f"Requested: spatial_scheme='{params.grid.spatial_scheme}', time_scheme='{params.time.time_scheme}'")

        # Step 3: Solve ODEs for dt/2
        d_U_np1 = solve_ode_step_gpu(d_U_ss, dt / 2.0, grid, params, d_R)

        return d_U_np1

    elif params.device == 'cpu':
        # --- CPU Path ---
        if cuda.is_cuda_array(U_or_d_U_n):
            raise TypeError("Device is 'cpu' but input U_or_d_U_n is a GPU array.")

        U_n = U_or_d_U_n # Rename for clarity
        
        # --- INFLOW BC TIMING FIX (BUG_31) ---
        # Detect if we have inflow boundary conditions for THIS segment
        has_inflow_bc = False
        seg_with_inflow = None
        
        if DEBUG_LOGS_ENABLED:
            print(f"[STRANG_FIX_DEBUG] seg_id={seg_id}, has BC={hasattr(params, 'boundary_conditions')}, BC={params.boundary_conditions if hasattr(params, 'boundary_conditions') else None}")
        
        if hasattr(params, 'boundary_conditions') and params.boundary_conditions:
            # Check if params.boundary_conditions is segment-level or network-level
            # Segment-level has keys like 'left', 'right'
            # Network-level has keys like 'seg_0', 'seg_1'
            bc_dict = params.boundary_conditions
            
            # Heuristic: Check if we have a dictionary of BCs (network) or a single BC object (segment)
            bc_params = params.boundary_conditions
            if isinstance(bc_params, dict):
                # Network-level BC (dictionary mapping seg_id to BC objects)
                if seg_id is not None:
                    bc_config = bc_params.get(seg_id) # Get the BC object for this segment
                    if bc_config and hasattr(bc_config, 'left') and bc_config.left.type == 'inflow':
                        has_inflow_bc = True
                        seg_with_inflow = seg_id
                        if DEBUG_LOGS_ENABLED:
                            print(f"[STRANG_FIX_DEBUG] Network-level INFLOW BC DETECTED! seg={seg_with_inflow}")
                else:
                    # Fallback for single-segment simulation with network-style config
                    # Try to find the first segment with an inflow BC
                    for seg_key, config in bc_params.items():
                        if hasattr(config, 'left') and config.left.type == 'inflow':
                            has_inflow_bc = True
                            seg_with_inflow = seg_key
                            if DEBUG_LOGS_ENABLED:
                                print(f"[STRANG_FIX_DEBUG] Network-level INFLOW BC DETECTED (fallback)! seg={seg_with_inflow}")
                            break
            elif hasattr(bc_params, 'left'):
                # Segment-level BC (already a single BoundaryConditionsConfig object)
                if bc_params.left.type == 'inflow':
                    has_inflow_bc = True
                    seg_with_inflow = seg_id if seg_id is not None else 'seg_0' # Assume seg_0 if not specified
                    if DEBUG_LOGS_ENABLED:
                        print(f"[STRANG_FIX_DEBUG] Segment-level INFLOW BC DETECTED! seg={seg_with_inflow}")
        
        if has_inflow_bc:
            # --- OPTION 2: BOUNDARY CORRECTION METHOD (Einkemmer et al. 2018) ---
            # Apply boundary correction to source term to prevent order reduction
            # This is more robust than Option 1 (BC timing modification)
            
            if should_log(current_time):
                print(f"[BC_CORRECTION] t={current_time:.1f}s - Computing boundary correction for segment {seg_with_inflow}")
            
            # Step 1: Compute correction function q at boundaries
            q_left, q_right = compute_boundary_correction(grid, params, seg_with_inflow)
            
            if should_log(current_time):
                print(f"[BC_CORRECTION] t={current_time:.1f}s - q_left={q_left:.6f}, q_right={q_right:.6f}")
            
            # Step 2: Compute spatial weight functions
            weight_left = compute_boundary_weight(grid, 'left')
            weight_right = compute_boundary_weight(grid, 'right')
            
            # Step 3: Combine corrections into full correction term
            # correction_term[j] = q_left * weight_left[j] + q_right * weight_right[j]
            correction_term = q_left * weight_left + q_right * weight_right
            
            if should_log(current_time):
                print(f"[BC_CORRECTION] t={current_time:.1f}s - Max correction: {np.max(np.abs(correction_term)):.6f}")
                print(f"[BC_CORRECTION] t={current_time:.1f}s - Correction at boundary cells: {correction_term[:5]}")
            
            # Step 4: First ODE substep WITH correction
            U_star = solve_ode_step_cpu(U_n, dt / 2.0, grid, params, correction_term=correction_term)
            
            # Step 5: Apply BC after first ODE substep
            U_star = apply_inflow_bc_manually(U_star, grid, params, seg_with_inflow)
            
            # Step 6: Hyperbolic substep (standard, with BC)
        # Dynamic solver selection
        if params.grid.spatial_scheme == 'first_order':
            # first_order + SSP-RK3
            U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params, current_bc_params, apply_bc)
        elif params.grid.spatial_scheme == 'godunov':
            U_ss = solve_hyperbolic_step_godunov_cpu(U_star, dt, grid, params, current_bc_params)
        elif params.grid.spatial_scheme == 'weno5':
            U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params, current_bc_params, apply_bc)
        else:
            raise ValueError(f"Unsupported CPU spatial_scheme: '{params.grid.spatial_scheme}'")
        
        # Step 7: Second ODE substep WITH correction
        U_np1 = solve_ode_step_cpu(U_ss, dt / 2.0, grid, params, correction_term=correction_term)
        
        # Step 8: Apply BC after second ODE substep
        U_np1 = apply_inflow_bc_manually(U_np1, grid, params, seg_with_inflow)
    
    else:
        # --- ORIGINAL STRANG SPLITTING (for outflow BC or no BC) ---
        # Step 1: Solve ODEs for dt/2
        U_star = solve_ode_step_cpu(U_n, dt / 2.0, grid, params)

        # Step 2: Solve Hyperbolic part for full dt
        # Dynamic solver selection based on spatial_scheme and time_scheme
        if params.grid.spatial_scheme == 'first_order' and params.time.time_scheme == 'euler':
            # Use SSP-RK3 as fallback for simple first-order Euler
            U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params, current_bc_params, apply_bc)
        elif params.grid.spatial_scheme == 'first_order' and params.time.time_scheme == 'ssprk3':
            # Phase 4.2: First-order spatial + SSP-RK3 temporal (CPU et GPU)
            if params.device == 'gpu':
                U_ss = solve_hyperbolic_step_ssprk3_gpu(U_star, dt, grid, params, current_bc_params)
            else:
                U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params, current_bc_params, apply_bc)
        elif params.grid.spatial_scheme == 'weno5' and params.time.time_scheme == 'euler':
            # Use SSP-RK3 as fallback for WENO5 + Euler
            U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params, current_bc_params, apply_bc)
        elif params.grid.spatial_scheme == 'weno5' and params.time.time_scheme == 'ssprk3':
            U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params, current_bc_params, apply_bc)
        elif params.grid.spatial_scheme == 'godunov' and params.time.time_scheme == 'euler':
            # Godunov + Euler (use SSP-RK3 as fallback)
            U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params, current_bc_params, apply_bc)
            U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params, current_bc_params, apply_bc)
        elif params.grid.spatial_scheme == 'godunov' and params.time.time_scheme == 'ssprk3':
            # Godunov + SSP-RK3 (recommended combination)
            U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params, current_bc_params, apply_bc)
        else:
            raise ValueError(f"Unsupported scheme combination: spatial_scheme='{params.grid.spatial_scheme}', time_scheme='{params.time.time_scheme}'. "
                           f"Supported combinations: (first_order, euler), (first_order, ssprk3), (weno5, euler), (weno5, ssprk3), "
                           f"(godunov, euler), (godunov, ssprk3)")

        # Step 3: Solve ODEs for dt/2
        U_np1 = solve_ode_step_cpu(U_ss, dt / 2.0, grid, params)
    
    # CRITICAL: Apply physical bounds to prevent state explosion
    # This is a last-resort safety net to catch any numerical instabilities
    # Use 1.5 * rho_jam to allow temporary over-jam states (congestion peaks)
    # but prevent extreme explosions (2025-11-02: Balance stability vs physics)
    U_np1 = apply_physical_state_bounds(U_np1, grid, params, rho_max=1.5 * params.physics.rho_jam)

    return U_np1


def strang_splitting_step_gpu(U_gpu, dt: float, grid: Grid1D, params: ModelParameters, seg_id: str = None):
    """
    Pure GPU Strang splitting using Numba CUDA (no CuPy, no conversions, no CPU transfers).
    
    Architecture:
        1. ODE substep (dt/2): w_t = S(w) via Numba GPU kernels
        2. Hyperbolic substep (dt): w_t + F(w)_x = 0 via SSP-RK3 + WENO5 GPU kernels
        3. ODE substep (dt/2): w_t = S(w) via Numba GPU kernels
    
    Implementation:
        - Pure Numba CUDA throughout (no CuPy conversions)
        - All computations stay on GPU (zero CPU transfers)
        - Leverages existing Numba kernels: solve_ode_step_gpu, solve_hyperbolic_step_ssprk3_gpu
    
    Args:
        U_gpu: Numba device array (4, N_total) on GPU
        dt: Timestep (s)
        grid: Grid1D (CPU object with geometry info)
        params: ModelParameters (device='gpu')
        seg_id: Segment identifier for BC handling
        
    Returns:
        Numba device array U_new_gpu (4, N_total) on GPU
        
    Performance:
        5-10x speedup vs CPU version by keeping all data on GPU
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU kernels not available. Check Numba CUDA installation.")
    
    # ===== PREPARE ROAD QUALITY ARRAY (needed for ODE step) =====
    # Transfer road quality to GPU (small array, ideally cached in NetworkGrid)
       # NOTE: This could be optimized by caching d_R in the segment dictionary
    d_R = cuda.to_device(grid.road_quality[grid.physical_cell_indices])
    
    # ===== STEP 1: ODE substep (dt/2) - PURE GPU =====
    U_star = solve_ode_step_gpu(U_gpu, dt / 2.0, grid, params, d_R)
    
    # ===== STEP 2: Hyperbolic substep (dt) - PURE GPU =====
    current_bc_params = None  # TODO: Pass from NetworkGrid if needed for dynamic BC
    U_ss = solve_hyperbolic_step_ssprk3_gpu(U_star, dt, grid, params, current_bc_params)
    
    # ===== STEP 3: ODE substep (dt/2) - PURE GPU =====
    U_new = solve_ode_step_gpu(U_ss, dt / 2.0, grid, params, d_R)
    
    # ===== APPLY PHYSICAL BOUNDS (PURE GPU) =====
    U_new = apply_physical_state_bounds_gpu(U_new, grid, params, rho_max=1.5 * params.physics.rho_jam)
    
    # Return Numba device array (stays on GPU)
    return U_new


# --- SSP-RK3 Time Integration ---

def solve_hyperbolic_step_ssprk3(U_in: np.ndarray, dt_hyp: float, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None, apply_bc: bool = True) -> np.ndarray:
    """
    Résout l'étape hyperbolique dU/dt + dF/dx = 0 en utilisant le schéma SSP-RK3.
    
    Support les combinaisons:
    - first_order + SSP-RK3 (via flux Central-Upwind)
    - weno5 + SSP-RK3 (via WENO5 + SSP-RK3)
    
    Le schéma SSP-RK3 (Strong Stability Preserving Runge-Kutta d'ordre 3) est :
    U^{(1)} = U^n + dt * L(U^n)
    U^{(2)} = 3/4 * U^n + 1/4 * U^{(1)} + 1/4 * dt * L(U^{(1)})
    U^{n+1} = 1/3 * U^n + 2/3 * U^{(2)} + 2/3 * dt * L(U^{(2)})

    où L(U) est la discrétisation spatiale selon params.spatial_scheme.

    Args:
        U_in (np.ndarray): État d'entrée (4, N_total)
        dt_hyp (float): Pas de temps hyperbolique
        grid (Grid1D): Objet grille
        params (ModelParameters): Paramètres du modèle
        current_bc_params (dict | None): Mise à jour des paramètres BC (pour inflow dynamique). Defaults to None.

    Returns:
        np.ndarray: État mis à jour après SSP-RK3
    """
    # DEBUG: Confirm SSP-RK3 is called
    if not hasattr(solve_hyperbolic_step_ssprk3, '_call_count'):
        solve_hyperbolic_step_ssprk3._call_count = 0
    solve_hyperbolic_step_ssprk3._call_count += 1
    
    if solve_hyperbolic_step_ssprk3._call_count <= 5:
        print(f"[SSP-RK3 #{solve_hyperbolic_step_ssprk3._call_count}] spatial_scheme={params.grid.spatial_scheme}, current_bc_params={current_bc_params is not None}")
    
    # Choisir la fonction de discrétisation spatiale selon le schéma
    if params.grid.spatial_scheme == 'first_order':
        compute_L = lambda U: -compute_flux_divergence_first_order(U, grid, params, current_bc_params)
    elif params.grid.spatial_scheme == 'weno5':
        compute_L = lambda U: calculate_spatial_discretization_weno(U, grid, params, current_bc_params, apply_bc)
    elif params.grid.spatial_scheme == 'godunov':
        compute_L = lambda U: calculate_spatial_discretization_godunov(U, grid, params, current_bc_params)
    else:
        raise ValueError(f"Unsupported spatial_scheme '{params.grid.spatial_scheme}' for SSP-RK3. "
                        f"Valid options: 'first_order', 'weno5', 'godunov'")
    
    # --- Étape 1 : U^{(1)} = U^n + dt * L(U^n) ---
    L_U_n = compute_L(U_in)
    U_1 = apply_temporal_update_ssprk3(U_in, L_U_n, dt_hyp, grid, params)
    
    # --- Étape 2 : U^{(2)} = 3/4 * U^n + 1/4 * U^{(1)} + 1/4 * dt * L(U^{(1)}) ---
    L_U_1 = compute_L(U_1)
    U_2 = np.copy(U_in)
    g, N = grid.num_ghost_cells, grid.N_physical
    
    for j in range(g, g + N):  # Cellules physiques seulement
        U_2[:, j] = (3.0/4.0) * U_in[:, j] + (1.0/4.0) * U_1[:, j] + (1.0/4.0) * dt_hyp * L_U_1[:, j]
    
    # Application du plancher après l'étape 2
    U_2[0, :] = np.maximum(U_2[0, :], params.physics.epsilon)  # rho_m
    U_2[2, :] = np.maximum(U_2[2, :], params.physics.epsilon)  # rho_c
    
    # --- Étape 3 : U^{n+1} = 1/3 * U^n + 2/3 * U^{(2)} + 2/3 * dt * L(U^{(2)}) ---
    L_U_2 = compute_L(U_2)
    U_out = np.copy(U_in)
    
    for j in range(g, g + N):  # Cellules physiques seulement
        U_out[:, j] = (1.0/3.0) * U_in[:, j] + (2.0/3.0) * U_2[:, j] + (2.0/3.0) * dt_hyp * L_U_2[:, j]
    
    # Vérification des densités négatives
    neg_rho_m_indices = np.where(U_out[0, :] < 0)[0]
    neg_rho_c_indices = np.where(U_out[2, :] < 0)[0]

    if len(neg_rho_m_indices) > 0:
        print(f"Warning: Negative rho_m detected in SSP-RK3 step in cells: {neg_rho_m_indices}. Applying floor.")
    if len(neg_rho_c_indices) > 0:
        print(f"Warning: Negative rho_c detected in SSP-RK3 step in cells: {neg_rho_c_indices}. Applying floor.")

    # Application du plancher final
    U_out[0, :] = np.maximum(U_out[0, :], params.physics.epsilon)  # rho_m
    U_out[2, :] = np.maximum(U_out[2, :], params.physics.epsilon)  # rho_c

    return U_out


def apply_temporal_update_ssprk3(U_in: np.ndarray, L_U: np.ndarray, dt: float, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    """
    Applique la mise à jour temporelle U_out = U_in + dt * L_U avec plancher de densité pour SSP-RK3.
    
    Args:
        U_in (np.ndarray): État d'entrée
        L_U (np.ndarray): Discrétisation spatiale L(U)
        dt (float): Pas de temps
        grid (Grid1D): Objet grille
        params (ModelParameters): Paramètres du modèle
        
    Returns:
        np.ndarray: État mis à jour
    """
    U_out = np.copy(U_in)
    g, N = grid.num_ghost_cells, grid.N_physical
    
    for j in range(g, g + N):  # Cellules physiques seulement
        U_out[:, j] = U_in[:, j] + dt * L_U[:, j]
    
    # Application du plancher
    U_out[0, :] = np.maximum(U_out[0, :], params.physics.epsilon)  # rho_m
    U_out[2, :] = np.maximum(U_out[2, :], params.physics.epsilon)  # rho_c
    
    return U_out


def compute_flux_divergence_first_order(U: np.ndarray, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None) -> np.ndarray:
    """
    Calcule la divergence des flux -dF/dx pour le schéma du premier ordre.
    
    Args:
        U (np.ndarray): État conservé (4, N_total)
        grid (Grid1D): Objet grille
        params (ModelParameters): Paramètres du modèle
        current_bc_params (dict | None): Mise à jour des paramètres BC (pour inflow dynamique). Defaults to None.
        
    Returns:
        np.ndarray: Divergence des flux dF/dx (4, N_total)
    """
    # Application des conditions aux limites
    U_bc = np.copy(U)
    boundary_conditions.apply_boundary_conditions(U_bc, grid, params, current_bc_params)
    
    fluxes = np.zeros((4, grid.N_total))
    g = grid.num_ghost_cells
    N = grid.N_physical
    
    # Calcul des flux aux interfaces
    for j in range(g - 1, g + N):  # F_{j+1/2} pour j=g-1..g+N-1
        U_L = U_bc[:, j]
        U_R = U_bc[:, j + 1]
        
        # Check if this is the junction interface (rightmost physical cell)
        junction_info = None
        if j == g + N - 1 and hasattr(grid, 'junction_at_right') and grid.junction_at_right is not None:
            junction_info = grid.junction_at_right
        
        fluxes[:, j] = riemann_solvers.central_upwind_flux(U_L, U_R, params, junction_info)
    
    # Calcul de la divergence dF/dx
    flux_div = np.zeros_like(U)
    for j in range(g, g + N):  # Cellules physiques seulement
        flux_right = fluxes[:, j]      # F_{j+1/2}
        flux_left = fluxes[:, j - 1]   # F_{j-1/2}
        flux_div[:, j] = (flux_right - flux_left) / grid.dx
    
    return flux_div


# --- GPU WENO and SSP-RK3 Implementations ---

def solve_hyperbolic_step_gpu(d_U_in: cuda.devicearray.DeviceNDArray, dt_hyp: float, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None) -> cuda.devicearray.DeviceNDArray:
    """
    Version GPU générique de l'étape hyperbolique - utilise SSP-RK3 par défaut.
    Cette fonction est un wrapper qui dirige vers la bonne implémentation selon le schéma.
    
    Args:
        d_U_in (cuda.devicearray.DeviceNDArray): État d'entrée sur GPU (4, N_total)
        dt_hyp (float): Pas de temps hyperbolique
        grid (Grid1D): Objet grille
        params (ModelParameters): Paramètres du modèle
        current_bc_params (dict | None): Mise à jour des paramètres BC (pour inflow dynamique)

    Returns:
        cuda.devicearray.DeviceNDArray: État mis à jour après intégration sur GPU
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU kernels not available. Check GPU imports.")
    
    if not cuda.is_cuda_array(d_U_in):
        raise TypeError("d_U_in must be a CUDA device array")
    
    # Utiliser SSP-RK3 par défaut pour la compatibilité
    return solve_hyperbolic_step_ssprk3_gpu(d_U_in, dt_hyp, grid, params, current_bc_params)


def solve_hyperbolic_step_weno_gpu(d_U_in: cuda.devicearray.DeviceNDArray, dt_hyp: float, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None) -> cuda.devicearray.DeviceNDArray:
    """
    Version GPU de solve_hyperbolic_step_weno_cpu() utilisant WENO5 + Euler.
    
    Args:
        d_U_in (cuda.devicearray.DeviceNDArray): État d'entrée sur GPU (4, N_total)
        dt_hyp (float): Pas de temps hyperbolique
        grid (Grid1D): Objet grille
        params (ModelParameters): Paramètres du modèle
        current_bc_params (dict | None): Mise à jour des paramètres BC (pour inflow dynamique)

    Returns:
        cuda.devicearray.DeviceNDArray: État mis à jour après WENO5 + Euler sur GPU
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU WENO implementation not available. Check GPU imports.")
    
    if not cuda.is_cuda_array(d_U_in):
        raise TypeError("d_U_in must be a CUDA device array")
    
    # DEBUG: Log parameter passing
    print(f"[DEBUG_SOLVE_HYPERBOLIC_WENO_GPU] current_bc_params: {type(current_bc_params)}")
    
    # Calcul de la discrétisation spatiale L(U) = -dF/dx avec WENO5 GPU
    d_L_U = calculate_spatial_discretization_weno_gpu(d_U_in, grid, params, current_bc_params)
    
    # Mise à jour temporelle Euler sur GPU
    d_U_out = cuda.device_array_like(d_U_in)
    
    # Configuration des kernels
    threadsperblock = 256
    blockspergrid = (grid.N_physical + threadsperblock - 1) // threadsperblock
    
    # Kernel pour la mise à jour temporelle
    _apply_euler_update_kernel[blockspergrid, threadsperblock](
        d_U_in, d_L_U, d_U_out, dt_hyp, params.physics.epsilon, 
        grid.num_ghost_cells, grid.N_physical
    )
    
    return d_U_out


def solve_hyperbolic_step_ssprk3_gpu(d_U_in: cuda.devicearray.DeviceNDArray, dt_hyp: float, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None) -> cuda.devicearray.DeviceNDArray:
    """
    Version GPU de solve_hyperbolic_step_ssprk3() utilisant les kernels CUDA existants.
    
    Support les combinaisons:
    - first_order + SSP-RK3 (via flux Central-Upwind existant)
    - weno5 + SSP-RK3 (via WENO5 GPU + SSP-RK3 GPU)
    
    Args:
        d_U_in (cuda.devicearray.DeviceNDArray): État d'entrée sur GPU (4, N_total)
        dt_hyp (float): Pas de temps hyperbolique
        grid (Grid1D): Objet grille
        params (ModelParameters): Paramètres du modèle
        current_bc_params (dict | None): Mise à jour des paramètres BC (pour inflow dynamique)

    Returns:
        cuda.devicearray.DeviceNDArray: État mis à jour après SSP-RK3 sur GPU
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU SSP-RK3 implementation not available. Check GPU imports.")
    
    if not cuda.is_cuda_array(d_U_in):
        raise TypeError("d_U_in must be a CUDA device array")
    
    N_physical = grid.N_physical
    N_total = grid.N_total
    num_variables = 4  # rho_m, w_m, rho_c, w_c
    
    # Fonction pour calculer la discrétisation spatiale L(U) = -dF/dx
    def compute_spatial_discretization_gpu_callback(d_U_state, d_L_out):
        """
        Callback pour SSP_RK3 qui calcule L(U) = -dF/dx.
        
        Args:
            d_U_state: État sur GPU (N_physical, 4) - format attendu par SSP_RK3_GPU
            d_L_out: Sortie L(U) sur GPU (N_physical, 4)
        """
        # Convertir le format (N_physical, 4) vers (4, N_total) avec cellules fantômes
        d_U_extended = cuda.device_array((4, N_total), dtype=d_U_in.dtype)
        
        # Copier les cellules fantômes depuis l'état d'entrée original
        n_ghost = grid.num_ghost_cells
        d_U_extended[:, :n_ghost] = d_U_in[:, :n_ghost]  # Cellules fantômes gauches
        d_U_extended[:, n_ghost+N_physical:] = d_U_in[:, n_ghost+N_physical:]  # Cellules fantômes droites


        # Copier les cellules physiques depuis d_U_state (transposer)
        _transpose_physical_cells_kernel[
            (N_physical + 255) // 256, 256
        ](d_U_state, d_U_extended, n_ghost, N_physical)
        
        # Appliquer les conditions aux limites avec current_bc_params
        if current_bc_params is not None:
            boundary_conditions.apply_boundary_conditions(d_U_extended, grid, params, current_bc_params)
        
        # Calculer la discrétisation spatiale selon le schéma choisi
        if params.grid.spatial_scheme == 'first_order':
            # Utiliser la méthode du premier ordre existante
            d_fluxes = riemann_solvers.central_upwind_flux_gpu(d_U_extended, params)
            
            # Calculer la divergence des flux : L(U) = -dF/dx
            _compute_flux_divergence_kernel[
                (N_physical + 255) // 256, 256
            ](d_U_extended, d_fluxes, d_L_out, grid.dx, params.physics.epsilon, n_ghost, N_physical)
            
        elif params.grid.spatial_scheme == 'weno5':
            # Utiliser WENO5 GPU avec current_bc_params
            d_L_extended = calculate_spatial_discretization_weno_gpu(d_U_extended, grid, params, current_bc_params)
            
            # Extraire les cellules physiques et transposer vers le format (N_physical, 4)
            _extract_physical_cells_kernel[
                (N_physical + 255) // 256, 256
            ](d_L_extended, d_L_out, n_ghost, N_physical)
        else:
            raise ValueError(f"Unsupported spatial_scheme '{params.grid.spatial_scheme}' for GPU SSP-RK3")
    
    # Préparer les données pour SSP_RK3_GPU
    # Convertir d_U_in de (4, N_total) vers (N_physical, 4) pour les cellules physiques uniquement
    d_U_physical = cuda.device_array((N_physical, num_variables), dtype=d_U_in.dtype)
    n_ghost = grid.num_ghost_cells
    
    _extract_physical_cells_to_format_kernel[
        (N_physical + 255) // 256, 256
    ](d_U_in, d_U_physical, n_ghost, N_physical)
    
    # Créer l'intégrateur SSP-RK3 GPU
    integrator = SSP_RK3_GPU(N_physical, num_variables)
    
    # Préparer la sortie
    d_U_result = cuda.device_array_like(d_U_physical)
    
    # Effectuer l'intégration SSP-RK3
    integrator.integrate_step(d_U_physical, d_U_result, dt_hyp, compute_spatial_discretization_gpu_callback)
    
    # Convertir le résultat de (N_physical, 4) vers (4, N_total)
    d_U_out = cuda.device_array_like(d_U_in)
    
    # Copier les cellules fantômes depuis l'entrée
    d_U_out[:, :n_ghost] = d_U_in[:, :n_ghost]
    d_U_out[:, n_ghost+N_physical:] = d_U_in[:, n_ghost+N_physical:]
    
    # Copier les cellules physiques depuis le résultat (transposer)
    _insert_physical_cells_from_format_kernel[
        (N_physical + 255) // 256, 256
    ](d_U_result, d_U_out, n_ghost, N_physical)
    
    # Appliquer le plancher de densité
    _apply_density_floor_kernel[
        (N_physical + 255) // 256, 256
    ](d_U_out, params.physics.epsilon, n_ghost, N_physical)
    
    # Nettoyer l'intégrateur
    integrator.cleanup()
    
    return d_U_out


def calculate_spatial_discretization_weno_gpu(d_U_in: cuda.devicearray.DeviceNDArray, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None) -> cuda.devicearray.DeviceNDArray:
    """
    Version GPU de calculate_spatial_discretization_weno utilisant les kernels CUDA WENO5 existants.
    
    Args:
        d_U_in (cuda.devicearray.DeviceNDArray): État conservé sur GPU (4, N_total)
        grid (Grid1D): Objet grille
        params (ModelParameters): Paramètres du modèle
        current_bc_params (dict | None): Mise à jour des paramètres BC (pour inflow dynamique)
        
    Returns:
        cuda.devicearray.DeviceNDArray: L(U) = -dF/dx sur GPU (4, N_total)
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU WENO implementation not available. Check GPU imports.")
    
    # Utiliser l'implémentation GPU native complète
    try:
        from .reconstruction.weno_gpu import calculate_spatial_discretization_weno_gpu_native
        d_L_U = calculate_spatial_discretization_weno_gpu_native(d_U_in, grid, params, current_bc_params)
        return d_L_U
    except ImportError:
        # Fallback temporaire si la fonction native n'existe pas encore
        print("Warning: Using CPU fallback for WENO GPU calculation. Implementing native GPU version...")
        U_cpu = d_U_in.copy_to_host()
        L_U_cpu = calculate_spatial_discretization_weno(U_cpu, grid, params)
        d_L_U = cuda.to_device(L_U_cpu)
        return d_L_U


# --- Kernels CUDA Helper Functions ---

@cuda.jit
def _apply_euler_update_kernel(d_U_in, d_L_U, d_U_out, dt, epsilon, num_ghost_cells, N_physical):
    """
    Kernel pour appliquer la mise à jour temporelle Euler : U_out = U_in + dt * L_U
    """
    idx = cuda.grid(1)
    
    if idx < N_physical:
        j = num_ghost_cells + idx
        
        for var in range(4):
            d_U_out[var, j] = d_U_in[var, j] + dt * d_L_U[var, j]
        
        # Appliquer le plancher de densité
        d_U_out[0, j] = max(d_U_out[0, j], epsilon)  # rho_m
        d_U_out[2, j] = max(d_U_out[2, j], epsilon)  # rho_c
        d_U_out[2, j] = max(d_U_out[2, j], epsilon)  # rho_c


@cuda.jit
def _transpose_physical_cells_kernel(d_U_state, d_U_extended, n_ghost, N_physical):
    """
    Kernel pour transposer de (N_physical, 4) vers (4, N_physical) dans d_U_extended.
    """
    idx = cuda.grid(1)
    
    if idx < N_physical:
        j_extended = n_ghost + idx
        for var in range(4):
            d_U_extended[var, j_extended] = d_U_state[idx, var]


@cuda.jit
def _extract_physical_cells_kernel(d_L_extended, d_L_out, n_ghost, N_physical):
    """
    Kernel pour extraire les cellules physiques et transposer vers (N_physical, 4).
    """
    idx = cuda.grid(1)
    
    if idx < N_physical:
        j_extended = n_ghost + idx
        for var in range(4):
            d_L_out[idx, var] = d_L_extended[var, j_extended]


@cuda.jit
def _extract_physical_cells_to_format_kernel(d_U_in, d_U_physical, n_ghost, N_physical):
    """
    Kernel pour extraire les cellules physiques de (4, N_total) vers (N_physical, 4).
    """
    idx = cuda.grid(1)
    
    if idx < N_physical:
        j_in = n_ghost + idx
        for var in range(4):
            d_U_physical[idx, var] = d_U_in[var, j_in]


@cuda.jit
def _insert_physical_cells_from_format_kernel(d_U_result, d_U_out, n_ghost, N_physical):
    """
    Kernel pour insérer les cellules physiques de (N_physical, 4) vers (4, N_total).
    """
    idx = cuda.grid(1)
    
    if idx < N_physical:
        j_out = n_ghost + idx
        for var in range(4):
            d_U_out[var, j_out] = d_U_result[idx, var]


@cuda.jit
def _compute_flux_divergence_kernel(d_U, d_fluxes, d_L_out, dx, epsilon, num_ghost_cells, N_physical):
    """
    Kernel CUDA pour calculer la divergence des flux L(U) = -dF/dx.
    """
    idx = cuda.grid(1)
    
    if idx < N_physical:
        j = num_ghost_cells + idx  # Index global avec cellules fantômes
        dx_inv = 1.0 / dx
        
        for var in range(4):
            # Flux à droite et à gauche de la cellule j
            flux_right = d_fluxes[var, j]       # F_{j+1/2}
            flux_left = d_fluxes[var, j-1]      # F_{j-1/2}
            
            # Divergence: L(U) = -dF/dx = -(F_{j+1/2} - F_{j-1/2})/dx
            d_L_out[idx, var] = -(flux_right - flux_left) * dx_inv


@cuda.jit
def _apply_density_floor_kernel(d_U, epsilon, num_ghost_cells, N_physical):
    """
    Kernel CUDA pour appliquer le plancher de densité.
    """
    idx = cuda.grid(1)
    
    if idx < N_physical:
        j = num_ghost_cells + idx
        
        # Appliquer le plancher pour les densités
        d_U[0, j] = max(d_U[0, j], epsilon)  # rho_m
        d_U[2, j] = max(d_U[2, j], epsilon)  # rho_c


# --- Network-Aware Time Integration ---

def strang_splitting_step_with_network(U_n, dt, grid, params, nodes, network_coupling, current_bc_params=None):
    """
    Strang splitting avec couplage réseau stable.

    Args:
        U_n: État au temps n
        dt: Pas de temps
        grid: Grille
        params: Paramètres
        nodes: Liste des nœuds
        network_coupling: Gestionnaire de couplage réseau
        current_bc_params (dict | None): Mise à jour des paramètres BC (pour inflow dynamique)

    Returns:
        État au temps n+1
    """
    time = 0.0  # TODO: Passer le temps réel depuis SimulationRunner

    if params.device == 'gpu':
        # Version GPU avec couplage stable
        from .network_coupling_stable import apply_network_coupling_stable_gpu
        d_U_n = U_n  # Assume déjà sur GPU

        # Étape 1: ODE dt/2
        d_U_star = solve_ode_step_gpu(d_U_n, dt / 2.0, grid, params, d_R)

        # Étape 2: Solve Hyperbolic part for full dt (with current_bc_params for dynamic BC)
        # Dynamic solver selection
        if params.grid.spatial_scheme == 'first_order' and params.time.time_scheme == 'euler':
            # Use SSP-RK3 as fallback for simple first-order Euler GPU
            d_U_ss = solve_hyperbolic_step_ssprk3_gpu(d_U_star, dt, grid, params, current_bc_params)
        elif params.grid.spatial_scheme == 'first_order' and params.time.time_scheme == 'ssprk3':
            d_U_ss = solve_hyperbolic_step_ssprk3_gpu(d_U_star, dt, grid, params, current_bc_params)
        elif params.grid.spatial_scheme == 'weno5' and params.time.time_scheme == 'euler':
            d_U_ss = solve_hyperbolic_step_weno_gpu(d_U_star, dt, grid, params, current_bc_params)
        elif params.grid.spatial_scheme == 'weno5' and params.time.time_scheme == 'ssprk3':
            d_U_ss = solve_hyperbolic_step_ssprk3_gpu(d_U_star, dt, grid, params, current_bc_params)
        else:
            raise ValueError(f"GPU device currently supports: "
                           f"('first_order', 'euler'), ('first_order', 'ssprk3'), "
                           f"('weno5', 'euler'), ('weno5', 'ssprk3'). "
                           f"Requested: spatial_scheme='{params.grid.spatial_scheme}', time_scheme='{params.time.time_scheme}'")

        # Étape 3: ODE dt/2
        d_U_np1 = solve_ode_step_gpu(d_U_ss, dt / 2.0, grid, params, d_R)

        return d_U_np1

    else:
        # Version CPU avec couplage stable
        from .network_coupling_stable import apply_network_coupling_stable

        # Étape 1: ODE dt/2
        U_star = solve_ode_step_cpu(U_n, dt / 2.0, grid, params)

        # Étape 2: Hyperbolique avec couplage réseau stable et current_bc_params
        U_with_bc = apply_network_coupling_stable(U_star, dt, grid, params, time)
        U_ss = solve_hyperbolic_step_standard(U_with_bc, dt, grid, params)

        # Étape 3: ODE dt/2
        U_np1 = solve_ode_step_cpu(U_ss, dt / 2.0, grid, params)

        return U_np1


def solve_hyperbolic_step_standard(U, dt, grid, params):
    """
    Résout l'étape hyperbolique standard selon le schéma configuré.
    """
    if params.spatial_scheme == 'first_order' and params.time_scheme == 'euler':
        return solve_hyperbolic_step_ssprk3(U, dt, grid, params)
    elif params.spatial_scheme == 'first_order' and params.time_scheme == 'ssprk3':
        return solve_hyperbolic_step_ssprk3(U, dt, grid, params)
    elif params.spatial_scheme == 'weno5' and params.time_scheme == 'euler':
        return solve_hyperbolic_step_ssprk3(U, dt, grid, params)
    elif params.spatial_scheme == 'weno5' and params.time_scheme == 'ssprk3':
        return solve_hyperbolic_step_ssprk3(U, dt, grid, params)
    else:
        raise ValueError(f"Unsupported scheme combination: spatial_scheme='{params.spatial_scheme}', time_scheme='{params.time_scheme}'")


def solve_hyperbolic_step_standard_gpu(d_U, dt, grid, params, current_bc_params=None):
    """
    Résout l'étape hyperbolique standard selon le schéma configuré (GPU).
    """
    if params.spatial_scheme == 'first_order' and params.time_scheme == 'euler':
        return solve_hyperbolic_step_ssprk3_gpu(d_U, dt, grid, params, current_bc_params)
    elif params.spatial_scheme == 'first_order' and params.time_scheme == 'ssprk3':
        return solve_hyperbolic_step_ssprk3_gpu(d_U, dt, grid, params, current_bc_params)
    elif params.spatial_scheme == 'weno5' and params.time_scheme == 'euler':
        return solve_hyperbolic_step_weno_gpu(d_U, dt, grid, params, current_bc_params)
    elif params.spatial_scheme == 'weno5' and params.time_scheme == 'ssprk3':
        return solve_hyperbolic_step_ssprk3_gpu(d_U, dt, grid, params, current_bc_params)
    else:
        raise ValueError(f"Unsupported scheme combination: spatial_scheme='{params.spatial_scheme}', time_scheme='{params.time_scheme}'")

