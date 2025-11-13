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
                                           params: 'PhysicsConfig', stream: cuda.cudadrv.driver.Stream):
    """
    Applies physical bounds to a state array on the GPU, modifying it in-place.
    This is a helper to be called from within other GPU-orchestrating functions.
    """
    threads_per_block = 256
    blocks_per_grid = math.ceil(grid.N_physical / threads_per_block)
    
    # Max velocity is not directly in params, needs conversion if units are km/h
    # Assuming v_max is provided in m/s or calculated before calling
    v_max_physical = 50.0 # m/s, equivalent to 180 km/h. TODO: Get from config.

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
    DEPRECATED: This function is part of the legacy CPU/GPU hybrid architecture.
    In the GPU-only architecture, this logic is replaced by `strang_splitting_step_gpu_native`.
    This function will be removed in a future version.
    """
    raise NotImplementedError(
        "DEPRECATED: `strang_splitting_step` is a legacy CPU/hybrid function. "
        "The GPU-only architecture uses `strang_splitting_step_gpu_native` "
        "which is orchestrated by the `NetworkSimulator`."
    )


def strang_splitting_step_gpu(U_gpu, dt: float, grid: Grid1D, params: ModelParameters, seg_id: str = None):
    """
    DEPRECATED: This function is part of the legacy CPU/GPU hybrid architecture.
    In the GPU-only architecture, this logic is replaced by `strang_splitting_step_gpu_native`.
    This function will be removed in a future version.
    """
    raise NotImplementedError(
        "DEPRECATED: `strang_splitting_step_gpu` is a legacy hybrid function. "
        "The GPU-only architecture uses `strang_splitting_step_gpu_native`."
    )


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

    # --- RK Stage 2 ---
    # L_U1 = L(U_1)
    L_U1 = calculate_spatial_discretization_weno_gpu_native(d_U1, grid, params, gpu_pool, seg_id, current_time)
    # U_2 = (3/4)U_n + (1/4)U_1 + (1/4)dt * L(U_1)
    ssp_rk3_stage_2_kernel[blockspergrid, threadsperblock](d_U_in, d_U1, L_U1, dt, d_U2)

    # --- RK Stage 3 ---
    # L_U2 = L(U_2)
    L_U2 = calculate_spatial_discretization_weno_gpu_native(d_U2, grid, params, gpu_pool, seg_id, current_time)
    # U_np1 = (1/3)U_n + (2/3)U_2 + (2/3)dt * L(U_2)
    d_U_out = gpu_pool.get_temp_array(d_U_in.shape, d_U_in.dtype) # Get a new array for the output
    ssp_rk3_stage_3_kernel[blockspergrid, threadsperblock](d_U_in, d_U2, L_U2, dt, d_U_out)

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



