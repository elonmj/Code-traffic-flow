import numpy as np
from numba import cuda # Import cuda
from scipy.integrate import solve_ivp
from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters
from ..core import physics
import math # Import math for ceil
from . import riemann_solvers # Import the riemann solver module
from . import boundary_conditions
from .reconstruction.weno import reconstruct_weno5
from .reconstruction.converter import conserved_to_primitives_arr, primitives_to_conserved_arr

# Import GPU implementations
try:
    from .gpu.weno_cuda import reconstruct_weno5_gpu_naive, reconstruct_weno5_gpu_optimized
    from .gpu.ssp_rk3_cuda import SSP_RK3_GPU, integrate_ssp_rk3_gpu
    GPU_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GPU implementations not available: {e}")
    GPU_AVAILABLE = False

# --- WENO-Based Spatial Discretization ---

def calculate_spatial_discretization_weno(U: np.ndarray, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    """
    Calcule la discrétisation spatiale L(U) = -dF/dx en utilisant la reconstruction WENO5.
    
    Cette fonction orchestre :
    1. La conversion des variables conservées vers les variables primitives
    2. La reconstruction WENO5 des variables primitives aux interfaces
    3. Le calcul des flux via le solveur de Riemann Central-Upwind
    4. Le calcul de la dérivée spatiale du flux
    
    Args:
        U (np.ndarray): État conservé (4, N_total) incluant les cellules fantômes
        grid (Grid1D): Objet grille
        params (ModelParameters): Paramètres du modèle
        
    Returns:
        np.ndarray: Discrétisation spatiale L(U) = -dF/dx (4, N_total)
    """
    # 0. Application des conditions aux limites sur l'état d'entrée
    U_bc = np.copy(U)
    boundary_conditions.apply_boundary_conditions(U_bc, grid, params)
    
    # 1. Conversion vers les variables primitives
    P = conserved_to_primitives_arr(
        U_bc, params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c
    )
    
    # 2. Reconstruction WENO5 pour chaque variable primitive
    N_total = U.shape[1]
    P_left = np.zeros_like(P)   # Variables primitives reconstruites à gauche des interfaces
    P_right = np.zeros_like(P)  # Variables primitives reconstruites à droite des interfaces
    
    for var_idx in range(4):  # Pour chaque variable (rho_m, v_m, rho_c, v_c)
        P_left[var_idx, :], P_right[var_idx, :] = reconstruct_weno5(P[var_idx, :])
    
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
            
            # Conversion vers variables conservées pour le flux
            U_L = primitives_to_conserved_single(P_L, params)
            U_R = primitives_to_conserved_single(P_R, params)
            
            # Calcul du flux Central-Upwind
            fluxes[:, j] = riemann_solvers.central_upwind_flux(U_L, U_R, params)
    
    # 4. Calcul de la discrétisation spatiale L(U) = -dF/dx
    L_U = np.zeros_like(U)
    
    for j in range(g, g + N):  # Cellules physiques seulement
        flux_right = fluxes[:, j]      # F_{j+1/2}
        flux_left = fluxes[:, j - 1]   # F_{j-1/2}
        L_U[:, j] = -(flux_right - flux_left) / grid.dx
    
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
        params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c
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

    # Calculate equilibrium speeds and relaxation times (these are not Numba-fied yet)
    Ve_m, Ve_c = physics.calculate_equilibrium_speed(rho_m_calc, rho_c_calc, R_local, params)
    tau_m, tau_c = physics.calculate_relaxation_time(rho_m_calc, rho_c_calc, params)

    # Calculate the source term.
    # Note: This function (_ode_rhs) is called by scipy.integrate.solve_ivp
    # for each cell individually. This structure is inherently CPU-based
    # and not suitable for direct GPU acceleration using Numba CUDA kernels,
    # which operate on arrays.
    # The 'device' parameter primarily influences the hyperbolic step and
    # other array-based physics calculations if they were moved here.
    # For now, the source term calculation within the ODE solver remains CPU-based.

    source = physics.calculate_source_term( # This is the Numba-optimized CPU version
        y,
        # Pressure params
        params.alpha, params.rho_jam, params.K_m, params.gamma_m, params.K_c, params.gamma_c,
        # Equilibrium speeds
        Ve_m, Ve_c,
        # Relaxation times
        tau_m, tau_c,
        # Epsilon
        params.epsilon
    )
    return source


def solve_ode_step_cpu(U_in: np.ndarray, dt_ode: float, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    """
    Solves the ODE system dU/dt = S(U) for each cell over a time step dt_ode using the CPU.

    Args:
        U_in (np.ndarray): Input state array (including ghost cells). Shape (4, N_total).
        dt_ode (float): Time step for the ODE integration.
        grid (Grid1D): Grid object.
        params (ModelParameters): Model parameters.

    Returns:
        np.ndarray: Output state array after the ODE step. Shape (4, N_total).
    """
    U_out = np.copy(U_in) # Start with the input state

    for j in range(grid.N_total):
        # Define the RHS function specific to this cell index j
        rhs_func = lambda t, y: _ode_rhs(t, y, j, grid, params)

        # Initial state for this cell
        y0 = U_in[:, j]

        # Solve the ODE for this cell
        sol = solve_ivp(
            fun=rhs_func,
            t_span=[0, dt_ode],
            y0=y0,
            method=params.ode_solver,
            rtol=params.ode_rtol,
            atol=params.ode_atol,
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
            U_out[0, j] = np.maximum(U_out[0, j], params.epsilon) # rho_m
            U_out[2, j] = np.maximum(U_out[2, j], params.epsilon) # rho_c

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
        v_max_m_cat1 = params.Vmax_m[1]
        v_max_m_cat2 = params.Vmax_m.get(2, params.Vmax_m[1]) # Default cat 2 to 1 if missing
        v_max_m_cat3 = params.Vmax_m.get(3, params.Vmax_m[1]) # Default cat 3 to 1 if missing

        v_max_c_cat1 = params.Vmax_c[1]
        v_max_c_cat2 = params.Vmax_c.get(2, params.Vmax_c[1]) # Default cat 2 to 1 if missing
        v_max_c_cat3 = params.Vmax_c.get(3, params.Vmax_c[1]) # Default cat 3 to 1 if missing
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
        params.alpha, params.rho_jam, params.K_m, params.gamma_m, params.K_c, params.gamma_c,
        # Equilibrium speed params (base + extracted category Vmax)
        params.rho_jam, params.V_creeping, # Note: rho_jam passed twice, once for pressure, once for eq speed
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


# --- Strang Splitting Step ---

def strang_splitting_step(U_or_d_U_n, dt: float, grid: Grid1D, params: ModelParameters, d_R=None):
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

    Returns:
        np.ndarray or cuda.devicearray.DeviceNDArray: State array at time n+1 (same type as input).
    """
    if params.device == 'gpu':
        # --- GPU Path ---
        if not cuda.is_cuda_array(U_or_d_U_n):
            raise TypeError("Device is 'gpu' but input U_or_d_U_n is not a GPU array.")
        if d_R is None or not cuda.is_cuda_array(d_R):
             raise ValueError("GPU road quality array d_R must be provided for GPU Strang splitting.")

        d_U_n = U_or_d_U_n # Rename for clarity

        # Step 1: Solve ODEs for dt/2
        d_U_star = solve_ode_step_gpu(d_U_n, dt / 2.0, grid, params, d_R)

        # Step 2: Solve Hyperbolic part for full dt
        # Dynamic solver selection
        if params.spatial_scheme == 'first_order' and params.time_scheme == 'euler':
            # Use SSP-RK3 as fallback for simple first-order Euler GPU
            d_U_ss = solve_hyperbolic_step_ssprk3_gpu(d_U_star, dt, grid, params)
        elif params.spatial_scheme == 'first_order' and params.time_scheme == 'ssprk3':
            d_U_ss = solve_hyperbolic_step_ssprk3_gpu(d_U_star, dt, grid, params)
        elif params.spatial_scheme == 'weno5' and params.time_scheme == 'euler':
            d_U_ss = solve_hyperbolic_step_weno_gpu(d_U_star, dt, grid, params)
        elif params.spatial_scheme == 'weno5' and params.time_scheme == 'ssprk3':
            d_U_ss = solve_hyperbolic_step_ssprk3_gpu(d_U_star, dt, grid, params)
        else:
            raise ValueError(f"GPU device currently supports: "
                           f"('first_order', 'euler'), ('first_order', 'ssprk3'), "
                           f"('weno5', 'euler'), ('weno5', 'ssprk3'). "
                           f"Requested: spatial_scheme='{params.spatial_scheme}', time_scheme='{params.time_scheme}'")

        # Step 3: Solve ODEs for dt/2
        d_U_np1 = solve_ode_step_gpu(d_U_ss, dt / 2.0, grid, params, d_R)

        return d_U_np1

    elif params.device == 'cpu':
        # --- CPU Path ---
        if cuda.is_cuda_array(U_or_d_U_n):
            raise TypeError("Device is 'cpu' but input U_or_d_U_n is a GPU array.")

        U_n = U_or_d_U_n # Rename for clarity

        # Step 1: Solve ODEs for dt/2
        U_star = solve_ode_step_cpu(U_n, dt / 2.0, grid, params)

        # Step 2: Solve Hyperbolic part for full dt
        # Dynamic solver selection based on spatial_scheme and time_scheme
        if params.spatial_scheme == 'first_order' and params.time_scheme == 'euler':
            # Use SSP-RK3 as fallback for simple first-order Euler
            U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params)
        elif params.spatial_scheme == 'first_order' and params.time_scheme == 'ssprk3':
            # Phase 4.2: First-order spatial + SSP-RK3 temporal (CPU et GPU)
            if params.device == 'gpu':
                U_ss = solve_hyperbolic_step_ssprk3_gpu(U_star, dt, grid, params)
            else:
                U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params)
        elif params.spatial_scheme == 'weno5' and params.time_scheme == 'euler':
            # Use SSP-RK3 as fallback for WENO5 + Euler
            U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params)
        elif params.spatial_scheme == 'weno5' and params.time_scheme == 'ssprk3':
            U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params)
        else:
            raise ValueError(f"Unsupported scheme combination: spatial_scheme='{params.spatial_scheme}', time_scheme='{params.time_scheme}'. "
                           f"Supported combinations: (first_order, euler), (first_order, ssprk3), (weno5, euler), (weno5, ssprk3)")

        # Step 3: Solve ODEs for dt/2
        U_np1 = solve_ode_step_cpu(U_ss, dt / 2.0, grid, params)

        return U_np1

    else:
        raise ValueError("Invalid device type in parameters. Expected 'cpu' or 'gpu'.")

# --- SSP-RK3 Time Integration ---

def solve_hyperbolic_step_ssprk3(U_in: np.ndarray, dt_hyp: float, grid: Grid1D, params: ModelParameters) -> np.ndarray:
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

    Returns:
        np.ndarray: État mis à jour après SSP-RK3
    """
    
    # Choisir la fonction de discrétisation spatiale selon le schéma
    if params.spatial_scheme == 'first_order':
        compute_L = lambda U: -compute_flux_divergence_first_order(U, grid, params)
    elif params.spatial_scheme == 'weno5':
        compute_L = lambda U: calculate_spatial_discretization_weno(U, grid, params)
    else:
        raise ValueError(f"Unsupported spatial_scheme '{params.spatial_scheme}' for SSP-RK3")
    
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
    U_2[0, :] = np.maximum(U_2[0, :], params.epsilon)  # rho_m
    U_2[2, :] = np.maximum(U_2[2, :], params.epsilon)  # rho_c
    
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
    U_out[0, :] = np.maximum(U_out[0, :], params.epsilon)  # rho_m
    U_out[2, :] = np.maximum(U_out[2, :], params.epsilon)  # rho_c

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
    U_out[0, :] = np.maximum(U_out[0, :], params.epsilon)  # rho_m
    U_out[2, :] = np.maximum(U_out[2, :], params.epsilon)  # rho_c
    
    return U_out


def compute_flux_divergence_first_order(U: np.ndarray, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    """
    Calcule la divergence des flux -dF/dx pour le schéma du premier ordre.
    
    Args:
        U (np.ndarray): État conservé (4, N_total)
        grid (Grid1D): Objet grille
        params (ModelParameters): Paramètres du modèle
        
    Returns:
        np.ndarray: Divergence des flux dF/dx (4, N_total)
    """
    # Application des conditions aux limites
    U_bc = np.copy(U)
    boundary_conditions.apply_boundary_conditions(U_bc, grid, params)
    
    fluxes = np.zeros((4, grid.N_total))
    g = grid.num_ghost_cells
    N = grid.N_physical
    
    # Calcul des flux aux interfaces
    for j in range(g - 1, g + N):  # F_{j+1/2} pour j=g-1..g+N-1
        U_L = U_bc[:, j]
        U_R = U_bc[:, j + 1]
        fluxes[:, j] = riemann_solvers.central_upwind_flux(U_L, U_R, params)
    
    # Calcul de la divergence dF/dx
    flux_div = np.zeros_like(U)
    for j in range(g, g + N):  # Cellules physiques seulement
        flux_right = fluxes[:, j]      # F_{j+1/2}
        flux_left = fluxes[:, j - 1]   # F_{j-1/2}
        flux_div[:, j] = (flux_right - flux_left) / grid.dx
    
    return flux_div


# --- GPU WENO and SSP-RK3 Implementations ---

def solve_hyperbolic_step_gpu(d_U_in: cuda.devicearray.DeviceNDArray, dt_hyp: float, grid: Grid1D, params: ModelParameters) -> cuda.devicearray.DeviceNDArray:
    """
    Version GPU générique de l'étape hyperbolique - utilise SSP-RK3 par défaut.
    Cette fonction est un wrapper qui dirige vers la bonne implémentation selon le schéma.
    
    Args:
        d_U_in (cuda.devicearray.DeviceNDArray): État d'entrée sur GPU (4, N_total)
        dt_hyp (float): Pas de temps hyperbolique
        grid (Grid1D): Objet grille
        params (ModelParameters): Paramètres du modèle

    Returns:
        cuda.devicearray.DeviceNDArray: État mis à jour après intégration sur GPU
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU implementation not available. Check GPU imports.")
    
    if not cuda.is_cuda_array(d_U_in):
        raise TypeError("d_U_in must be a CUDA device array")
    
    # Utiliser SSP-RK3 par défaut pour la compatibilité
    return solve_hyperbolic_step_ssprk3_gpu(d_U_in, dt_hyp, grid, params)


def solve_hyperbolic_step_weno_gpu(d_U_in: cuda.devicearray.DeviceNDArray, dt_hyp: float, grid: Grid1D, params: ModelParameters) -> cuda.devicearray.DeviceNDArray:
    """
    Version GPU de solve_hyperbolic_step_weno_cpu() utilisant WENO5 + Euler.
    
    Args:
        d_U_in (cuda.devicearray.DeviceNDArray): État d'entrée sur GPU (4, N_total)
        dt_hyp (float): Pas de temps hyperbolique
        grid (Grid1D): Objet grille
        params (ModelParameters): Paramètres du modèle

    Returns:
        cuda.devicearray.DeviceNDArray: État mis à jour après WENO5 + Euler sur GPU
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU WENO implementation not available. Check GPU imports.")
    
    if not cuda.is_cuda_array(d_U_in):
        raise TypeError("d_U_in must be a CUDA device array")
    
    # Calcul de la discrétisation spatiale L(U) = -dF/dx avec WENO5 GPU
    d_L_U = calculate_spatial_discretization_weno_gpu(d_U_in, grid, params)
    
    # Mise à jour temporelle Euler sur GPU
    d_U_out = cuda.device_array_like(d_U_in)
    
    # Configuration des kernels
    threadsperblock = 256
    blockspergrid = (grid.N_physical + threadsperblock - 1) // threadsperblock
    
    # Kernel pour la mise à jour temporelle
    _apply_euler_update_kernel[blockspergrid, threadsperblock](
        d_U_in, d_L_U, d_U_out, dt_hyp, params.epsilon, 
        grid.num_ghost_cells, grid.N_physical
    )
    
    return d_U_out


def solve_hyperbolic_step_ssprk3_gpu(d_U_in: cuda.devicearray.DeviceNDArray, dt_hyp: float, grid: Grid1D, params: ModelParameters) -> cuda.devicearray.DeviceNDArray:
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
        Callback pour SSP_RK3_GPU qui calcule L(U) = -dF/dx.
        
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
        
        # Appliquer les conditions aux limites
        boundary_conditions.apply_boundary_conditions_gpu(d_U_extended, grid, params)
        
        # Calculer la discrétisation spatiale selon le schéma choisi
        if params.spatial_scheme == 'first_order':
            # Utiliser la méthode du premier ordre existante
            d_fluxes = riemann_solvers.central_upwind_flux_gpu(d_U_extended, params)
            
            # Calculer la divergence des flux : L(U) = -dF/dx
            _compute_flux_divergence_kernel[
                (N_physical + 255) // 256, 256
            ](d_U_extended, d_fluxes, d_L_out, grid.dx, params.epsilon, n_ghost, N_physical)
            
        elif params.spatial_scheme == 'weno5':
            # Utiliser WENO5 GPU
            d_L_extended = calculate_spatial_discretization_weno_gpu(d_U_extended, grid, params)
            
            # Extraire les cellules physiques et transposer vers le format (N_physical, 4)
            _extract_physical_cells_kernel[
                (N_physical + 255) // 256, 256
            ](d_L_extended, d_L_out, n_ghost, N_physical)
        else:
            raise ValueError(f"Unsupported spatial_scheme '{params.spatial_scheme}' for GPU SSP-RK3")
    
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
    ](d_U_out, params.epsilon, n_ghost, N_physical)
    
    # Nettoyer l'intégrateur
    integrator.cleanup()
    
    return d_U_out


def calculate_spatial_discretization_weno_gpu(d_U_in: cuda.devicearray.DeviceNDArray, grid: Grid1D, params: ModelParameters) -> cuda.devicearray.DeviceNDArray:
    """
    Version GPU de calculate_spatial_discretization_weno utilisant les kernels CUDA WENO5 existants.
    
    Args:
        d_U_in (cuda.devicearray.DeviceNDArray): État conservé sur GPU (4, N_total)
        grid (Grid1D): Objet grille
        params (ModelParameters): Paramètres du modèle
        
    Returns:
        cuda.devicearray.DeviceNDArray: L(U) = -dF/dx sur GPU (4, N_total)
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU WENO implementation not available. Check GPU imports.")
    
    # Utiliser l'implémentation GPU native complète
    try:
        from .reconstruction.weno_gpu import calculate_spatial_discretization_weno_gpu_native
        d_L_U = calculate_spatial_discretization_weno_gpu_native(d_U_in, grid, params)
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

def strang_splitting_step_with_network(U_n, dt, grid, params, nodes, network_coupling):
    """
    Strang splitting avec couplage réseau stable.

    Args:
        U_n: État au temps n
        dt: Pas de temps
        grid: Grille
        params: Paramètres
        nodes: Liste des nœuds
        network_coupling: Gestionnaire de couplage réseau

    Returns:
        État au temps n+1
    """
    time = 0.0  # TODO: Passer le temps réel depuis SimulationRunner

    if params.device == 'gpu':
        # Version GPU avec couplage stable
        from .network_coupling_stable import apply_network_coupling_stable_gpu
        d_U_n = U_n  # Assume déjà sur GPU

        # Étape 1: ODE dt/2
        d_U_star = solve_ode_step_gpu(d_U_n, dt / 2.0, grid, params, None)

        # Étape 2: Hyperbolique avec couplage réseau stable
        d_U_with_bc = apply_network_coupling_stable_gpu(d_U_star, dt, grid, params, time)
        d_U_ss = solve_hyperbolic_step_standard_gpu(d_U_with_bc, dt, grid, params)

        # Étape 3: ODE dt/2
        d_U_np1 = solve_ode_step_gpu(d_U_ss, dt / 2.0, grid, params, None)

        return d_U_np1

    else:
        # Version CPU avec couplage stable
        from .network_coupling_stable import apply_network_coupling_stable

        # Étape 1: ODE dt/2
        U_star = solve_ode_step_cpu(U_n, dt / 2.0, grid, params)

        # Étape 2: Hyperbolique avec couplage réseau stable
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


def solve_hyperbolic_step_standard_gpu(d_U, dt, grid, params):
    """
    Résout l'étape hyperbolique standard selon le schéma configuré (GPU).
    """
    if params.spatial_scheme == 'first_order' and params.time_scheme == 'euler':
        return solve_hyperbolic_step_ssprk3_gpu(d_U, dt, grid, params)
    elif params.spatial_scheme == 'first_order' and params.time_scheme == 'ssprk3':
        return solve_hyperbolic_step_ssprk3_gpu(d_U, dt, grid, params)
    elif params.spatial_scheme == 'weno5' and params.time_scheme == 'euler':
        return solve_hyperbolic_step_weno_gpu(d_U, dt, grid, params)
    elif params.spatial_scheme == 'weno5' and params.time_scheme == 'ssprk3':
        return solve_hyperbolic_step_ssprk3_gpu(d_U, dt, grid, params)
    else:
        raise ValueError(f"Unsupported scheme combination: spatial_scheme='{params.spatial_scheme}', time_scheme='{params.time_scheme}'")

