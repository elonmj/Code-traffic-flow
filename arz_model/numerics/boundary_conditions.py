import numpy as np
from numba import cuda, float64, int32 # Import cuda and types
import math # For ceil
from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters
from ..core import physics # Import the physics module for pressure calculation

# --- CUDA Kernel ---

@cuda.jit
def _apply_boundary_conditions_kernel(d_U, n_ghost, n_phys,
                                      left_type_code, right_type_code,
                                      inflow_L_0, inflow_L_1, inflow_L_2, inflow_L_3,
                                      inflow_R_0, inflow_R_1, inflow_R_2, inflow_R_3,
                                      # Add physics parameters needed for wall BC pressure calc
                                      alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c,
                                      # Placeholder for future capped BC parameters if needed separately
                                      rho_cap_factor): # Added for potential future use, needed for consistency now
    """
    CUDA kernel to apply boundary conditions directly on the GPU device array d_U.

    Args:
        d_U: Device array (4, N_total) modified in-place.
        n_ghost: Number of ghost cells.
        n_phys: Number of physical cells.
        left_type_code: 0=inflow, 1=outflow, 2=periodic.
        right_type_code: 0=inflow, 1=outflow, 2=periodic.
        inflow_L_0..3: Left inflow state values (if left_type_code==0).
        inflow_R_0..3: Right inflow state values (if right_type_code==0).
    """
    # Thread index corresponds to the ghost cell layer (0 to n_ghost-1)
    i = cuda.grid(1)

    if i < n_ghost:
        # --- Left Boundary ---
        left_ghost_idx = i
        if left_type_code == 0: # Inflow (Modified: Impose full state [rho_m, w_m, rho_c, w_c])
            first_phys_idx = n_ghost
            d_U[0, left_ghost_idx] = inflow_L_0 # Impose rho_m
            d_U[1, left_ghost_idx] = inflow_L_1 # Impose w_m (FIXED: was extrapolated)
            d_U[2, left_ghost_idx] = inflow_L_2 # Impose rho_c
            d_U[3, left_ghost_idx] = inflow_L_3 # Impose w_c (FIXED: was extrapolated)
        elif left_type_code == 1: # Outflow (zero-order extrapolation)
            first_phys_idx = n_ghost
            d_U[0, left_ghost_idx] = d_U[0, first_phys_idx]
            d_U[1, left_ghost_idx] = d_U[1, first_phys_idx]
            d_U[2, left_ghost_idx] = d_U[2, first_phys_idx]
            d_U[3, left_ghost_idx] = d_U[3, first_phys_idx]
        elif left_type_code == 2: # Periodic
            src_idx = n_phys + i # Copy from right physical cells
            d_U[0, left_ghost_idx] = d_U[0, src_idx]
            d_U[1, left_ghost_idx] = d_U[1, src_idx]
            d_U[2, left_ghost_idx] = d_U[2, src_idx]
            d_U[3, left_ghost_idx] = d_U[3, src_idx]
        elif left_type_code == 3: # Wall (v=0 -> w=p)
            first_phys_idx = n_ghost
            # Get state from first physical cell
            rho_m_phys = d_U[0, first_phys_idx]
            rho_c_phys = d_U[2, first_phys_idx]
            # Calculate pressure at physical cell state
            p_m_phys, p_c_phys = physics._calculate_pressure_cuda(
                rho_m_phys, rho_c_phys, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c
            )
            # Set ghost cell state
            d_U[0, left_ghost_idx] = rho_m_phys # Copy density
            d_U[1, left_ghost_idx] = p_m_phys   # Set w = p
            d_U[2, left_ghost_idx] = rho_c_phys # Copy density
            d_U[3, left_ghost_idx] = p_c_phys   # Set w = p

        # --- Right Boundary ---
        right_ghost_idx = n_phys + n_ghost + i
        if right_type_code == 0: # Inflow
            d_U[0, right_ghost_idx] = inflow_R_0
            d_U[1, right_ghost_idx] = inflow_R_1
            d_U[2, right_ghost_idx] = inflow_R_2
            d_U[3, right_ghost_idx] = inflow_R_3
        elif right_type_code == 1: # Outflow (zero-order extrapolation)
            last_phys_idx = n_phys + n_ghost - 1
            d_U[0, right_ghost_idx] = d_U[0, last_phys_idx]
            d_U[1, right_ghost_idx] = d_U[1, last_phys_idx]
            d_U[2, right_ghost_idx] = d_U[2, last_phys_idx]
            d_U[3, right_ghost_idx] = d_U[3, last_phys_idx]
        elif right_type_code == 2: # Periodic
            src_idx = n_ghost + i # Copy from left physical cells
            d_U[0, right_ghost_idx] = d_U[0, src_idx]
            d_U[1, right_ghost_idx] = d_U[1, src_idx]
            d_U[2, right_ghost_idx] = d_U[2, src_idx]
            d_U[3, right_ghost_idx] = d_U[3, src_idx]
        elif right_type_code == 3: # Wall (Reflection Boundary Condition - Mirroring CPU)
            last_phys_idx = n_phys + n_ghost - 1
            # Get state from last physical cell
            rho_m_phys = d_U[0, last_phys_idx]
            w_m_phys   = d_U[1, last_phys_idx]
            rho_c_phys = d_U[2, last_phys_idx]
            w_c_phys   = d_U[3, last_phys_idx]

            # Calculate pressure at physical cell state (using CUDA helper)
            p_m_phys, p_c_phys = physics._calculate_pressure_cuda(
                rho_m_phys, rho_c_phys, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c
            )

            # Calculate physical velocities (using CUDA helper)
            v_m_phys, v_c_phys = physics._calculate_physical_velocity_cuda(
                w_m_phys, w_c_phys, p_m_phys, p_c_phys
            )

            # Set ghost cell state (reflection)
            # Copy density
            d_U[0, right_ghost_idx] = rho_m_phys
            d_U[2, right_ghost_idx] = rho_c_phys

            # Set ghost velocity to negative of physical velocity
            v_m_ghost = -v_m_phys
            v_c_ghost = -v_c_phys

            # Recalculate momentum density in ghost cell: w = v + P(rho_eff)
            # Need pressure in ghost cell based on copied densities
            p_m_ghost, p_c_ghost = physics._calculate_pressure_cuda(
                rho_m_phys, rho_c_phys, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c
            )

            d_U[1, right_ghost_idx] = v_m_ghost + p_m_ghost
            d_U[3, right_ghost_idx] = v_c_ghost + p_c_ghost
        elif right_type_code == 4: # Wall (Capped Reflection Boundary Condition)
            last_phys_idx = n_phys + n_ghost - 1
            # Get state from last physical cell
            rho_m_phys = d_U[0, last_phys_idx]
            w_m_phys   = d_U[1, last_phys_idx]
            rho_c_phys = d_U[2, last_phys_idx]
            w_c_phys   = d_U[3, last_phys_idx]

            # Calculate pressure at physical cell state (using CUDA helper)
            p_m_phys, p_c_phys = physics._calculate_pressure_cuda(
                rho_m_phys, rho_c_phys, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c
            )

            # Calculate physical velocities (using CUDA helper)
            v_m_phys, v_c_phys = physics._calculate_physical_velocity_cuda(
                w_m_phys, w_c_phys, p_m_phys, p_c_phys
            )

            # --- Capping Logic ---
            rho_cap = rho_jam * rho_cap_factor
            rho_m_ghost_capped = min(rho_m_phys, rho_cap)
            rho_c_ghost_capped = min(rho_c_phys, rho_cap)
            # --- End Capping Logic ---

            # Set ghost cell state (reflection using capped densities for pressure)
            # Copy density (use capped for consistency, although original phys could also be argued)
            d_U[0, right_ghost_idx] = rho_m_ghost_capped
            d_U[2, right_ghost_idx] = rho_c_ghost_capped

            # Set ghost velocity to negative of physical velocity
            v_m_ghost = -v_m_phys
            v_c_ghost = -v_c_phys

            # Recalculate momentum density in ghost cell: w = v + P(rho_eff_capped)
            # Need pressure in ghost cell based on CAPPED densities
            p_m_ghost_capped, p_c_ghost_capped = physics._calculate_pressure_cuda(
                rho_m_ghost_capped, rho_c_ghost_capped, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c
            )

            d_U[1, right_ghost_idx] = v_m_ghost + p_m_ghost_capped
            d_U[3, right_ghost_idx] = v_c_ghost + p_c_ghost_capped

# --- Main Function ---

def apply_boundary_conditions(U_or_d_U, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None, t_current: float = -1.0): # Add t_current
    """
    Applies boundary conditions to the state vector array including ghost cells.
    Works for both CPU (NumPy) and GPU (Numba DeviceNDArray) arrays.
    Includes t_current for debug printing.

    Modifies U_or_d_U in-place.

    Args:
        U_or_d_U (np.ndarray or numba.cuda.DeviceNDArray): State vector array (CPU or GPU). Shape (4, N_total).
        grid (Grid1D): The computational grid object.
        params (ModelParameters): Object containing parameters, including boundary condition definitions
                                  and the target device ('cpu' or 'gpu').
                                  Expects params.boundary_conditions to be a dict like:
                                  {'left': {'type': 'inflow', 'state': [rho_m, w_m, rho_c, w_c]},
                                   'right': {'type': 'outflow'}}
                                  or {'left': {'type': 'periodic'}, 'right': {'type': 'periodic'}}

    Raises:
        ValueError: If an unknown boundary condition type is specified.
    """
    n_ghost = grid.num_ghost_cells
    n_phys = grid.N_physical
    N_total = grid.N_total
    
   # Use current_bc_params if provided, otherwise default to params.boundary_conditions
    bc_config = current_bc_params if current_bc_params is not None else params.boundary_conditions

    # --- Determine Device and Prepare ---
    is_gpu = hasattr(params, 'device') and params.device == 'gpu'
    if is_gpu and not cuda.is_cuda_array(U_or_d_U):
        raise TypeError("Device is 'gpu' but input array is not a Numba CUDA device array.")
    if not is_gpu and cuda.is_cuda_array(U_or_d_U):
         raise TypeError("Device is 'cpu' but input array is a Numba CUDA device array.")

    # --- Get BC Types and Inflow States ---
    left_bc = bc_config.get('left', {'type': 'outflow'})
    right_bc = bc_config.get('right', {'type': 'outflow'})
    left_type_str = left_bc.get('type', 'outflow').lower()
    right_type_str = right_bc.get('type', 'outflow').lower()

    type_map = {'inflow': 0, 'outflow': 1, 'periodic': 2, 'wall': 3, 'wall_capped_reflection': 4} # Added wall types
    left_type_code = type_map.get(left_type_str)
    right_type_code = type_map.get(right_type_str)

    if left_type_code is None: raise ValueError(f"Unknown left boundary condition type: {left_type_str}")
    if right_type_code is None: raise ValueError(f"Unknown right boundary condition type: {right_type_str}")

    # Prepare inflow states (use dummy values if not inflow)
    inflow_L = [0.0] * 4
    inflow_R = [0.0] * 4
    if left_type_code == 0:
        inflow_L_state = left_bc.get('state')
        if inflow_L_state is None or len(inflow_L_state) != 4:
            raise ValueError("Left inflow BC requires a 'state' list/array of length 4.")
        inflow_L = list(inflow_L_state) # Ensure it's a list of floats
    if right_type_code == 0:
        inflow_R_state = right_bc.get('state')
        if inflow_R_state is None or len(inflow_R_state) != 4:
            raise ValueError("Right inflow BC requires a 'state' list/array of length 4.")
        inflow_R = list(inflow_R_state) # Ensure it's a list of floats


    # --- Apply BCs ---
    if is_gpu:
        # --- GPU Implementation ---
        d_U = U_or_d_U # Rename for clarity

        # Configure kernel launch
        threadsperblock = 64 # Can be tuned, but likely small enough
        blockspergrid = math.ceil(n_ghost / threadsperblock)

        # Launch kernel
        # --- DEBUG PRINT: GPU Right Wall (Before Kernel) (Commented out) ---
        # if is_gpu and right_type_code == 3 and t_current < 61.0:
        #     # Copy last physical cell back to host for printing
        #     last_phys_idx = n_phys + n_ghost - 1
        #     last_phys_state_host = d_U[:, last_phys_idx:last_phys_idx+1].copy_to_host()[:, 0]
        #     rho_m_phys_h = last_phys_state_host[0]
        #     w_m_phys_h = last_phys_state_host[1]
        #     rho_c_phys_h = last_phys_state_host[2]
        #     w_c_phys_h = last_phys_state_host[3]
        #     # Calculate pressure and velocity for debug print
        #     p_m_phys_h, p_c_phys_h = physics.calculate_pressure(
        #         rho_m_phys_h, rho_c_phys_h, params.alpha, params.rho_jam, params.epsilon,
        #         params.K_m, params.gamma_m, params.K_c, params.gamma_c
        #     )
        #     v_m_phys_h, v_c_phys_h = physics.calculate_physical_velocity(
        #         w_m_phys_h, w_c_phys_h, p_m_phys_h, p_c_phys_h
        #     )
        #     print(f"DEBUG GPU BC @ t={t_current:.4f} (Right Wall Reflection): BEFORE Kernel - Phys Cell {last_phys_idx}: {last_phys_state_host} (v_m={v_m_phys_h:.4f}, v_c={v_c_phys_h:.4f})")
        #     # Calculate intended ghost state for debug print
        #     v_m_ghost_h = -v_m_phys_h
        #     v_c_ghost_h = -v_c_phys_h
        #     # Need pressure in ghost cell based on copied densities
        #     p_m_ghost_h, p_c_ghost_h = physics.calculate_pressure(
        #         rho_m_phys_h, rho_c_phys_h, params.alpha, params.rho_jam, params.epsilon,
        #         params.K_m, params.gamma_m, params.K_c, params.gamma_c
        #     )
        #     w_m_ghost_h = v_m_ghost_h + p_m_ghost_h
        #     w_c_ghost_h = v_c_ghost_h + p_c_ghost_h
        #     print(f"DEBUG GPU BC @ t={t_current:.4f} (Right Wall Reflection): INTENDED Ghost State: [{rho_m_phys_h:.4f}, {w_m_ghost_h:.4f}, {rho_c_phys_h:.4f}, {w_c_ghost_h:.4f}] (v_m={v_m_ghost_h:.4f}, v_c={v_c_ghost_h:.4f})")
        # # -------------------------------------------------

        _apply_boundary_conditions_kernel[blockspergrid, threadsperblock](
            d_U, n_ghost, n_phys,
            left_type_code, right_type_code,
            # Pass inflow states as individual float arguments
            float64(inflow_L[0]), float64(inflow_L[1]), float64(inflow_L[2]), float64(inflow_L[3]),
            float64(inflow_R[0]), float64(inflow_R[1]), float64(inflow_R[2]), float64(inflow_R[3]),
            # Pass pressure parameters needed by wall BC
            float64(params.alpha), float64(params.rho_jam), float64(params.epsilon),
            float64(params.K_m), float64(params.gamma_m),
            float64(params.K_c), float64(params.gamma_c),
            # Pass rho_cap_factor (even if not used by all BCs yet)
            float64(params.rho_cap_factor if hasattr(params, 'rho_cap_factor') else 0.99) # Default if not set
        )
        # No explicit sync needed here, subsequent kernels will sync

    else:
        # --- CPU Implementation (Original Logic) ---
        U = U_or_d_U # Rename for clarity

        # Left Boundary
        if left_type_code == 0: # Inflow (Modified: Impose full state [rho_m, w_m, rho_c, w_c])
            first_physical_cell_state = U[:, n_ghost:n_ghost+1]
            U[0, 0:n_ghost] = inflow_L[0] # Impose rho_m
            U[1, 0:n_ghost] = inflow_L[1] # Impose w_m (FIXED: was extrapolated)
            U[2, 0:n_ghost] = inflow_L[2] # Impose rho_c
            U[3, 0:n_ghost] = inflow_L[3] # Impose w_c (FIXED: was extrapolated)
        elif left_type_code == 1: # Outflow
            first_physical_cell_state = U[:, n_ghost:n_ghost+1]
            U[:, 0:n_ghost] = first_physical_cell_state
        elif left_type_code == 2: # Periodic
            U[:, 0:n_ghost] = U[:, n_phys:n_phys + n_ghost]
        elif left_type_code == 3: # Wall (v=0 -> w=p)
            first_physical_cell_state = U[:, n_ghost] # Shape (4,)
            rho_m_phys = first_physical_cell_state[0]
            rho_c_phys = first_physical_cell_state[2]
            # Calculate pressure (CPU version)
            p_m_phys, p_c_phys = physics.calculate_pressure(
                rho_m_phys, rho_c_phys, params.alpha, params.rho_jam, params.epsilon,
                params.K_m, params.gamma_m, params.K_c, params.gamma_c
            )
            # Set ghost cells
            U[0, 0:n_ghost] = rho_m_phys
            U[1, 0:n_ghost] = p_m_phys
            U[2, 0:n_ghost] = rho_c_phys
            U[3, 0:n_ghost] = p_c_phys

        # Right Boundary
        if right_type_code == 0: # Inflow
            U[:, n_phys + n_ghost:] = np.array(inflow_R).reshape(-1, 1)
        elif right_type_code == 1: # Outflow
            last_physical_cell_state = U[:, n_phys + n_ghost - 1 : n_phys + n_ghost]
            U[:, n_phys + n_ghost:] = last_physical_cell_state
        elif right_type_code == 2: # Periodic
            U[:, n_phys + n_ghost:] = U[:, n_ghost:n_ghost + n_ghost]
        elif right_type_code == 3: # Wall (Reflection Boundary Condition)
            last_physical_cell_state = U[:, n_phys + n_ghost - 1] # Shape (4,)
            rho_m_phys = last_physical_cell_state[0]
            w_m_phys   = last_physical_cell_state[1]
            rho_c_phys = last_physical_cell_state[2]
            w_c_phys   = last_physical_cell_state[3]

            # Calculate pressure (CPU version)
            p_m_phys, p_c_phys = physics.calculate_pressure(
                rho_m_phys, rho_c_phys, params.alpha, params.rho_jam, params.epsilon,
                params.K_m, params.gamma_m, params.K_c, params.gamma_c
            )

            # Calculate physical velocities (CPU version)
            v_m_phys, v_c_phys = physics.calculate_physical_velocity(
                w_m_phys, w_c_phys, p_m_phys, p_c_phys
            )

            # Set ghost cell state (reflection)
            U[0, n_phys + n_ghost:] = rho_m_phys # Copy density
            U[2, n_phys + n_ghost:] = rho_c_phys # Copy density

            # Set ghost velocity to negative of physical velocity
            v_m_ghost = -v_m_phys
            v_c_ghost = -v_c_phys

            # Recalculate momentum density in ghost cell: w = v + P(rho_eff)
            # Need pressure in ghost cell based on copied densities
            p_m_ghost, p_c_ghost = physics.calculate_pressure(
                rho_m_phys, rho_c_phys, params.alpha, params.rho_jam, params.epsilon,
                params.K_m, params.gamma_m, params.K_c, params.gamma_c
            )

            U[1, n_phys + n_ghost:] = v_m_ghost + p_m_ghost # Note: Using p_m_ghost based on copied densities
            U[3, n_phys + n_ghost:] = v_c_ghost + p_c_ghost # Note: Using p_c_ghost based on copied densities

        elif right_type_code == 4: # Wall (Capped Reflection Boundary Condition)
            last_physical_cell_state = U[:, n_phys + n_ghost - 1] # Shape (4,)
            rho_m_phys = last_physical_cell_state[0]
            w_m_phys   = last_physical_cell_state[1]
            rho_c_phys = last_physical_cell_state[2]
            w_c_phys   = last_physical_cell_state[3]

            # Calculate pressure (CPU version)
            p_m_phys, p_c_phys = physics.calculate_pressure(
                rho_m_phys, rho_c_phys, params.alpha, params.rho_jam, params.epsilon,
                params.K_m, params.gamma_m, params.K_c, params.gamma_c
            )

            # Calculate physical velocities (CPU version)
            v_m_phys, v_c_phys = physics.calculate_physical_velocity(
                w_m_phys, w_c_phys, p_m_phys, p_c_phys
            )

            # --- Capping Logic ---
            rho_cap_factor = getattr(params, 'rho_cap_factor', 0.99) # Get factor or default
            rho_cap = params.rho_jam * rho_cap_factor
            rho_m_ghost_capped = min(rho_m_phys, rho_cap)
            rho_c_ghost_capped = min(rho_c_phys, rho_cap)
            # --- End Capping Logic ---

            # Set ghost cell state (reflection using capped densities for pressure)
            # Copy density (use capped for consistency)
            U[0, n_phys + n_ghost:] = rho_m_ghost_capped
            U[2, n_phys + n_ghost:] = rho_c_ghost_capped

            # Set ghost velocity to negative of physical velocity
            v_m_ghost = -v_m_phys
            v_c_ghost = -v_c_phys

            # Recalculate momentum density in ghost cell: w = v + P(rho_eff_capped)
            # Need pressure in ghost cell based on CAPPED densities
            p_m_ghost_capped, p_c_ghost_capped = physics.calculate_pressure(
                rho_m_ghost_capped, rho_c_ghost_capped, params.alpha, params.rho_jam, params.epsilon,
                params.K_m, params.gamma_m, params.K_c, params.gamma_c
            )

            U[1, n_phys + n_ghost:] = v_m_ghost + p_m_ghost_capped
            U[3, n_phys + n_ghost:] = v_c_ghost + p_c_ghost_capped

    # Note: No return value, U_or_d_U is modified in-place.


def apply_boundary_conditions_gpu(d_U, grid, params):
    """
    Apply boundary conditions on GPU device array.
    
    Args:
        d_U: Device array (4, N_total) to modify in-place
        grid: Grid1D object with boundary info
        params: ModelParameters with boundary condition settings
    """
    n_ghost = grid.num_ghost_cells
    n_phys = grid.N_physical
    
    # Encode boundary condition types
    bc_type_map = {'inflow': 0, 'outflow': 1, 'periodic': 2, 'wall': 3, 'wall_capped': 4}
    
    left_type_code = bc_type_map.get(params.boundary_conditions['left']['type'], 1)  # Default to outflow
    right_type_code = bc_type_map.get(params.boundary_conditions['right']['type'], 1)  # Default to outflow
    
    # Get inflow values (use zeros if not inflow)
    left_bc = params.boundary_conditions['left']
    right_bc = params.boundary_conditions['right']
    
    if left_type_code == 0 and 'state' in left_bc:
        inflow_L = left_bc['state']
        # Handle both list [rho_m, w_m, rho_c, w_c] and dict {'rho_m': ..., 'w_m': ..., 'rho_c': ..., 'w_c': ...}
        if isinstance(inflow_L, dict):
            inflow_L_0, inflow_L_1, inflow_L_2, inflow_L_3 = inflow_L['rho_m'], inflow_L['w_m'], inflow_L['rho_c'], inflow_L['w_c']
        else: # list or tuple
            inflow_L_0, inflow_L_1, inflow_L_2, inflow_L_3 = inflow_L[0], inflow_L[1], inflow_L[2], inflow_L[3]
    else:
        inflow_L_0, inflow_L_1, inflow_L_2, inflow_L_3 = 0.0, 0.0, 0.0, 0.0
    
    if right_type_code == 0 and 'state' in right_bc:
        inflow_R = right_bc['state']
        # Handle both list [rho_m, w_m, rho_c, w_c] and dict {'rho_m': ..., 'w_m': ..., 'rho_c': ..., 'w_c': ...}
        if isinstance(inflow_R, dict):
            inflow_R_0, inflow_R_1, inflow_R_2, inflow_R_3 = inflow_R['rho_m'], inflow_R['w_m'], inflow_R['rho_c'], inflow_R['w_c']
        else: # list or tuple
            inflow_R_0, inflow_R_1, inflow_R_2, inflow_R_3 = inflow_R[0], inflow_R[1], inflow_R[2], inflow_R[3]
    else:
        inflow_R_0, inflow_R_1, inflow_R_2, inflow_R_3 = 0.0, 0.0, 0.0, 0.0
    
    # Physics parameters for wall BC
    alpha = params.alpha
    rho_jam = params.rho_jam
    epsilon = params.epsilon
    K_m = params.K_m
    gamma_m = params.gamma_m
    K_c = params.K_c
    gamma_c = params.gamma_c
    rho_cap_factor = getattr(params, 'rho_cap_factor', 0.99)
    
    # Launch kernel
    threads_per_block = min(256, n_ghost)
    blocks_per_grid = max(1, math.ceil(n_ghost / threads_per_block))
    
    _apply_boundary_conditions_kernel[blocks_per_grid, threads_per_block](
        d_U, n_ghost, n_phys,
        left_type_code, right_type_code,
        inflow_L_0, inflow_L_1, inflow_L_2, inflow_L_3,
        inflow_R_0, inflow_R_1, inflow_R_2, inflow_R_3,
        alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c,
        rho_cap_factor
    )
