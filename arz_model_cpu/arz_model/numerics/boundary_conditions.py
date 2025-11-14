import numpy as np
from numba import cuda, float64, int32 # Import cuda and types
import math # For ceil
from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters
from ..core import physics # Import the physics module for pressure calculation
from ..config.debug_config import DEBUG_LOGS_ENABLED

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
    
    # In network mode, BCs are handled by NetworkGrid, not globally
    if hasattr(params, 'is_network_mode') and params.is_network_mode and current_bc_params is None:
        if DEBUG_LOGS_ENABLED:
            logger.debug("Skipping global BC apply in network mode without specific segment BCs.")
        return

    # Determine which BC parameters to use
    bc_params_to_use = current_bc_params if current_bc_params is not None else getattr(params, 'boundary_conditions', None)
    
    # Fallback if still no BCs
    if bc_params_to_use is None:
        if DEBUG_LOGS_ENABLED and t_current != -1.0: # Avoid logging during initial setup
            # Use a local logger to avoid circular import issues if any
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Time {t_current:.2f}s: No boundary conditions provided. Skipping BC application.")
        return
