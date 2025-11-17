import numpy as np
from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters
from ..core import physics
from typing import Optional

def apply_boundary_conditions(U: np.ndarray, grid: Grid1D, params: ModelParameters, current_bc_params: Optional[dict] = None):
    """
    Applies boundary conditions to the ghost cells of the state vector U.

    This function handles different types of boundary conditions, such as
    inflow, outflow, and periodic.

    Args:
        U (np.ndarray): State array (4, N_total) including ghost cells.
        grid (Grid1D): The grid object.
        params (ModelParameters): The model parameters.
        current_bc_params (dict, optional): Dynamically updated BC parameters,
                                             e.g., for time-varying inflow.
    """
    g = grid.num_ghost_cells
    
    # Determine which BC configuration to use
    bc_config = params.boundary_conditions
    if current_bc_params:
        # Dynamic params override static ones if provided
        # This is useful for network simulations where inflow can change
        left_bc_config = current_bc_params.get('left', bc_config.left)
        right_bc_config = current_bc_params.get('right', bc_config.right)
    else:
        left_bc_config = bc_config.left
        right_bc_config = bc_config.right

    # --- Left Boundary Condition ---
    if left_bc_config.type == 'inflow':
        # Prescribe fixed state at the inflow boundary
        state = left_bc_config.state
        U[0, :g] = state.rho_m
        U[1, :g] = state.w_m
        U[2, :g] = state.rho_c
        U[3, :g] = state.w_c
    elif left_bc_config.type == 'outflow':
        # Zero-gradient (extrapolation) for outflow
        for i in range(g):
            U[:, i] = U[:, g]
            
    # --- Right Boundary Condition ---
    if right_bc_config.type == 'inflow':
         # This is unusual but supported
        state = right_bc_config.state
        U[0, -g:] = state.rho_m
        U[1, -g:] = state.w_m
        U[2, -g:] = state.rho_c
        U[3, -g:] = state.w_c
    elif right_bc_config.type == 'outflow':
        # Zero-gradient (extrapolation) for outflow
        for i in range(1, g + 1):
            U[:, -i] = U[:, -(g + 1)]