"""
GPU-Native Node Solver for ARZ Model

This module provides a pure GPU implementation of network junction handling
that eliminates CPU↔GPU transfers in the network coupling step.

Academic References:
- Garavello & Piccoli (2005): "Traffic flow on a road network."
- Daganzo (1995): "The cell transmission model, part II: Network traffic."
"""

import math
from numba import cuda
import numba as nb
import numpy as np
from typing import List, Dict, Tuple, Optional
from arz_model.core.parameters import ModelParameters

from ..core.physics import (
    _calculate_demand_flux_cuda, 
    _calculate_supply_flux_cuda,
    _invert_flux_function_cuda,
    calculate_equilibrium_speed_gpu
)

# ============================================================================
# CUDA KERNELS FOR NODE SOLVING
# ============================================================================

@cuda.jit(device=True)
def solve_node_fluxes_gpu(
    U_L_m, U_L_c, num_incoming, num_outgoing,
    alpha, rho_max, epsilon, k_m, gamma_m, k_c, gamma_c,
    v_max_m, v_max_c, v_creeping
):
    """
    GPU device function to solve for the equilibrium state at a single node
    using a demand-supply model based on Daganzo (1995).

    This solver determines the total demand from all incoming links and the
    total supply from all outgoing links, then calculates the actual flow
    that can be accommodated. This flow is then distributed among the
    outgoing links.

    Args:
        U_L_m: Incoming states for motorcycles [rho_m, w_m]
        U_L_c: Incoming states for cars [rho_c, w_c]
        num_incoming: Number of incoming segments
        num_outgoing: Number of outgoing segments
        (physics_params): All 10 physics parameters required.

    Returns:
        U_star_m: The resolved state [rho_m, w_m] to be applied to outgoing ghost cells.
        U_star_c: The resolved state [rho_c, w_c] to be applied to outgoing ghost cells.
    """
    # --- 1. Calculate Total Demand from Incoming Links ---
    total_demand_m = 0.0
    total_demand_c = 0.0
    if num_incoming > 0:
        for i in range(num_incoming):
            # Ensure all 7 physics params are passed to demand calculation
            demand_m, demand_c = _calculate_demand_flux_cuda(
                U_L_m[i, 0], U_L_m[i, 1], U_L_c[i, 0], U_L_c[i, 1],
                alpha, rho_max, epsilon, k_m, gamma_m, k_c, gamma_c
            )
            total_demand_m += demand_m
            total_demand_c += demand_c

    # --- 2. Calculate Total Supply from Outgoing Links ---
    total_supply_m = 0.0
    total_supply_c = 0.0
    if num_outgoing > 0:
        # Ensure all required params are passed to supply calculation
        supply_m_per_link, supply_c_per_link = _calculate_supply_flux_cuda(
            rho_max, k_m, gamma_m, k_c, gamma_c
        )
        total_supply_m = num_outgoing * supply_m_per_link
        total_supply_c = num_outgoing * supply_c_per_link

    # --- 3. Determine Actual Flux through the Junction ---
    actual_flux_m = min(total_demand_m, total_supply_m)
    actual_flux_c = min(total_demand_c, total_supply_c)
    
    # --- 4. Distribute Flux to Outgoing Links ---
    flux_per_outgoing_m = 0.0
    flux_per_outgoing_c = 0.0
    if num_outgoing > 0:
        flux_per_outgoing_m = actual_flux_m / num_outgoing
        flux_per_outgoing_c = actual_flux_c / num_outgoing

    # --- 5. Determine the State U* for Ghost Cells of Outgoing Links ---
    # Invert the flux function to get the density for the outgoing ghost cells.
    # This requires Vmax, which is now passed directly.
    rho_star_m = _invert_flux_function_cuda(flux_per_outgoing_m, rho_max, v_max_m, epsilon)
    rho_star_c = _invert_flux_function_cuda(flux_per_outgoing_c, rho_max, v_max_c, epsilon)

    # Calculate the corresponding equilibrium speed for this new density.
    # We assume a placeholder road quality category for outgoing links.
    R_local = 3 
    Ve_star_m, Ve_star_c = calculate_equilibrium_speed_gpu(
        rho_star_m, rho_star_c, R_local,
        rho_max, v_creeping,
        v_max_m, v_max_m, v_max_m, # Placeholder v_max for categories 1, 2, 3
        v_max_c, v_max_c, v_max_c  # Placeholder v_max for categories 1, 2, 3
    )

    # Simplification: Assume momentum w_star is approximately the equilibrium speed Ve_star.
    # This is a reasonable approximation for free-flow conditions.
    w_star_m = Ve_star_m
    w_star_c = Ve_star_c

    # Create the final state vector U_star
    U_star_m_out = cuda.local.array(2, dtype=nb.float64)
    U_star_c_out = cuda.local.array(2, dtype=nb.float64)
    
    U_star_m_out[0] = rho_star_m
    U_star_m_out[1] = w_star_m
    U_star_c_out[0] = rho_star_c
    U_star_c_out[1] = w_star_c

    return U_star_m_out, U_star_c_out


# The dead code including solve_node_fluxes_kernel, 
# apply_network_coupling_gpu_native, and create_gpu_node_solver_for_network
# has been removed to prevent Numba compilation errors.