"""
This module defines functions for creating initial conditions for the simulation.
"""
import numpy as np

from arz_model.core.parameters import ModelParameters
from arz_model.grid.grid1d import Grid1D


def uniform_initial_condition(grid: Grid1D, params: ModelParameters, rho: float, v: float, p: float):
    """
    Creates a uniform initial condition.
    """
    U = np.zeros((4, grid.N_total))
    U[0, :] = rho
    U[1, :] = rho * v
    U[2, :] = 0.0  # No heavy vehicles initially
    U[3, :] = p
    return U

def riemann_problem(grid: Grid1D, params: ModelParameters, rho_l: float, v_l: float, p_l: float, rho_r: float, v_r: float, p_r: float):
    """
    Creates a Riemann problem initial condition.
    """
    U = np.zeros((4, grid.N_total))
    midpoint = grid.x.size // 2
    U[0, :midpoint] = rho_l
    U[1, :midpoint] = rho_l * v_l
    U[2, :midpoint] = 0.0
    U[3, :midpoint] = p_l
    U[0, midpoint:] = rho_r
    U[1, midpoint:] = rho_r * v_r
    U[2, midpoint:] = 0.0
    U[3, midpoint:] = p_r
    return U
