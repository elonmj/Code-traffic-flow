"""
Initial Conditions Builder

Extracts IC creation logic from runner.py into a clean, testable class.
Converts Pydantic IC configs into numpy state arrays.
"""

import numpy as np
from typing import Tuple
from ...config import InitialConditionsConfig
from ...grid.grid1d import Grid1D
from ...core.parameters import ModelParameters
from .. import initial_conditions  # Import the initial conditions module


# Constants
VEH_KM_TO_VEH_M = 1.0 / 1000.0  # Convert veh/km to veh/m


class ICBuilder:
    """Builds initial state arrays from Pydantic IC configs."""
    
    @staticmethod
    def build(ic_config: InitialConditionsConfig, 
              grid: Grid1D, 
              params: ModelParameters,
              quiet: bool = False) -> np.ndarray:
        """
        Creates the initial state array U based on IC config.
        
        Args:
            ic_config: Pydantic initial conditions configuration
            grid: Grid1D object defining the spatial discretization
            params: Model parameters (for physics calculations)
            quiet: If True, suppress informational messages
            
        Returns:
            U_init: Initial state array [4, N_total] with ghost cells
            
        Raises:
            ValueError: If IC config is invalid or incomplete
        """
        ic_type = ic_config.type.lower() if hasattr(ic_config, 'type') else None
        
        if not ic_type:
            raise ValueError("Initial conditions config must have a 'type' field.")
        
        # === UNIFORM IC ===
        if ic_type == 'uniform':
            state_vals = [ic_config.rho_m, ic_config.w_m, ic_config.rho_c, ic_config.w_c]
            U_init = initial_conditions.uniform_state(grid, *state_vals)
            
            if not quiet:
                print(f"  ✅ IC: Uniform state = {state_vals}")
        
        # === UNIFORM EQUILIBRIUM IC ===
        elif ic_type == 'uniform_equilibrium':
            rho_m = ic_config.rho_m
            rho_c = ic_config.rho_c
            R_val = ic_config.R_val
            
            # Convert densities from veh/km (config) to veh/m (SI units)
            rho_m_si = rho_m * VEH_KM_TO_VEH_M
            rho_c_si = rho_c * VEH_KM_TO_VEH_M
            
            # Compute equilibrium state
            U_init, eq_state_vector = initial_conditions.uniform_state_from_equilibrium(
                grid, rho_m_si, rho_c_si, R_val, params
            )
            
            if not quiet:
                print(f"  ✅ IC: Uniform equilibrium ρ_m={rho_m}, ρ_c={rho_c}, R={R_val}")
                print(f"      Equilibrium velocities: {eq_state_vector[1]:.2f}, {eq_state_vector[3]:.2f} m/s")
        
        # === RIEMANN PROBLEM IC ===
        elif ic_type == 'riemann':
            U_L = [ic_config.rho_m_left, ic_config.w_m_left, 
                   ic_config.rho_c_left, ic_config.w_c_left]
            U_R = [ic_config.rho_m_right, ic_config.w_m_right,
                   ic_config.rho_c_right, ic_config.w_c_right]
            split_pos = ic_config.x_discontinuity
            
            U_init = initial_conditions.riemann_problem(grid, U_L, U_R, split_pos)
            
            if not quiet:
                print(f"  ✅ IC: Riemann problem at x={split_pos} km")
                print(f"      Left: {U_L}, Right: {U_R}")
        
        # === GAUSSIAN PULSE IC ===
        elif ic_type == 'gaussian_pulse':
            bg_state = [ic_config.background_rho_m, 0.0, ic_config.background_rho_c, 0.0]
            center = ic_config.x_center
            sigma = ic_config.sigma
            amplitude = ic_config.amplitude
            
            # Create density hump (Gaussian pulse)
            # Note: Using density_hump which is similar to Gaussian
            rho_m_max = ic_config.background_rho_m + amplitude
            rho_c_max = ic_config.background_rho_c + amplitude
            
            U_init = initial_conditions.density_hump(
                grid, *bg_state, center, sigma, rho_m_max, rho_c_max
            )
            
            if not quiet:
                print(f"  ✅ IC: Gaussian pulse at x={center} km, σ={sigma} km")
        
        # === FILE-BASED IC ===
        elif ic_type == 'from_file':
            filepath = ic_config.filepath
            
            # Load from file
            if filepath.endswith('.npy'):
                U_init = np.load(filepath)
            elif filepath.endswith('.txt'):
                U_init = np.loadtxt(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}. Use .npy or .txt")
            
            # Validate shape
            expected_shape = (4, grid.N + 2 * grid.ghost_cells)
            if U_init.shape != expected_shape:
                raise ValueError(
                    f"Loaded IC has shape {U_init.shape}, expected {expected_shape}"
                )
            
            if not quiet:
                print(f"  ✅ IC: Loaded from file {filepath}")
        
        else:
            raise ValueError(f"Unknown initial condition type: '{ic_type}'")
        
        # Return the raw initial state without BCs applied yet
        return U_init
    
    @staticmethod
    def build_from_legacy_dict(ic_dict: dict, 
                               grid: Grid1D, 
                               params: ModelParameters,
                               quiet: bool = False) -> np.ndarray:
        """
        Legacy compatibility: Build IC from old dict-based config.
        
        This method supports the old YAML-based IC format during transition period.
        
        Args:
            ic_dict: Dictionary with IC configuration (old format)
            grid: Grid1D object
            params: Model parameters
            quiet: Suppress messages
            
        Returns:
            U_init: Initial state array
        """
        ic_type = ic_dict.get('type', '').lower()
        
        if ic_type == 'uniform':
            state_vals = ic_dict.get('state')
            if state_vals is None or len(state_vals) != 4:
                raise ValueError("Uniform IC requires 'state': [rho_m, w_m, rho_c, w_c]")
            U_init = initial_conditions.uniform_state(grid, *state_vals)
        
        elif ic_type == 'uniform_equilibrium':
            rho_m = ic_dict.get('rho_m')
            rho_c = ic_dict.get('rho_c')
            R_val = ic_dict.get('R_val')
            
            if rho_m is None or rho_c is None or R_val is None:
                raise ValueError("Uniform Equilibrium IC requires 'rho_m', 'rho_c', 'R_val'.")
            
            # Convert densities from veh/km to veh/m
            rho_m_si = rho_m * VEH_KM_TO_VEH_M
            rho_c_si = rho_c * VEH_KM_TO_VEH_M
            
            U_init, _ = initial_conditions.uniform_state_from_equilibrium(
                grid, rho_m_si, rho_c_si, R_val, params
            )
        
        elif ic_type == 'riemann':
            U_L = ic_dict.get('U_L')
            U_R = ic_dict.get('U_R')
            split_pos = ic_dict.get('split_pos')
            
            if U_L is None or U_R is None or split_pos is None:
                raise ValueError("Riemann IC requires 'U_L', 'U_R', 'split_pos'.")
            
            U_init = initial_conditions.riemann_problem(grid, U_L, U_R, split_pos)
        
        elif ic_type == 'density_hump':
            bg_state = ic_dict.get('background_state')
            center = ic_dict.get('center')
            width = ic_dict.get('width')
            rho_m_max = ic_dict.get('rho_m_max')
            rho_c_max = ic_dict.get('rho_c_max')
            
            if None in [bg_state, center, width, rho_m_max, rho_c_max] or len(bg_state) != 4:
                raise ValueError(
                    "Density Hump IC requires 'background_state' [rho_m, w_m, rho_c, w_c], "
                    "'center', 'width', 'rho_m_max', 'rho_c_max'."
                )
            
            U_init = initial_conditions.density_hump(
                grid, *bg_state, center, width, rho_m_max, rho_c_max
            )
        
        elif ic_type == 'sine_wave_perturbation':
            bg_state_config = ic_dict.get('background_state', {})
            perturbation_config = ic_dict.get('perturbation', {})
            
            rho_m_bg = bg_state_config.get('rho_m')
            rho_c_bg = bg_state_config.get('rho_c')
            epsilon_rho_m = perturbation_config.get('amplitude')
            wave_number = perturbation_config.get('wave_number')
            
            R_val = ic_dict.get(
                'R_val',
                getattr(params, 'road_quality_definition', None) 
                if isinstance(getattr(params, 'road_quality_definition', None), int) 
                else None
            )
            
            if None in [rho_m_bg, rho_c_bg, epsilon_rho_m, wave_number, R_val]:
                raise ValueError(
                    "Sine Wave Perturbation IC requires nested 'background_state' "
                    "(with 'rho_m', 'rho_c'), 'perturbation' (with 'amplitude', 'wave_number'), "
                    "and 'R_val'."
                )
            
            U_init = initial_conditions.sine_wave_perturbation(
                grid, params, rho_m_bg, rho_c_bg, R_val, epsilon_rho_m, wave_number
            )
        
        else:
            raise ValueError(f"Unknown initial condition type: '{ic_type}'")
        
        return U_init
