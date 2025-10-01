import numpy as np

# Assuming modules are importable from the parent directory
try:
    from ..grid.grid1d import Grid1D
    from ..core.parameters import ModelParameters
except ImportError:
    # Fallback for direct execution or testing
    print("Warning: Could not perform relative imports in metrics.py. Assuming modules are in sys.path.")
    # You might need to adjust sys.path if running this file directly for testing
    pass


def calculate_total_mass(state_physical: np.ndarray, grid: Grid1D, class_index: int) -> float:
    """
    Calculates the total mass (total number of vehicles) for a specific class
    within the physical domain.

    Args:
        state_physical (np.ndarray): State array for physical cells only. Shape (4, N_physical).
        grid (Grid1D): The grid object.
        class_index (int): Index of the density variable for the class (0 for motorcycles, 2 for cars).

    Returns:
        float: The total number of vehicles for the specified class.

    Raises:
        ValueError: If class_index is not 0 or 2, or if state_physical shape is incorrect.
    """
    if class_index not in [0, 2]:
        raise ValueError("class_index must be 0 (motorcycles) or 2 (cars).")
    if state_physical.shape[1] != grid.N_physical or state_physical.shape[0] != 4:
         raise ValueError(f"State array shape {state_physical.shape} does not match expected (4, {grid.N_physical}).")

    # Density is the 0th row for motorcycles, 2nd row for cars
    density = state_physical[class_index, :]

    # Total mass is the sum of density * cell width over all physical cells
    total_mass = np.sum(density * grid.dx)

    return total_mass

# Add other analysis metrics here as needed, e.g.:
# calculate_average_velocity(state_physical, grid, class_index, params)
# calculate_flow_rate(state_physical, grid, interface_index, params) # Requires flux calculation

# Example Usage (for testing purposes)
# if __name__ == '__main__':
#     # Setup dummy grid
#     N_phys = 10
#     n_ghost = 2
#     dummy_grid = Grid1D(N=N_phys, xmin=0, xmax=100, num_ghost_cells=n_ghost)
#
#     # Create a dummy state (physical cells only)
#     # Example: uniform density
#     rho_m_uniform = 50.0 / 1000.0 # veh/m
#     rho_c_uniform = 25.0 / 1000.0 # veh/m
#     U_phys_uniform = np.zeros((4, N_phys))
#     U_phys_uniform[0, :] = rho_m_uniform
#     U_phys_uniform[2, :] = rho_c_uniform
#     # Fill w with dummy values (not needed for mass calculation)
#     U_phys_uniform[1, :] = 10.0
#     U_phys_uniform[3, :] = 8.0
#
#     # Example: varying density
#     rho_m_varying = np.linspace(10/1000, 100/1000, N_phys)
#     rho_c_varying = np.linspace(5/1000, 50/1000, N_phys)
#     U_phys_varying = np.zeros((4, N_phys))
#     U_phys_varying[0, :] = rho_m_varying
#     U_phys_varying[2, :] = rho_c_varying
#     U_phys_varying[1, :] = 10.0
#     U_phys_varying[3, :] = 8.0
#
#     # --- Test calculate_total_mass ---
#     print("--- Testing calculate_total_mass ---")
#
#     # Uniform case
#     mass_m_uniform = calculate_total_mass(U_phys_uniform, dummy_grid, 0)
#     mass_c_uniform = calculate_total_mass(U_phys_uniform, dummy_grid, 2)
#     expected_mass_m_uniform = rho_m_uniform * dummy_grid.N_physical * dummy_grid.dx
#     expected_mass_c_uniform = rho_c_uniform * dummy_grid.N_physical * dummy_grid.dx
#     print(f"Uniform Mass (Motos): Calculated={mass_m_uniform:.4f}, Expected={expected_mass_m_uniform:.4f}")
#     print(f"Uniform Mass (Cars): Calculated={mass_c_uniform:.4f}, Expected={expected_mass_c_uniform:.4f}")
#     assert np.isclose(mass_m_uniform, expected_mass_m_uniform)
#     assert np.isclose(mass_c_uniform, expected_mass_c_uniform)
#
#     # Varying case
#     mass_m_varying = calculate_total_mass(U_phys_varying, dummy_grid, 0)
#     mass_c_varying = calculate_total_mass(U_phys_varying, dummy_grid, 2)
#     # For varying density, the sum is the integral approximation
#     expected_mass_m_varying = np.sum(rho_m_varying * dummy_grid.dx)
#     expected_mass_c_varying = np.sum(rho_c_varying * dummy_grid.dx)
#     print(f"Varying Mass (Motos): Calculated={mass_m_varying:.4f}, Expected={expected_mass_m_varying:.4f}")
#     print(f"Varying Mass (Cars): Calculated={mass_c_varying:.4f}, Expected={expected_mass_c_varying:.4f}")
#     assert np.isclose(mass_m_varying, expected_mass_m_varying)
#     assert np.isclose(mass_c_varying, expected_mass_c_varying)
#
#     # Test invalid class_index
#     try:
#         calculate_total_mass(U_phys_uniform, dummy_grid, 1)
#     except ValueError as e:
#         print(f"Caught expected error for invalid class_index: {e}")
#     except Exception as e:
#         print(f"Caught unexpected error for invalid class_index: {e}")
#
#     print("calculate_total_mass tests completed.")

def compute_mape(observed, simulated):
    """
    Calcule l'erreur relative moyenne absolue (MAPE)
    
    Args:
        observed (np.ndarray): Valeurs observées/référence
        simulated (np.ndarray): Valeurs simulées
    
    Returns:
        float: MAPE en pourcentage
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)
    mask = observed != 0
    if not np.any(mask):
        return np.inf
    return np.mean(np.abs((observed[mask] - simulated[mask]) / observed[mask])) * 100

def compute_rmse(observed, simulated):
    """
    Calcule l'erreur quadratique moyenne (RMSE)
    
    Args:
        observed (np.ndarray): Valeurs observées/référence
        simulated (np.ndarray): Valeurs simulées
    
    Returns:
        float: RMSE
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)
    return np.sqrt(np.mean((observed - simulated)**2))

def compute_geh(observed, simulated):
    """
    Calcule la statistique GEH pour les flux de trafic
    
    Args:
        observed (np.ndarray): Flux observés
        simulated (np.ndarray): Flux simulés
    
    Returns:
        np.ndarray: Statistiques GEH pour chaque point
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)
    # Éviter division par zéro
    denominator = observed + simulated
    mask = denominator > 0
    geh = np.zeros_like(observed)
    geh[mask] = np.sqrt(2 * (observed[mask] - simulated[mask])**2 / denominator[mask])
    return geh

def compute_theil_u(observed, simulated):
    """
    Calcule le coefficient de Theil U
    
    Args:
        observed (np.ndarray): Valeurs observées
        simulated (np.ndarray): Valeurs simulées
    
    Returns:
        float: Coefficient de Theil U
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)
    numerator = np.sqrt(np.mean((observed - simulated)**2))
    denominator = np.sqrt(np.mean(observed**2)) + np.sqrt(np.mean(simulated**2))
    return numerator / denominator if denominator > 0 else np.inf

def calculate_convergence_order(grid_sizes, errors):
    """
    Calcule l'ordre de convergence numérique
    
    Args:
        grid_sizes (list): Tailles de grille (N)
        errors (list): Erreurs correspondantes (L2, etc.)
    
    Returns:
        list: Ordres de convergence entre grilles consécutives
    """
    grid_sizes = np.asarray(grid_sizes)
    errors = np.asarray(errors)
    
    if len(grid_sizes) < 2:
        return []
    
    orders = []
    for i in range(1, len(grid_sizes)):
        h1 = 1.0 / grid_sizes[i-1]  # dx = L/N
        h2 = 1.0 / grid_sizes[i]
        e1 = errors[i-1]
        e2 = errors[i]
        
        if e1 > 0 and e2 > 0 and h1 != h2:
            order = np.log(e1/e2) / np.log(h1/h2)
            orders.append(order)
        else:
            orders.append(np.nan)
    
    return orders

def analytical_riemann_solution(x, t, rho_left, v_left, rho_right, v_right, params):
    """
    Solution analytique pour problème de Riemann ARZ simplifié
    
    Args:
        x (np.ndarray): Positions
        t (float): Temps
        rho_left, v_left: État gauche (densité, vitesse)
        rho_right, v_right: État droit (densité, vitesse)
        params: Paramètres du modèle (V0, tau, etc.)
    
    Returns:
        tuple: (rho(x,t), v(x,t)) solutions analytiques
    """
    # Solution simplifiée pour cas particuliers (onde de raréfaction/choc)
    # Implémentation basique - à raffiner selon les cas
    
    x = np.asarray(x)
    rho = np.zeros_like(x)
    v = np.zeros_like(x)
    
    # Position de discontinuité initiale
    x0 = 0.0
    
    # Vitesses caractéristiques approximatives
    c_left = v_left + rho_left * params.V0 / params.tau  # Approximation
    c_right = v_right + rho_right * params.V0 / params.tau
    
    # Position du front à l'instant t
    front_pos = x0 + 0.5 * (c_left + c_right) * t
    
    # Assignation simple gauche/droite
    mask_left = x < front_pos
    rho[mask_left] = rho_left
    v[mask_left] = v_left
    rho[~mask_left] = rho_right
    v[~mask_left] = v_right
    
    return rho, v

def analytical_equilibrium_profile(x, params, rho0=0.1, perturbation_amplitude=0.05):
    """
    Profil d'équilibre analytique pour validation
    
    Args:
        x (np.ndarray): Positions
        params: Paramètres du modèle
        rho0: Densité d'équilibre de base
        perturbation_amplitude: Amplitude des perturbations
    
    Returns:
        tuple: (rho_eq(x), v_eq(x)) profils d'équilibre
    """
    x = np.asarray(x)
    
    # Profil de densité avec petites variations
    rho_eq = rho0 * (1 + perturbation_amplitude * np.sin(2 * np.pi * x / (x[-1] - x[0])))
    
    # Vitesse d'équilibre selon relation V-rho
    v_eq = params.V0 * (1 - rho_eq / params.rho_max)
    
    # Assurer positivité
    rho_eq = np.maximum(rho_eq, 0.01)
    v_eq = np.maximum(v_eq, 0.1)
    
    return rho_eq, v_eq