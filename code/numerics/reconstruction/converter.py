import numpy as np
from numba import njit, cuda
from ...core import physics

@njit
def conserved_to_primitives_arr(U, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c):
    """
    Convertit un tableau de variables d'état conservées U en variables primitives P.
    
    Args:
        U (np.ndarray): Tableau des états conservés (4, N).
        ... (params): Paramètres physiques scalaires pour le calcul de la pression.
        
    Returns:
        np.ndarray: Tableau des états primitifs P = (rho_m, v_m, rho_c, v_c) (4, N).
    """
    rho_m, w_m, rho_c, w_c = U[0,:], U[1,:], U[2,:], U[3,:]
    
    # Calcule la pression nécessaire pour obtenir la vitesse
    p_m, p_c = physics.calculate_pressure(rho_m, rho_c, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c)
    
    # Calcule la vitesse physique v = w - p
    v_m = w_m - p_m
    v_c = w_c - p_c
    
    # Construit le tableau des variables primitives
    P = np.empty_like(U)
    P[0,:], P[1,:], P[2,:], P[3,:] = rho_m, v_m, rho_c, v_c
    return P

@njit
def primitives_to_conserved_arr(P, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c):
    """
    Convertit un tableau de variables d'état primitives P en variables conservées U.
    
    Args:
        P (np.ndarray): Tableau des états primitifs (4, N).
        ... (params): Paramètres physiques scalaires pour le calcul de la pression.
        
    Returns:
        np.ndarray: Tableau des états conservés U = (rho_m, w_m, rho_c, w_c) (4. N).
    """
    rho_m, v_m, rho_c, v_c = P[0,:], P[1,:], P[2,:], P[3,:]
    
    # Calcule la pression nécessaire pour obtenir la variable lagrangienne w
    p_m, p_c = physics.calculate_pressure(rho_m, rho_c, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c)
    
    # Calcule la variable lagrangienne w = v + p
    w_m = v_m + p_m
    w_c = v_c + p_c
    
    # Construit le tableau des variables conservées
    U = np.empty_like(P)
    U[0,:], U[1,:], U[2,:], U[3,:] = rho_m, w_m, rho_c, w_c
    return U

# --- GPU Versions ---

@cuda.jit
def conserved_to_primitives_arr_gpu(U, P, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c):
    """
    GPU KERNEL: Convertit un tableau de variables d'état conservées U en variables primitives P.
    Modifie P sur place.
    """
    i = cuda.grid(1)
    if i < U.shape[1]:
        # Extraire les variables conservées pour la cellule i
        rho_m = U[0, i]
        w_m = U[1, i]
        rho_c = U[2, i]
        w_c = U[3, i]

        # Calcule la pression (réutilisation de la logique CPU, qui est compatible)
        p_m_i, p_c_i = physics.calculate_pressure_component(rho_m, rho_c, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c)

        # Calcule la vitesse physique v = w - p
        v_m = w_m - p_m_i
        v_c = w_c - p_c_i

        # Stocke les variables primitives dans le tableau de sortie P
        P[0, i] = rho_m
        P[1, i] = v_m
        P[2, i] = rho_c
        P[3, i] = v_c

@cuda.jit
def primitives_to_conserved_arr_gpu(P, U, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c):
    """
    GPU KERNEL: Convertit un tableau de variables d'état primitives P en variables conservées U.
    Modifie U sur place.
    """
    i = cuda.grid(1)
    if i < P.shape[1]:
        # Extraire les variables primitives pour la cellule i
        rho_m = P[0, i]
        v_m = P[1, i]
        rho_c = P[2, i]
        v_c = P[3, i]

        # Calcule la pression
        p_m_i, p_c_i = physics.calculate_pressure_component(rho_m, rho_c, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c)

        # Calcule la variable lagrangienne w = v + p
        w_m = v_m + p_m_i
        w_c = v_c + p_c_i

        # Stocke les variables conservées dans le tableau de sortie U
        U[0, i] = rho_m
        U[1, i] = w_m
        U[2, i] = rho_c
        U[3, i] = w_c