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
def conserved_to_primitives_kernel(U, P, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c):
    """
    GPU KERNEL: Convertit un tableau de variables d'état conservées U en variables primitives P.
    Modifie P sur place.
    """
    i = cuda.grid(1)
    if i >= U.shape[1]:
        return

    rho_m, w_m, rho_c, w_c = U[0, i], U[1, i], U[2, i], U[3, i]
    
    # Inline pressure calculation for GPU
    rho_total = rho_m + rho_c
    p_m = 0.0
    p_c = 0.0
    if rho_total > epsilon:
        rho_ratio = rho_total / rho_jam
        if rho_ratio > 1.0:
            p_m = K_m * (rho_ratio**gamma_m - 1.0)
            p_c = K_c * (rho_ratio**gamma_c - 1.0)

    v_m = w_m - p_m
    v_c = w_c - p_c
    
    P[0, i] = rho_m
    P[1, i] = v_m
    P[2, i] = rho_c
    P[3, i] = v_c

def conserved_to_primitives_arr_gpu(d_U, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c, target_array=None):
    """
    GPU Dispatcher: Converts a device array of conserved variables U to primitive variables P.
    """
    if target_array is None:
        d_P = cuda.device_array_like(d_U)
    else:
        d_P = target_array

    threadsperblock = 256
    blockspergrid = (d_U.shape[1] + threadsperblock - 1) // threadsperblock
    
    conserved_to_primitives_kernel[blockspergrid, threadsperblock](
        d_U, d_P, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c
    )
    return d_P

@cuda.jit
def primitives_to_conserved_kernel(P, U, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c):
    """
    GPU KERNEL: Convertit un tableau de variables d'état primitives P en variables conservées U.
    Modifie U sur place.
    """
    i = cuda.grid(1)
    if i >= P.shape[1]:
        return

    rho_m, v_m, rho_c, v_c = P[0, i], P[1, i], P[2, i], P[3, i]
    
    # Inline pressure calculation for GPU
    rho_total = rho_m + rho_c
    p_m = 0.0
    p_c = 0.0
    if rho_total > epsilon:
        rho_ratio = rho_total / rho_jam
        if rho_ratio > 1.0:
            p_m = K_m * (rho_ratio**gamma_m - 1.0)
            p_c = K_c * (rho_ratio**gamma_c - 1.0)

    w_m = v_m + p_m
    w_c = v_c + p_c
    
    U[0, i] = rho_m
    U[1, i] = w_m
    U[2, i] = rho_c
    U[3, i] = w_c

def primitives_to_conserved_arr_gpu(d_P, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c, target_array=None):
    """
    GPU Dispatcher: Converts a device array of primitive variables P to conserved variables U.
    """
    if target_array is None:
        d_U = cuda.device_array_like(d_P)
    else:
        d_U = target_array
        
    threadsperblock = 256
    blockspergrid = (d_P.shape[1] + threadsperblock - 1) // threadsperblock
    
    primitives_to_conserved_kernel[blockspergrid, threadsperblock](
        d_P, d_U, alpha, rho_jam, epsilon, K_m, gamma_m, K_c, gamma_c
    )
    return d_U
