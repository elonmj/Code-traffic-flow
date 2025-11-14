"""
Node Solver Module - Junction State Transmission and Behavioral Coupling

This module provides utilities for network junction handling in the ARZ model:
1. State transmission between segments (simple pass-through)
2. Behavioral coupling parameter calculation (θ_k)

ARCHITECTURAL NOTE (2025-11-08 Refactoring):
--------------------------------------------- 
This module has been significantly simplified as part of the junction-aware 
solver architecture migration. Previously, this module applied queue congestion 
effects directly to states, which violated the separation of concerns.

NEW ARCHITECTURE:
- NetworkGrid._prepare_junction_info_with_theta(): Prepares ALL junction metadata
  (light_factor, queue_factor, theta_k)
- central_upwind_flux(): Applies ALL junction effects during numerical evolution
- This module: State transmission (pass-through) + theta_k calculation

DEPRECATED FUNCTIONS REMOVED:
- _calculate_queue_factor() → moved to NetworkGrid
- _calculate_pressures() → was helper for removed queue logic
- _calculate_velocities() → was helper for removed queue logic

REMAINING FUNCTIONS:
- solve_node_fluxes(): High-level node flux resolution (legacy interface)
- _calculate_outgoing_flux(): Simple pass-through state transmission
- _get_coupling_parameter(): Calculates θ_k behavioral coupling parameter
- _apply_behavioral_coupling(): Applies θ_k coupling to states

References:
    - Garavello & Piccoli (2005): Network junction coupling
    - Kolb et al. (2018): Behavioral coupling via θ_k parameter
    - Göttlich et al. (2021): Memory preservation at junctions
"""

import numpy as np
from typing import Dict, List, Any, Optional

# Imports absolus pour éviter les problèmes de top-level package
try:
    from arz_model.core.parameters import ModelParameters
    from arz_model.core.intersection import Intersection
    from arz_model.core.traffic_lights import TrafficLightController
    from arz_model.numerics.riemann_solvers import central_upwind_flux
except ImportError:
    # Fallback vers imports relatifs si exécuté comme module
    from .parameters import ModelParameters
    from .intersection import Intersection
    from .traffic_lights import TrafficLightController
    from ..numerics.riemann_solvers import central_upwind_flux


def solve_node_fluxes(node: Intersection, incoming_states: Dict[str, np.ndarray],
                     dt: float, params: ModelParameters, time: float) -> Dict[str, np.ndarray]:
    """
    Résout les flux aux nœuds pour un pas de temps donné.

    Cette fonction gère la résolution des problèmes de Riemann aux intersections,
    en tenant compte des feux de circulation, des priorités et des files d'attente.

    Args:
        node: Objet intersection
        incoming_states: États entrants par segment (clé: segment_id, valeur: état U[4])
        dt: Pas de temps
        params: Paramètres du modèle
        time: Temps actuel de simulation

    Returns:
        Dict[str, np.ndarray]: Flux sortants par segment
    """
    outgoing_fluxes = {}

    # Obtenir l'état des feux de circulation
    green_segments = node.traffic_lights.get_current_green_segments(time)

    # Pour chaque segment connecté au nœud
    for segment_id in node.segments:
        if segment_id in incoming_states:
            # État entrant depuis ce segment
            U_in = incoming_states[segment_id]

            # Calculer le flux sortant selon les règles du nœud
            outgoing_fluxes[segment_id] = _calculate_outgoing_flux(
                node, segment_id, U_in, green_segments, dt, params
            )
        else:
            # Pas d'état entrant pour ce segment (segment vide ou boundary)
            outgoing_fluxes[segment_id] = _get_default_flux(segment_id, params)

    return outgoing_fluxes


def _calculate_outgoing_flux(node: Intersection, segment_id: str, U_in: np.ndarray,
                           green_segments: List[str], dt: float,
                           params: ModelParameters) -> np.ndarray:
    """
    Transmit conserved state to downstream segment (simple pass-through).
    
    ARCHITECTURAL NOTE (2025-11-08):
    This function has been simplified to be a pure state transmission function.
    ALL junction effects are now handled by the junction-aware Riemann solver via 
    the JunctionInfo object:
    
    - Physical flux blocking: light_factor (traffic signals)
    - Queue congestion effects: queue_factor (velocity reduction)  
    - Behavioral coupling: theta_k (driver memory preservation)
    
    The separation of concerns is now restored:
    - NetworkGrid._prepare_junction_info_with_theta(): Prepares ALL junction metadata
    - central_upwind_flux(): Applies ALL junction effects during numerical evolution
    - This function: Simple state transmission (no modification)
    
    Args:
        node: Intersection node (kept for API compatibility)
        segment_id: Segment identifier (kept for API compatibility)
        U_in: Incoming conserved state [rho_m, w_m, rho_c, w_c]
        green_segments: Green light segments (kept for API compatibility)
        dt: Timestep size (kept for API compatibility)
        params: Model parameters (kept for API compatibility)
        
    Returns:
        U_out: Outgoing conserved state [rho_m, w_m, rho_c, w_c] (unchanged)
        
    Academic References:
        - Garavello & Piccoli (2005): Network junction coupling
        - Kolb et al. (2018): Behavioral coupling via θ_k parameter
        - Daganzo (1995): Supply-demand junction paradigm
        
    Implementation Note:
        Previous implementation applied queue_factor here, violating the 
        junction-aware solver pattern. This has been corrected by moving 
        queue_factor calculation to NetworkGrid and application to the Riemann solver.
    """
    # Simple pass-through: all junction effects handled by Riemann solver
    return U_in


def _get_coupling_parameter(node: Intersection, 
                            segment_id: str,
                            vehicle_class: str, 
                            params: ModelParameters,
                            time: float) -> float:
    """
    Determine θ_k coupling parameter for behavioral transmission.
    
    Based on:
        - Kolb et al. (2018): Phenomenological coupling for ARZ networks
        - Göttlich et al. (2021): Memory preservation at junctions
        - Thesis Section 4.2: θ_k specialization by junction type
    
    Args:
        node: Intersection object
        segment_id: Outgoing segment identifier
        vehicle_class: 'motorcycle' or 'car'
        params: Model parameters with θ_k values
        time: Current simulation time (s)
    
    Returns:
        θ_k ∈ [0,1]: Coupling parameter (0=reset, 1=preserve)
    """
    # Signalized intersection (traffic lights active)
    if node.traffic_lights is not None:
        green_segments = node.traffic_lights.get_current_green_segments(time)
        
        if segment_id in green_segments:
            # Green light: moderate memory (acceleration scenario)
            # Motos accelerate more aggressively than cars
            if vehicle_class == 'motorcycle':
                # ✅ FIX: Use default 0.8 if params.theta_moto_signalized is None
                return params.theta_moto_signalized if params.theta_moto_signalized is not None else 0.8
            else:
                # ✅ FIX: Use default 0.5 if params.theta_car_signalized is None
                return params.theta_car_signalized if params.theta_car_signalized is not None else 0.5
        else:
            # Red light: complete behavioral reset (vehicles stopped)
            return 0.0
    
    # Unsignalized junction: check priority hierarchy
    # Note: priority_segments attribute to be added to Intersection in future
    if hasattr(node, 'priority_segments') and segment_id in node.priority_segments:
        # Priority road: minimal behavioral disruption
        if vehicle_class == 'motorcycle':
            # ✅ FIX: Use default 0.9 if params.theta_moto_priority is None
            return params.theta_moto_priority if params.theta_moto_priority is not None else 0.9
        else:
            # ✅ FIX: Use default 0.9 if params.theta_car_priority is None
            return params.theta_car_priority if params.theta_car_priority is not None else 0.9
    else:
        # Secondary road (stop/yield): strong behavioral reset
        if vehicle_class == 'motorcycle':
            # ✅ FIX: Use default 0.1 if params.theta_moto_secondary is None
            return params.theta_moto_secondary if params.theta_moto_secondary is not None else 0.1
        else:
            # ✅ FIX: Use default 0.1 if params.theta_car_secondary is None
            return params.theta_car_secondary if params.theta_car_secondary is not None else 0.1


def _apply_behavioral_coupling(U_in: np.ndarray,
                               U_out: np.ndarray,
                               theta_k: float,
                               params: ModelParameters,
                               vehicle_class: str) -> np.ndarray:
    """
    Apply phenomenological behavioral coupling at junction.
    
    Implements thesis equation (Section 4, line 28):
        w_out = (V_e(ρ_out) + p(ρ_out)) + θ_k * [w_in - (V_e(ρ_in) + p(ρ_in))]
    
    Simplifies to: w_out = w_eq_out + θ_k * (w_in - w_eq_in)
    
    Based on:
        - Kolb et al. (2018): ARZ junction coupling conditions
        - Göttlich et al. (2021): Second-order network coupling
        - Herty & Klar (2003): Behavioral memory framework
    
    Args:
        U_in: Incoming state [ρ_m, w_m, ρ_c, w_c] at segment end
        U_out: Downstream state [ρ_m, w_m, ρ_c, w_c] at segment start
        theta_k: Coupling parameter ∈ [0,1]
        params: Model parameters
        vehicle_class: 'motorcycle' or 'car'
    
    Returns:
        Modified state vector with coupled w value
    """
    # State vector indices
    if vehicle_class == 'motorcycle':
        rho_idx, w_idx = 0, 1
        gamma = params.gamma_m
        K = params.K_m
    else:  # car
        rho_idx, w_idx = 2, 3
        gamma = params.gamma_c
        K = params.K_c
    
    # Extract densities and w values
    rho_in = U_in[rho_idx]
    w_in = U_in[w_idx]
    rho_out = U_out[rho_idx]
    
    # Compute equilibrium w values (w_eq = V_e + p)
    # Pressure term: p(ρ) = K * ρ^γ (from thesis Section 2)
    p_in = K * (rho_in ** gamma) if rho_in > params.epsilon else 0.0
    p_out = K * (rho_out ** gamma) if rho_out > params.epsilon else 0.0
    
    # Equilibrium velocity V_e(ρ) from fundamental diagram
    # Note: Simplified version using default Vmax, full implementation needs road quality R(x)
    Vmax = params.Vmax_c.get(3, 35/3.6)  # Default to category 3 (urban)
    V_e_in = Vmax * (1 - rho_in/params.rho_jam) if rho_in < params.rho_jam else 0.0
    V_e_out = Vmax * (1 - rho_out/params.rho_jam) if rho_out < params.rho_jam else 0.0
    
    w_eq_in = V_e_in + p_in
    w_eq_out = V_e_out + p_out
    
    # Apply coupling formula (thesis Section 4, Equation after line 28)
    w_out_coupled = w_eq_out + theta_k * (w_in - w_eq_in)
    
    # Create output state (preserve ρ, update w)
    U_coupled = U_out.copy()
    U_coupled[w_idx] = w_out_coupled
    
    return U_coupled


def _get_default_flux(segment_id: str, params: ModelParameters) -> np.ndarray:
    """
    Retourne un flux par défaut pour les segments sans état entrant.
    """
    # État d'équilibre avec faible densité
    return np.array([
        params.rho_eq_m,  # Densité moto d'équilibre
        0.0,              # Vitesse moto nulle
        params.rho_eq_c,  # Densité voiture d'équilibre
        0.0               # Vitesse voiture nulle
    ])


def solve_intersection_riemann(node: Intersection,
                              incoming_states: Dict[str, np.ndarray],
                              params: ModelParameters) -> Dict[str, np.ndarray]:
    """
    Résout le problème de Riemann à une intersection en utilisant
    les solveurs de Riemann standards pour chaque paire de segments.
    """
    outgoing_fluxes = {}

    # Pour une intersection simple à 4 branches
    if len(node.segments) == 4:
        # Résoudre les problèmes de Riemann par paires opposées
        pairs = [
            (node.segments[0], node.segments[2]),  # Nord-Sud
            (node.segments[1], node.segments[3])   # Est-Ouest
        ]

        for seg1, seg2 in pairs:
            if seg1 in incoming_states and seg2 in incoming_states:
                # Résoudre Riemann entre seg1 et seg2
                U_L = incoming_states[seg1]
                U_R = incoming_states[seg2]

                # Calculer le flux numérique
                flux = central_upwind_flux(U_L, U_R, params)

                # Distribuer le flux aux segments sortants
                outgoing_fluxes[seg1] = flux
                outgoing_fluxes[seg2] = -flux  # Flux opposé

    return outgoing_fluxes


def apply_priority_rules(node: Intersection, fluxes: Dict[str, np.ndarray],
                        time: float) -> Dict[str, np.ndarray]:
    """
    Applique les règles de priorité aux flux (feux de circulation, etc.).
    """
    modified_fluxes = fluxes.copy()

    # Obtenir les segments avec priorité (feu vert)
    green_segments = node.traffic_lights.get_current_green_segments(time)

    # Réduire les flux des segments sans priorité
    for segment_id in modified_fluxes:
        if segment_id not in green_segments:
            # Appliquer une forte réduction pour simuler l'arrêt
            modified_fluxes[segment_id] *= 0.1  # 10% du flux

    return modified_fluxes


def update_node_queues(node: Intersection, incoming_fluxes: Dict[str, float],
                      outgoing_fluxes: Dict[str, float], dt: float):
    """
    Met à jour les files d'attente au nœud.
    """
    for vehicle_class in ['motorcycle', 'car']:
        # Calculer la différence entre flux entrant et sortant
        incoming = incoming_fluxes.get(vehicle_class, 0.0)
        outgoing = outgoing_fluxes.get(vehicle_class, 0.0)

        # Accumulation dans la file
        delta_queue = (incoming - outgoing) * dt

        # Mettre à jour la file (ne peut pas être négative)
        node.queues[vehicle_class] = max(0.0, node.queues[vehicle_class] + delta_queue)
