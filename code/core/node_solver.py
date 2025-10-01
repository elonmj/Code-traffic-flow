import numpy as np
from typing import Dict, List, Any, Optional
from ..core.parameters import ModelParameters
from ..core.intersection import Intersection
from ..core.traffic_lights import TrafficLightController
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
    Calcule le flux sortant pour un segment spécifique.
    """
    # Extraire les densités et vitesses
    rho_m, w_m, rho_c, w_c = U_in

    # Calculer les vitesses physiques
    p_m, p_c = _calculate_pressures(rho_m, rho_c, params)
    v_m, v_c = _calculate_velocities(w_m, w_c, p_m, p_c)

    # Vérifier si le segment a le feu vert
    has_green_light = segment_id in green_segments

    # Facteur de réduction si feu rouge
    light_factor = 1.0 if has_green_light else params.red_light_factor

    # Vérifier les files d'attente au nœud
    queue_factor = _calculate_queue_factor(node, segment_id)

    # Calculer les flux avec réduction
    flux_m = rho_m * v_m * light_factor * queue_factor
    flux_c = rho_c * v_c * light_factor * queue_factor

    # Créer le vecteur de flux sortant
    # Note: Pour les nœuds, on impose directement les densités et vitesses
    # plutôt que de calculer un vrai flux numérique
    U_out = np.array([
        rho_m,  # Densité moto conservée
        w_m * light_factor * queue_factor,  # Vitesse moto réduite
        rho_c,  # Densité voiture conservée
        w_c * light_factor * queue_factor   # Vitesse voiture réduite
    ])

    return U_out


def _calculate_pressures(rho_m: float, rho_c: float, params: ModelParameters) -> tuple:
    """
    Calcule les pressions pour les deux classes de véhicules.
    """
    from ..core import physics
    return physics.calculate_pressure(
        rho_m, rho_c, params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c
    )


def _calculate_velocities(w_m: float, w_c: float, p_m: float, p_c: float) -> tuple:
    """
    Calcule les vitesses physiques.
    """
    from ..core import physics
    return physics.calculate_physical_velocity(w_m, w_c, p_m, p_c)


def _calculate_queue_factor(node: Intersection, segment_id: str) -> float:
    """
    Calcule le facteur de réduction dû aux files d'attente.
    """
    # Logique simplifiée: réduction linéaire avec la longueur de la file
    # À affiner selon le modèle de files d'attente
    total_queue = (node.queues.get('motorcycle', 0.0) +
                   node.queues.get('car', 0.0))

    # Paramètre de réduction maximale
    max_reduction = 0.1  # 10% du flux normal si file très longue
    queue_threshold = 50.0  # Longueur de file pour réduction maximale

    if total_queue >= queue_threshold:
        return max_reduction
    else:
        # Réduction linéaire
        return 1.0 - (total_queue / queue_threshold) * (1.0 - max_reduction)


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
