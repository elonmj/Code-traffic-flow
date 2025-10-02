import numpy as np
from typing import List, Dict, Any, Optional
from numba import cuda
import numba as nb

from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters
from ..core.node_solver import solve_node_fluxes
from ..core.intersection import Intersection

class NetworkCoupling:
    """
    Gestionnaire du couplage réseau pour les simulations multi-segments.
    Coordonne les conditions aux limites entre segments connectés via des nœuds.
    """

    def __init__(self, nodes: List[Intersection], params: ModelParameters):
        """
        Initialise le système de couplage réseau.

        Args:
            nodes: Liste des intersections du réseau
            params: Paramètres du modèle
        """
        self.nodes = nodes
        self.params = params
        self.node_states = {}  # État des nœuds (files, flux)

        # Initialiser les états des nœuds
        for node in nodes:
            self.node_states[node.node_id] = {
                'queues': {'motorcycle': 0.0, 'car': 0.0},
                'incoming_fluxes': {},
                'outgoing_fluxes': {},
                'last_update': 0.0
            }

    def apply_network_coupling(self, U: np.ndarray, dt: float,
                              grid: Grid1D, time: float) -> np.ndarray:
        """
        Applique le couplage réseau aux conditions aux limites.

        Args:
            U: État du système (4, N_total)
            dt: Pas de temps
            grid: Grille 1D
            time: Temps actuel de simulation

        Returns:
            État modifié avec conditions aux limites réseau
        """
        U_modified = U.copy()

        # Pour chaque nœud, résoudre les flux et appliquer les conditions aux limites
        for node in self.nodes:
            node_state = self.node_states[node.node_id]

            # Collecter les états entrants depuis les segments connectés
            incoming_states = self._collect_incoming_states(U, node, grid)

            # Résoudre les flux au nœud
            outgoing_fluxes = solve_node_fluxes(
                node, incoming_states, dt, self.params, time
            )

            # Appliquer les conditions aux limites aux extrémités des segments
            U_modified = self._apply_node_boundary_conditions(
                U_modified, node, outgoing_fluxes, grid
            )

            # Mettre à jour l'état du nœud (files d'attente, etc.)
            self._update_node_state(node, incoming_states, outgoing_fluxes, dt)

        return U_modified

    def _collect_incoming_states(self, U: np.ndarray, node: Intersection,
                                grid: Grid1D) -> Dict[str, np.ndarray]:
        """
        Collecte les états entrants aux nœuds depuis les segments connectés.
        """
        incoming_states = {}

        for segment_id in node.segments:
            # Pour l'instant, on suppose une grille simple où tous les segments
            # sont représentés dans la même grille U
            # TODO: Implémenter la logique pour grilles multi-segments

            # Utiliser une logique simplifiée: prendre l'état au milieu du segment
            # pour les tests. En production, il faudrait une cartographie segment->indices
            mid_idx = grid.N_physical // 2
            incoming_states[segment_id] = U[:, mid_idx]

        return incoming_states

    def _apply_node_boundary_conditions(self, U: np.ndarray, node: Intersection,
                                      outgoing_fluxes: Dict[str, np.ndarray],
                                      grid: Grid1D) -> np.ndarray:
        """
        Applique les conditions aux limites provenant des nœuds.
        """
        U_bc = U.copy()

        for segment_id, flux in outgoing_fluxes.items():
            # Logique simplifiée pour les tests: appliquer le flux au milieu du segment
            # TODO: Implémenter la logique appropriée selon la topologie du réseau
            mid_idx = grid.N_physical // 2
            U_bc[:, mid_idx] = flux

        return U_bc

    def _update_node_state(self, node: Intersection,
                          incoming_states: Dict[str, np.ndarray],
                          outgoing_fluxes: Dict[str, np.ndarray],
                          dt: float):
        """
        Met à jour l'état interne du nœud (files d'attente, etc.).
        """
        node_state = self.node_states[node.node_id]

        # Calculer les différences de flux pour mettre à jour les files
        for segment_id in incoming_states:
            if segment_id in outgoing_fluxes:
                incoming_flux = incoming_states[segment_id]
                outgoing_flux = outgoing_fluxes[segment_id]

                # Différence = accumulation dans la file
                # Note: Logique simplifiée, à affiner selon le modèle de files
                for i in range(4):  # Pour chaque composante de l'état
                    delta = (incoming_flux[i] - outgoing_flux[i]) * dt
                    # Accumuler dans les files appropriées
                    if i < 2:  # Composantes moto
                        node_state['queues']['motorcycle'] += max(0, delta)
                    else:  # Composantes voiture
                        node_state['queues']['car'] += max(0, delta)

        # Appliquer les limites de files (capacité maximale)
        max_queue = self.params.max_queue_length if hasattr(self.params, 'max_queue_length') else 100.0
        for vehicle_class in node_state['queues']:
            node_state['queues'][vehicle_class] = min(
                node_state['queues'][vehicle_class], max_queue
            )

    def get_node_state(self, node_id: str) -> Dict[str, Any]:
        """
        Retourne l'état d'un nœud spécifique.
        """
        return self.node_states.get(node_id, {})

    def reset_node_states(self):
        """
        Réinitialise les états de tous les nœuds.
        """
        for node_id in self.node_states:
            self.node_states[node_id] = {
                'queues': {'motorcycle': 0.0, 'car': 0.0},
                'incoming_fluxes': {},
                'outgoing_fluxes': {},
                'last_update': 0.0
            }


def apply_network_coupling_cpu(U: np.ndarray, dt: float, grid: Grid1D,
                              params: ModelParameters, nodes: List[Intersection],
                              time: float) -> np.ndarray:
    """
    Fonction CPU pour appliquer le couplage réseau.
    """
    if not params.has_network:
        return U

    coupling = NetworkCoupling(nodes, params)
    return coupling.apply_network_coupling(U, dt, grid, time)


def apply_network_coupling_gpu(d_U, dt: float, grid: Grid1D,
                              params: ModelParameters, nodes: List[Intersection],
                              time: float):
    """
    Fonction GPU pour appliquer le couplage réseau.
    Note: Pour l'instant, le couplage réseau est fait sur CPU puis transféré.
    À optimiser pour un vrai calcul GPU.
    """
    if not params.has_network:
        return d_U

    # Transférer sur CPU pour le calcul
    U_cpu = d_U.copy_to_host()

    # Appliquer le couplage
    coupling = NetworkCoupling(nodes, params)
    U_modified = coupling.apply_network_coupling(U_cpu, dt, grid, time)

    # Recopier sur GPU
    d_U_modified = cuda.to_device(U_modified)

    return d_U_modified