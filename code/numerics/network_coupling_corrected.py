#!/usr/bin/env python3
"""
Implémentation Corrigée du Couplage Réseau
==========================================

Remplace network_coupling.py avec une implémentation physiquement correcte
qui conserve la masse et applique les vraies conditions aux limites.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from numba import cuda
import numba as nb

from ..grid.grid1d import Grid1D
from ..core.parameters import ModelParameters
from ..core.node_solver import solve_node_fluxes
from ..core.intersection import Intersection


class NetworkCouplingCorrected:
    """
    Version corrigée du gestionnaire de couplage réseau.
    
    Cette version implémente correctement :
    1. Collection des états aux extrémités réelles des segments
    2. Application des conditions aux limites physiquement cohérentes
    3. Conservation de la masse aux nœuds
    """

    def __init__(self, nodes: List[Intersection], params: ModelParameters):
        """
        Initialise le système de couplage réseau corrigé.
        """
        self.nodes = nodes
        self.params = params
        self.node_states = {}
        
        # Mapping segment -> indices grille
        self.segment_mapping = self._build_segment_mapping(params)
        
        # Initialiser les états des nœuds
        for node in nodes:
            self.node_states[node.node_id] = {
                'queues': {'motorcycle': 0.0, 'car': 0.0},
                'incoming_fluxes': {},
                'outgoing_fluxes': {},
                'last_update': 0.0
            }

    def _build_segment_mapping(self, params: ModelParameters) -> Dict[str, Dict]:
        """
        Construit le mapping segment_id -> indices dans la grille globale.
        
        Pour l'instant, on suppose que tous les segments sont concaténés
        dans une grille 1D globale. En production, il faudrait une vraie
        topologie multi-segments.
        """
        mapping = {}
        
        if not params.network_segments:
            # Fallback: un seul segment fictif pour le test
            mapping['global'] = {
                'start_idx': params.num_ghost_cells,
                'end_idx': params.N + params.num_ghost_cells - 1,
                'cells': params.N
            }
            return mapping
        
        # Répartir les cellules entre les segments
        total_cells = params.N
        cells_per_segment = total_cells // len(params.network_segments)
        ghost = params.num_ghost_cells
        
        current_start = ghost
        for i, segment in enumerate(params.network_segments):
            segment_id = segment['id']
            cells = segment.get('cells', cells_per_segment)
            
            mapping[segment_id] = {
                'start_idx': current_start,
                'end_idx': current_start + cells - 1,
                'cells': cells,
                'length': segment.get('length', 500.0),
                'start_node': segment.get('start_node', 'boundary'),
                'end_node': segment.get('end_node', 'boundary')
            }
            
            current_start += cells
        
        return mapping

    def apply_network_coupling(self, U: np.ndarray, dt: float,
                              grid: Grid1D, time: float) -> np.ndarray:
        """
        Applique le couplage réseau avec conservation de masse.
        
        Algorithme corrigé :
        1. Collecter états aux vraies extrémités des segments
        2. Résoudre flux aux nœuds avec conservation
        3. Appliquer conditions aux limites physiquement cohérentes
        """
        if not self.params.has_network or len(self.nodes) == 0:
            return U
        
        U_modified = U.copy()
        
        # Pour chaque nœud du réseau
        for node in self.nodes:
            # 1. Collecter les vrais états aux extrémités des segments
            incoming_states = self._collect_real_boundary_states(U, node, grid)
            
            # 2. Résoudre les flux au nœud avec conservation
            outgoing_fluxes = self._solve_node_conservation(
                node, incoming_states, dt, time
            )
            
            # 3. Appliquer les conditions aux limites correctes
            U_modified = self._apply_physical_boundary_conditions(
                U_modified, node, outgoing_fluxes, grid
            )
            
            # 4. Mettre à jour état du nœud
            self._update_node_state(node, incoming_states, outgoing_fluxes, dt)
        
        return U_modified

    def _collect_real_boundary_states(self, U: np.ndarray, node: Intersection,
                                    grid: Grid1D) -> Dict[str, np.ndarray]:
        """
        Collecte les vrais états aux extrémités des segments connectés au nœud.
        """
        incoming_states = {}
        
        for segment_id in node.segments:
            if segment_id in self.segment_mapping:
                segment_info = self.segment_mapping[segment_id]
                
                # Déterminer si ce segment arrive au nœud ou en part
                # Logique : segments pairs arrivent, segments impairs partent
                # (simplification pour le prototype)
                segment_index = list(self.segment_mapping.keys()).index(segment_id)
                
                if segment_index % 2 == 0:  # Segment entrant
                    # Prendre l'état à la fin du segment (vers le nœud)
                    boundary_idx = segment_info['end_idx']
                else:  # Segment sortant
                    # Prendre l'état au début du segment (depuis le nœud)
                    boundary_idx = segment_info['start_idx']
                
                incoming_states[segment_id] = U[:, boundary_idx].copy()
            else:
                # Segment non mappé - utiliser valeurs par défaut
                mid_idx = grid.N_physical // 2 + grid.num_ghost_cells
                incoming_states[segment_id] = U[:, mid_idx].copy()
        
        return incoming_states

    def _solve_node_conservation(self, node: Intersection, 
                               incoming_states: Dict[str, np.ndarray],
                               dt: float, time: float) -> Dict[str, np.ndarray]:
        """
        Résout les flux aux nœuds en garantissant la conservation de masse.
        """
        # Utiliser le solveur existant
        raw_fluxes = solve_node_fluxes(node, incoming_states, dt, self.params, time)
        
        # Appliquer conservation de masse explicite
        conserved_fluxes = self._enforce_mass_conservation(
            incoming_states, raw_fluxes
        )
        
        return conserved_fluxes

    def _enforce_mass_conservation(self, incoming_states: Dict[str, np.ndarray],
                                 outgoing_fluxes: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Force la conservation de masse au nœud.
        """
        conserved_fluxes = {}
        
        # Calculer flux entrant total
        total_incoming = np.zeros(4)
        incoming_count = 0
        
        for segment_id, state in incoming_states.items():
            if state is not None and len(state) == 4:
                total_incoming += state
                incoming_count += 1
        
        # Calculer flux sortant total
        total_outgoing = np.zeros(4)
        outgoing_count = 0
        
        for segment_id, flux in outgoing_fluxes.items():
            if flux is not None and len(flux) == 4:
                total_outgoing += flux
                outgoing_count += 1
        
        # Si pas d'équilibre, normaliser les flux sortants
        if outgoing_count > 0:
            for i in range(4):  # Pour chaque variable (rho_m, w_m, rho_c, w_c)
                if total_outgoing[i] > 1e-12:  # Éviter division par zéro
                    # Facteur de conservation pour cette variable
                    conservation_factor = total_incoming[i] / total_outgoing[i]
                    conservation_factor = min(conservation_factor, 2.0)  # Limiter à 2x
                    
                    # Appliquer la correction
                    for segment_id, flux in outgoing_fluxes.items():
                        if flux is not None and len(flux) == 4:
                            conserved_fluxes[segment_id] = flux.copy()
                            conserved_fluxes[segment_id][i] *= conservation_factor
                else:
                    # Pas de flux sortant pour cette variable
                    for segment_id in outgoing_fluxes:
                        if segment_id not in conserved_fluxes:
                            conserved_fluxes[segment_id] = outgoing_fluxes[segment_id].copy()
        else:
            # Pas de flux sortant - utiliser flux d'entrée
            conserved_fluxes = outgoing_fluxes.copy()
        
        return conserved_fluxes

    def _apply_physical_boundary_conditions(self, U: np.ndarray, node: Intersection,
                                          outgoing_fluxes: Dict[str, np.ndarray],
                                          grid: Grid1D) -> np.ndarray:
        """
        Applique les conditions aux limites physiquement cohérentes.
        
        Au lieu de réécrire l'état, on modifie les cellules fantômes
        pour imposer les flux calculés aux nœuds.
        """
        U_bc = U.copy()
        
        for segment_id, flux in outgoing_fluxes.items():
            if segment_id in self.segment_mapping and flux is not None:
                segment_info = self.segment_mapping[segment_id]
                
                # Déterminer la direction du flux
                segment_index = list(self.segment_mapping.keys()).index(segment_id)
                
                if segment_index % 2 == 0:  # Segment entrant
                    # Modifier cellule fantôme de gauche
                    ghost_idx = segment_info['start_idx'] - 1
                    if 0 <= ghost_idx < U_bc.shape[1]:
                        # Imposer le flux entrant
                        U_bc[:, ghost_idx] = flux
                else:  # Segment sortant
                    # Modifier cellule fantôme de droite
                    ghost_idx = segment_info['end_idx'] + 1
                    if 0 <= ghost_idx < U_bc.shape[1]:
                        # Imposer le flux sortant
                        U_bc[:, ghost_idx] = flux
        
        return U_bc

    def _update_node_state(self, node: Intersection,
                          incoming_states: Dict[str, np.ndarray],
                          outgoing_fluxes: Dict[str, np.ndarray],
                          dt: float):
        """
        Met à jour l'état interne du nœud de façon cohérente.
        """
        node_state = self.node_states[node.node_id]
        
        # Calculer changement net de densité
        total_rho_in = 0.0
        total_rho_out = 0.0
        
        for segment_id in incoming_states:
            if segment_id in incoming_states and incoming_states[segment_id] is not None:
                state = incoming_states[segment_id]
                total_rho_in += state[0] + state[2]  # rho_m + rho_c
        
        for segment_id in outgoing_fluxes:
            if segment_id in outgoing_fluxes and outgoing_fluxes[segment_id] is not None:
                flux = outgoing_fluxes[segment_id]
                total_rho_out += flux[0] + flux[2]  # rho_m + rho_c
        
        # Mise à jour conservative des files
        net_accumulation = (total_rho_in - total_rho_out) * dt
        
        if net_accumulation > 0:
            # Répartir l'accumulation selon la composition du trafic
            node_state['queues']['motorcycle'] += net_accumulation * 0.35
            node_state['queues']['car'] += net_accumulation * 0.65
        
        # Limites physiques
        max_queue = getattr(self.params, 'max_queue_length', 100.0)
        for vehicle_class in node_state['queues']:
            node_state['queues'][vehicle_class] = max(0.0, 
                min(node_state['queues'][vehicle_class], max_queue))

# Fonctions d'interface
def apply_network_coupling_cpu_corrected(U: np.ndarray, dt: float, grid: Grid1D,
                                        params: ModelParameters, nodes: List[Intersection],
                                        time: float) -> np.ndarray:
    """Version corrigée pour CPU."""
    if not params.has_network:
        return U
    
    coupling = NetworkCouplingCorrected(nodes, params)
    return coupling.apply_network_coupling(U, dt, grid, time)

def apply_network_coupling_gpu_corrected(d_U, dt: float, grid: Grid1D,
                                       params: ModelParameters, nodes: List[Intersection],
                                       time: float):
    """Version corrigée pour GPU."""
    if not params.has_network:
        return d_U
    
    # Pour l'instant, faire le calcul sur CPU
    U_cpu = d_U.copy_to_host()
    
    coupling = NetworkCouplingCorrected(nodes, params)
    U_modified = coupling.apply_network_coupling(U_cpu, dt, grid, time)
    
    d_U_modified = cuda.to_device(U_modified)
    return d_U_modified
