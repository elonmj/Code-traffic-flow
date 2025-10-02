from typing import List, Dict, Any, Optional
import numpy as np
from .traffic_lights import TrafficLightController, Phase
from .parameters import ModelParameters


class Intersection:
    """
    Représente une intersection dans le réseau de trafic.
    Gère les files d'attente, le comportement de creeping, et les connexions aux segments.
    """

    def __init__(self, node_id: str, position: float, segments: List[str],
                 traffic_lights: Optional[TrafficLightController] = None):
        """
        Initialise une intersection.

        Args:
            node_id: Identifiant unique du nœud
            position: Position sur la grille (m)
            segments: Liste des segments connectés
            traffic_lights: Contrôleur de feux de circulation (optionnel)
        """
        self.node_id = node_id
        self.position = position
        self.segments = segments

        # Classer les segments en entrants/sortants (logique simplifiée)
        # TODO: Implémenter une logique plus sophistiquée basée sur la topologie
        self.incoming_segments = segments[:len(segments)//2]  # Première moitié = entrants
        self.outgoing_segments = segments[len(segments)//2:]  # Seconde moitié = sortants

        # Files d'attente par classe de véhicule
        self.queues = {
            'motorcycle': 0.0,  # Longueur de file en mètres
            'car': 0.0
        }

        # Capacités maximales des files
        self.max_queue_lengths = {
            'motorcycle': 100.0,  # mètres
            'car': 100.0
        }

        # Feux de circulation
        self.traffic_lights = traffic_lights or TrafficLightController()

        # Paramètres de creeping
        self.creeping_enabled = True
        self.creeping_speed = 5.0  # km/h converti en m/s
        self.creeping_threshold = 50.0  # Longueur de file pour activer creeping

        # Statistiques
        self.stats = {
            'total_waiting_time': 0.0,
            'total_vehicles_passed': 0,
            'avg_queue_length': 0.0
        }

    def update_queues(self, incoming_fluxes: Dict[str, float],
                     outgoing_capacities: Dict[str, float], dt: float):
        """
        Met à jour les files d'attente en fonction des flux entrants et sortants.

        Args:
            incoming_fluxes: Flux entrants par classe {'motorcycle': flux, 'car': flux}
            outgoing_capacities: Capacités sortantes par classe
            dt: Pas de temps
        """
        for vehicle_class in ['motorcycle', 'car']:
            incoming = incoming_fluxes.get(vehicle_class, 0.0)
            outgoing = outgoing_capacities.get(vehicle_class, 0.0)

            # Calculer l'accumulation dans la file
            delta_queue = (incoming - outgoing) * dt

            # Mettre à jour la longueur de la file
            self.queues[vehicle_class] = max(0.0,
                self.queues[vehicle_class] + delta_queue)

            # Appliquer les limites de capacité
            max_length = self.max_queue_lengths[vehicle_class]
            if self.queues[vehicle_class] > max_length:
                # File saturée - véhicules en excès sont perdus ou redirigés
                excess = self.queues[vehicle_class] - max_length
                self.queues[vehicle_class] = max_length
                # TODO: Gérer les véhicules en excès (redirection, perte)

    def get_creeping_speed(self, vehicle_class: str) -> float:
        """
        Calcule la vitesse de creeping pour une classe de véhicule.

        Args:
            vehicle_class: 'motorcycle' ou 'car'

        Returns:
            Vitesse de creeping en m/s
        """
        if not self.creeping_enabled:
            return 0.0

        queue_length = self.queues[vehicle_class]

        if queue_length < self.creeping_threshold:
            return 0.0
        else:
            # Vitesse de creeping proportionnelle à la longueur de la file
            # Vitesse max atteinte à la capacité maximale
            max_queue = self.max_queue_lengths[vehicle_class]
            speed_factor = min(queue_length / max_queue, 1.0)
            return self.creeping_speed * speed_factor

    def get_outgoing_capacity(self, segment_id: str, vehicle_class: str,
                            time: float) -> float:
        """
        Calcule la capacité sortante pour un segment et une classe de véhicule.

        Args:
            segment_id: Identifiant du segment
            vehicle_class: Classe de véhicule
            time: Temps actuel

        Returns:
            Capacité sortante (véhicules/s)
        """
        # Vérifier l'état des feux
        green_segments = self.traffic_lights.get_current_green_segments(time)
        has_green = segment_id in green_segments

        if not has_green:
            return 0.0  # Feu rouge = capacité nulle

        # Capacité de base (à définir selon le modèle)
        base_capacity = 0.5  # véhicules/s par classe

        # Réduction due aux files d'attente
        queue_length = self.queues[vehicle_class]
        max_queue = self.max_queue_lengths[vehicle_class]

        if queue_length > max_queue * 0.8:  # File > 80% capacité
            # Réduction progressive
            reduction_factor = 1.0 - (queue_length - max_queue * 0.8) / (max_queue * 0.2)
            base_capacity *= max(reduction_factor, 0.1)

        return base_capacity

    def apply_creeping(self, U: np.ndarray, segment_id: str,
                      params: ModelParameters) -> np.ndarray:
        """
        Applique le comportement de creeping à l'état du segment.

        Args:
            U: État du système (4, N)
            segment_id: Segment concerné
            params: Paramètres du modèle

        Returns:
            État modifié avec creeping
        """
        U_creeping = U.copy()

        # Indices des cellules proches du nœud
        # TODO: Déterminer les bonnes cellules selon la position du nœud
        node_cells = slice(0, 5)  # 5 premières cellules pour l'exemple

        for vehicle_class, idx in [('motorcycle', 0), ('car', 2)]:
            creeping_speed = self.get_creeping_speed(vehicle_class)

            if creeping_speed > 0:
                # Appliquer la vitesse de creeping aux cellules concernées
                # Simplification: on modifie directement les vitesses
                rho = U_creeping[idx, node_cells]
                w_current = U_creeping[idx + 1, node_cells]

                # Calculer la nouvelle vitesse avec creeping
                # TODO: Implémenter la logique physique correcte
                w_creeping = np.minimum(w_current, creeping_speed)

                U_creeping[idx + 1, node_cells] = w_creeping

        return U_creeping

    def get_queue_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur les files d'attente.
        """
        return {
            'queues': self.queues.copy(),
            'max_queues': self.max_queue_lengths.copy(),
            'total_queue_length': sum(self.queues.values()),
            'creeping_active': any(
                self.queues[cls] >= self.creeping_threshold
                for cls in self.queues
            )
        }

    def reset_stats(self):
        """
        Réinitialise les statistiques de l'intersection.
        """
        self.stats = {
            'total_waiting_time': 0.0,
            'total_vehicles_passed': 0,
            'avg_queue_length': 0.0
        }

    def update_stats(self, dt: float):
        """
        Met à jour les statistiques.
        """
        # Longueur moyenne des files
        total_queue = sum(self.queues.values())
        self.stats['avg_queue_length'] = (
            self.stats['avg_queue_length'] + total_queue
        ) / 2.0  # Moyenne glissante simple

        # TODO: Calculer le temps d'attente et les véhicules passés
        # Cela nécessiterait de tracker les véhicules individuels


class NetworkNode:
    """
    Classe de base pour tous les types de nœuds du réseau.
    """

    def __init__(self, node_id: str, position: float, node_type: str = "intersection"):
        self.node_id = node_id
        self.position = position
        self.node_type = node_type
        self.connected_segments = []

    def add_segment(self, segment_id: str):
        """Ajoute un segment connecté."""
        if segment_id not in self.connected_segments:
            self.connected_segments.append(segment_id)

    def remove_segment(self, segment_id: str):
        """Retire un segment connecté."""
        if segment_id in self.connected_segments:
            self.connected_segments.remove(segment_id)

    def get_connected_segments(self) -> List[str]:
        """Retourne la liste des segments connectés."""
        return self.connected_segments.copy()


def create_intersection_from_config(config: Dict[str, Any]) -> Intersection:
    """
    Crée une intersection à partir d'une configuration YAML.

    Args:
        config: Configuration du nœud depuis le YAML

    Returns:
        Instance d'Intersection configurée
    """
    node_id = config['id']
    position = config['position']
    segments = config['segments']

    # Configuration des feux de circulation
    traffic_lights = None
    if 'traffic_lights' in config:
        tl_config = config['traffic_lights']
        # Convertir les phases de dictionnaires en objets Phase
        phases = []
        if 'phases' in tl_config:
            for phase_dict in tl_config['phases']:
                phase = Phase(
                    duration=phase_dict['duration'],
                    green_segments=phase_dict.get('green_segments', []),
                    yellow_segments=phase_dict.get('yellow_segments', [])
                )
                phases.append(phase)

        traffic_lights = TrafficLightController(
            cycle_time=tl_config.get('cycle_time', 60.0),
            phases=phases,
            offset=tl_config.get('offset', 0.0)
        )

    intersection = Intersection(node_id, position, segments, traffic_lights)

    # Configuration des files
    if 'max_queue_lengths' in config:
        intersection.max_queue_lengths.update(config['max_queue_lengths'])

    # Configuration du creeping
    if 'creeping' in config:
        creeping_config = config['creeping']
        intersection.creeping_enabled = creeping_config.get('enabled', True)
        intersection.creeping_speed = creeping_config.get('speed_kmh', 5.0) * (1000/3600)  # km/h -> m/s
        intersection.creeping_threshold = creeping_config.get('threshold', 50.0)

    return intersection