from typing import List, Dict, Any, Optional
import numpy as np


class Phase:
    """
    Représente une phase de feux de circulation.
    """

    def __init__(self, duration: float, green_segments: List[str],
                 yellow_segments: Optional[List[str]] = None):
        """
        Initialise une phase.

        Args:
            duration: Durée de la phase en secondes
            green_segments: Segments avec feu vert
            yellow_segments: Segments avec feu jaune (optionnel)
        """
        self.duration = duration
        self.green_segments = green_segments
        self.yellow_segments = yellow_segments or []

    def is_green(self, segment_id: str) -> bool:
        """Vérifie si un segment a le feu vert."""
        return segment_id in self.green_segments

    def is_yellow(self, segment_id: str) -> bool:
        """Vérifie si un segment a le feu jaune."""
        return segment_id in self.yellow_segments


class TrafficLightController:
    """
    Contrôleur de feux de circulation pour une intersection.
    Gère les cycles, phases et coordination avec d'autres intersections.
    """

    def __init__(self, cycle_time: float = 60.0, phases: Optional[List[Phase]] = None,
                 offset: float = 0.0):
        """
        Initialise le contrôleur de feux.

        Args:
            cycle_time: Durée totale d'un cycle en secondes
            phases: Liste des phases du cycle
            offset: Décalage temporel par rapport à t=0
        """
        self.cycle_time = cycle_time
        self.offset = offset
        self.current_phase_index = 0
        self.green_time = 30.0

        # Phases par défaut si non spécifiées
        if phases is None or len(phases) == 0:
            self.phases = self._create_default_phases()
        else:
            self.phases = phases

        # Statistiques
        self.stats = {
            'total_cycles': 0,
            'avg_green_time': 0.0,
            'phase_changes': 0
        }

    def _create_default_phases(self) -> List[Phase]:
        """
        Crée des phases par défaut pour une intersection à 4 branches.
        """
        return [
            Phase(duration=25.0, green_segments=['north', 'south']),
            Phase(duration=5.0, green_segments=[], yellow_segments=['north', 'south']),
            Phase(duration=25.0, green_segments=['east', 'west']),
            Phase(duration=5.0, green_segments=[], yellow_segments=['east', 'west'])
        ]

    def get_current_phase(self, time: float) -> Phase:
        """
        Retourne la phase actuelle à un instant donné.

        Args:
            time: Temps en secondes

        Returns:
            Phase actuelle
        """
        if len(self.phases) == 0:
            return Phase(0.0, [])  # Phase vide

        # Temps dans le cycle
        cycle_time = (time + self.offset) % self.cycle_time

        # Trouver la phase correspondante
        cumulative_time = 0.0
        for i, phase in enumerate(self.phases):
            cumulative_time += phase.duration
            if cycle_time < cumulative_time:
                if i != self.current_phase_index:
                    self.current_phase_index = i
                    self.stats['phase_changes'] += 1
                return phase

        # Par sécurité, retourner la dernière phase
        return self.phases[-1]

    def get_current_green_segments(self, time: float) -> List[str]:
        """
        Retourne la liste des segments avec feu vert à un instant donné.

        Args:
            time: Temps en secondes

        Returns:
            Liste des segments avec feu vert
        """
        phase = self.get_current_phase(time)
        return phase.green_segments.copy()

    def get_current_yellow_segments(self, time: float) -> List[str]:
        """
        Retourne la liste des segments avec feu jaune.

        Args:
            time: Temps en secondes

        Returns:
            Liste des segments avec feu jaune
        """
        phase = self.get_current_phase(time)
        return phase.yellow_segments.copy()

    def get_time_to_next_phase(self, time: float) -> float:
        """
        Calcule le temps restant avant la prochaine phase.

        Args:
            time: Temps actuel

        Returns:
            Temps en secondes avant changement de phase
        """
        cycle_time = (time + self.offset) % self.cycle_time
        cumulative_time = 0.0

        for phase in self.phases:
            cumulative_time += phase.duration
            if cycle_time < cumulative_time:
                return cumulative_time - cycle_time

        return 0.0

    def get_phase_progress(self, time: float) -> float:
        """
        Retourne le progrès dans la phase actuelle (0.0 à 1.0).

        Args:
            time: Temps actuel

        Returns:
            Progrès dans la phase (0.0 = début, 1.0 = fin)
        """
        cycle_time = (time + self.offset) % self.cycle_time
        cumulative_time = 0.0

        for phase in self.phases:
            phase_start = cumulative_time
            cumulative_time += phase.duration

            if cycle_time < cumulative_time:
                return (cycle_time - phase_start) / phase.duration

        return 1.0

    def set_offset(self, offset: float):
        """
        Définit le décalage temporel du cycle.
        """
        self.offset = offset

    def add_phase(self, phase: Phase):
        """
        Ajoute une phase au cycle.
        """
        self.phases.append(phase)
        self._update_cycle_time()

    def remove_phase(self, index: int):
        """
        Retire une phase du cycle.
        """
        if 0 <= index < len(self.phases):
            self.phases.pop(index)
            self._update_cycle_time()

    def _update_cycle_time(self):
        """
        Met à jour la durée totale du cycle.
        """
        self.cycle_time = sum(phase.duration for phase in self.phases)

    def get_cycle_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le cycle actuel.
        """
        return {
            'cycle_time': self.cycle_time,
            'num_phases': len(self.phases),
            'phase_durations': [p.duration for p in self.phases],
            'offset': self.offset
        }

    def reset_stats(self):
        """
        Réinitialise les statistiques.
        """
        self.stats = {
            'total_cycles': 0,
            'avg_green_time': 0.0,
            'phase_changes': 0
        }

    def update_stats(self, dt: float):
        """
        Met à jour les statistiques.
        """
        # TODO: Implémenter la mise à jour des statistiques
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the traffic light controller.
        """
        return {
            "cycle_time": self.cycle_time,
            "green_time": self.green_time,
            "offset": self.offset,
            "phases": [
                {
                    "duration": phase.duration,
                    "green_segments": phase.green_segments,
                    "yellow_segments": phase.yellow_segments,
                }
                for phase in self.phases
            ],
        }


class CoordinatedTrafficLightController(TrafficLightController):
    """
    Contrôleur de feux coordonnés pour plusieurs intersections.
    """

    def __init__(self, cycle_time: float = 60.0, phases: Optional[List[Phase]] = None,
                 master_offset: float = 0.0, slave_offsets: Optional[Dict[str, float]] = None):
        """
        Initialise un contrôleur coordonné.

        Args:
            cycle_time: Durée du cycle maître
            phases: Phases du cycle
            master_offset: Décalage du contrôleur maître
            slave_offsets: Décalages des contrôleurs esclaves par intersection
        """
        super().__init__(cycle_time, phases, master_offset)
        self.slave_offsets = slave_offsets or {}

    def get_slave_offset(self, intersection_id: str) -> float:
        """
        Retourne le décalage pour une intersection esclave.
        """
        return self.slave_offsets.get(intersection_id, 0.0)

    def set_slave_offset(self, intersection_id: str, offset: float):
        """
        Définit le décalage pour une intersection esclave.
        """
        self.slave_offsets[intersection_id] = offset

    def get_coordination_info(self) -> Dict[str, Any]:
        """
        Retourne les informations de coordination.
        """
        return {
            'master_offset': self.offset,
            'slave_offsets': self.slave_offsets.copy(),
            'coordination_ratio': len(self.slave_offsets)
        }


def create_traffic_light_from_config(config: Dict[str, Any]) -> TrafficLightController:
    """
    Crée un contrôleur de feux à partir d'une configuration YAML.

    Args:
        config: Configuration des feux depuis le YAML

    Returns:
        Contrôleur de feux configuré
    """
    cycle_time = config.get('cycle_time', 60.0)
    offset = config.get('offset', 0.0)

    # Créer les phases
    phases = []
    if 'phases' in config:
        for phase_config in config['phases']:
            phase = Phase(
                duration=phase_config['duration'],
                green_segments=phase_config['green_segments'],
                yellow_segments=phase_config.get('yellow_segments', [])
            )
            phases.append(phase)

    # Créer le contrôleur approprié
    if 'slave_offsets' in config:
        # Contrôleur coordonné
        return CoordinatedTrafficLightController(
            cycle_time=cycle_time,
            phases=phases,
            master_offset=offset,
            slave_offsets=config['slave_offsets']
        )
    else:
        # Contrôleur simple
        return TrafficLightController(
            cycle_time=cycle_time,
            phases=phases,
            offset=offset
        )


# Fonctions utilitaires pour les feux adaptatifs

def calculate_optimal_cycle_time(queue_lengths: Dict[str, float],
                               max_capacity: float = 0.8) -> float:
    """
    Calcule la durée optimale du cycle basée sur les files d'attente.

    Args:
        queue_lengths: Longueurs des files par segment
        max_capacity: Capacité maximale souhaitée

    Returns:
        Durée optimale du cycle
    """
    if not queue_lengths:
        return 60.0  # Cycle par défaut

    # Calcul basé sur la file la plus longue
    max_queue = max(queue_lengths.values())
    if max_queue == 0:
        return 60.0

    # Ajuster le cycle pour maintenir les files sous contrôle
    optimal_cycle = 60.0 * (max_queue / 100.0) / max_capacity
    return np.clip(optimal_cycle, 30.0, 120.0)  # Limites du cycle


def adapt_traffic_lights(controller: TrafficLightController,
                        queue_lengths: Dict[str, float],
                        time: float) -> bool:
    """
    Adapte les feux de circulation en fonction des files d'attente.

    Args:
        controller: Contrôleur à adapter
        queue_lengths: Longueurs des files actuelles
        time: Temps actuel

    Returns:
        True si les feux ont été modifiés
    """
    # Calculer le cycle optimal
    optimal_cycle = calculate_optimal_cycle_time(queue_lengths)

    # Adapter si nécessaire
    if abs(controller.cycle_time - optimal_cycle) > 5.0:  # Tolérance de 5s
        controller.cycle_time = optimal_cycle
        return True

    return False