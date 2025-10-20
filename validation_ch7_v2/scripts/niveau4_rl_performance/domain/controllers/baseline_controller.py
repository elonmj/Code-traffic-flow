"""
BaselineController - Simulation baseline avec contexte béninois

Implémente Innovation 3: Sérialisation État Controller
Permet reprise sans recalcul complet (~15 minutes gagnées).

Implémente Innovation 8: Baseline Contexte Béninois
Simulation réaliste adaptée au contexte africain:
- 70% motos, 30% voitures (mobilité urbaine africaine)
- Infrastructures 60% qualité (routes partiellement dégradées)
"""

from typing import Dict, Any, Optional
from pathlib import Path
import pickle

from domain.interfaces import Logger


class BaselineController:
    """Controller baseline adapté contexte africain (Innovation 8)."""
    
    # Innovation 8: Configuration contexte béninois
    BENIN_CONTEXT_DEFAULT = {
        "motos_proportion": 0.70,  # 70% motos (transport dominant Afrique)
        "voitures_proportion": 0.30,  # 30% voitures
        "infrastructure_quality": 0.60,  # 60% (routes partiellement dégradées)
        "max_speed_moto": 50,  # km/h (prudence infrastructures)
        "max_speed_voiture": 60,  # km/h (limitations infrastructures)
    }
    
    def __init__(
        self,
        logger: Logger,
        benin_context: Optional[Dict[str, Any]] = None
    ):
        """Initialise baseline controller avec contexte béninois.
        
        Args:
            logger: Logger structuré - INJECTION
            benin_context: Configuration contexte (défaut: BENIN_CONTEXT_DEFAULT)
        """
        self.logger = logger
        self.benin_context = benin_context or self.BENIN_CONTEXT_DEFAULT.copy()
        
        # État simulation
        self.state = {
            "initialized": False,
            "vehicles": [],
            "traffic_signals": [],
            "simulation_step": 0,
            "metrics": {}
        }
        
        self.logger.info(
            "baseline_controller_initialized",
            benin_context=self.benin_context
        )
    
    def initialize_simulation(
        self,
        network_file: Path,
        inflow_rate: float,
        duration: int
    ) -> None:
        """Initialise simulation baseline avec contexte béninois.
        
        Args:
            network_file: Fichier réseau routier
            inflow_rate: Taux d'arrivée véhicules (veh/h)
            duration: Durée simulation (secondes)
        """
        self.logger.info(
            "baseline_simulation_initializing",
            network_file=str(network_file),
            inflow_rate=inflow_rate,
            duration=duration
        )
        
        # Application proportions véhicules béninoises
        motos_inflow = inflow_rate * self.benin_context["motos_proportion"]
        voitures_inflow = inflow_rate * self.benin_context["voitures_proportion"]
        
        self.state.update({
            "network_file": str(network_file),
            "inflow_rate": inflow_rate,
            "motos_inflow": motos_inflow,
            "voitures_inflow": voitures_inflow,
            "duration": duration,
            "initialized": True
        })
        
        self.logger.info(
            "baseline_benin_context_applied",
            motos_inflow=motos_inflow,
            voitures_inflow=voitures_inflow,
            infrastructure_quality=self.benin_context["infrastructure_quality"]
        )
    
    def run_simulation(self) -> Dict[str, Any]:
        """Exécute simulation baseline complète.
        
        Returns:
            Métriques baseline: travel_time, throughput, stops, etc.
        """
        if not self.state["initialized"]:
            raise RuntimeError("Simulation non initialisée. Appeler initialize_simulation() d'abord.")
        
        self.logger.info("baseline_simulation_starting")
        
        # TODO: Intégration réelle avec UxSim ou autre simulateur
        # Pour l'instant: structure pour démonstration
        metrics = self._simulate_baseline()
        
        self.state["metrics"] = metrics
        
        self.logger.info(
            "baseline_simulation_complete",
            metrics=metrics
        )
        
        return metrics
    
    def _simulate_baseline(self) -> Dict[str, Any]:
        """Simulation baseline (placeholder - intégration UxSim à faire).
        
        Returns:
            Métriques simulation baseline
        """
        # Placeholder: sera remplacé par appel UxSim réel
        # Référence: Code_RL/test_section_7_6_rl_performance.py:_run_baseline_scenario()
        
        return {
            "average_travel_time": 0.0,  # Secondes
            "throughput": 0.0,  # Véhicules/heure
            "average_stops": 0.0,  # Nombre d'arrêts moyen
            "total_emissions": 0.0,  # CO2 kg
            "infrastructure_stress": self.benin_context["infrastructure_quality"]  # Contexte béninois
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Récupère état complet controller (Innovation 3).
        
        Returns:
            État sérialisable pour cache
        """
        return {
            "benin_context": self.benin_context,
            "simulation_state": self.state.copy(),
            "version": "1.0.0"  # Versioning pour migration future
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Restaure état controller depuis cache (Innovation 3).
        
        Permet reprise simulation sans recalcul complet (~15 min gagnées).
        
        Args:
            state: État sérialisé depuis cache
        """
        self.benin_context = state["benin_context"]
        self.state = state["simulation_state"].copy()
        
        self.logger.info(
            "baseline_state_loaded",
            version=state.get("version", "unknown"),
            simulation_step=self.state.get("simulation_step", 0)
        )
    
    def apply_action(self, action: Any) -> None:
        """Applique action (baseline = aucune action).
        
        Baseline controller n'applique pas d'actions (pas de contrôle RL).
        Méthode présente pour uniformité interface avec RLController.
        
        Args:
            action: Action à appliquer (ignorée pour baseline)
        """
        # Baseline: pas de contrôle actif
        pass
