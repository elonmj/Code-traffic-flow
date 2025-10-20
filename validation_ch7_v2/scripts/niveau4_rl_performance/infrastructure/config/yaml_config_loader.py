"""
YAMLConfigLoader - Chargement configuration YAML

Implémente Innovation 6: DRY Hyperparameters
Un seul fichier YAML source de vérité pour hyperparamètres validés.
Élimine duplication config entre scripts (baseline, RL, tests).
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml

from domain.interfaces import ConfigLoader


class YAMLConfigLoader(ConfigLoader):
    """Chargeur configuration YAML (Innovation 6)."""
    
    def __init__(self, config_path: Path):
        """Initialise loader avec chemin fichier config.
        
        Args:
            config_path: Chemin fichier YAML configuration
        """
        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None
        
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Fichier configuration introuvable: {self.config_path}"
            )
    
    def load_config(self) -> Dict[str, Any]:
        """Charge configuration complète depuis YAML.
        
        Returns:
            Configuration complète (scenarios + rl + benin_context)
        """
        if self._config is None:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        
        return self._config
    
    def get_scenarios(
        self,
        quick_test: bool = False
    ) -> List[Dict[str, Any]]:
        """Récupère liste scénarios de test.
        
        Args:
            quick_test: Si True, retourne uniquement scénarios quick test
            
        Returns:
            Liste configurations scénarios
        """
        config = self.load_config()
        
        if quick_test and "quick_test" in config:
            # Mode quick test: scénarios réduits pour validation rapide
            return config["quick_test"]["scenarios"]
        
        # Mode complet: tous les scénarios
        return config.get("scenarios", [])
    
    def get_rl_config(
        self,
        algorithm: str,
        quick_test: bool = False
    ) -> Dict[str, Any]:
        """Récupère configuration RL pour algorithme spécifique.
        
        Args:
            algorithm: Nom algorithme ("dqn", "ppo", "a2c")
            quick_test: Si True, applique overrides quick test
            
        Returns:
            Configuration RL (algorithm, hyperparameters, timesteps)
        """
        config = self.load_config()
        
        # Configuration algorithme de base
        rl_algorithms = config.get("rl_algorithms", {})
        
        if algorithm not in rl_algorithms:
            raise ValueError(
                f"Algorithme '{algorithm}' non trouvé dans config. "
                f"Disponibles: {list(rl_algorithms.keys())}"
            )
        
        algo_config = rl_algorithms[algorithm].copy()
        
        # Application overrides quick test si nécessaire
        if quick_test and "quick_test" in config:
            quick_overrides = config["quick_test"].get("rl_overrides", {})
            algo_config.update(quick_overrides)
        
        return {
            "algorithm": algorithm,
            "hyperparameters": algo_config.get("hyperparameters", {}),
            "total_timesteps": algo_config.get("total_timesteps", 10000)
        }
    
    def get_benin_context(self) -> Dict[str, Any]:
        """Récupère configuration contexte béninois (Innovation 8).
        
        Returns:
            Configuration contexte africain
        """
        config = self.load_config()
        return config.get("benin_context", {})
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Récupère configuration cache.
        
        Returns:
            Configuration cache (baseline_dir, rl_dir)
        """
        config = self.load_config()
        return config.get("cache", {})
    
    def get_checkpoint_config(self) -> Dict[str, Any]:
        """Récupère configuration checkpoints.
        
        Returns:
            Configuration checkpoints (checkpoints_dir, keep_last)
        """
        config = self.load_config()
        return config.get("checkpoints", {})
    
    def is_quick_test_mode(self) -> bool:
        """Vérifie si mode quick test est disponible.
        
        Returns:
            True si configuration contient section quick_test
        """
        config = self.load_config()
        return "quick_test" in config and config["quick_test"].get("enabled", False)
