"""
CacheManager - Gestion du cache baseline et RL

Implémente Innovation 1: Cache Additif Baseline
Système de cache permettant de sauvegarder et réutiliser les trajectoires baseline
pour éviter de les recalculer à chaque test RL (gain: 60% temps GPU).

Implémente également Innovation 4: Dual Cache System
Sépare cache baseline (universel) du cache RL (spécifique à config).
"""

from typing import Dict, Any, Optional
from pathlib import Path

from domain.interfaces import CacheStorage, Logger


class CacheManager:
    """Gestionnaire de cache pour baseline et RL avec injection de dépendances (DIP)."""
    
    def __init__(self, storage: CacheStorage, logger: Logger):
        """Initialise le cache manager.
        
        Args:
            storage: Implémentation du stockage (ex: PickleCacheStorage) - INJECTION
            logger: Implémentation du logging (ex: StructuredLogger) - INJECTION
        """
        self.storage = storage
        self.logger = logger
    
    def save_baseline(self, scenario_name: str, baseline_data: Dict[str, Any]) -> None:
        """Sauvegarde cache baseline pour un scénario (Innovation 1).
        
        Args:
            scenario_name: Nom du scénario (ex: "cotonou_morning_rush")
            baseline_data: Données baseline (travel_times, densities, speeds, etc.)
        """
        key = f"baseline_{scenario_name}"
        
        try:
            # Validation données avant sauvegarde
            self._validate_baseline_data(baseline_data)
            
            # Sauvegarde via storage abstrait (DIP)
            self.storage.save(key, baseline_data)
            
            # Logging structuré (Innovation 7)
            self.logger.info(
                "cache_baseline_saved",
                scenario=scenario_name,
                cache_key=key,
                has_travel_times=len(baseline_data.get('travel_times', [])),
                has_metrics='metrics' in baseline_data
            )
        except Exception as e:
            self.logger.error(
                "cache_baseline_save_failed",
                scenario=scenario_name,
                error=str(e)
            )
            raise
    
    def load_baseline(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """Charge cache baseline s'il existe (Innovation 1).
        
        Args:
            scenario_name: Nom du scénario
            
        Returns:
            Données baseline ou None si cache miss
        """
        key = f"baseline_{scenario_name}"
        
        try:
            # Tentative chargement
            baseline_data = self.storage.load(key)
            
            if baseline_data is not None:
                # CACHE HIT: Validation données
                self._validate_baseline_data(baseline_data)
                
                self.logger.info(
                    "cache_baseline_hit",
                    scenario=scenario_name,
                    cache_key=key,
                    travel_times_count=len(baseline_data.get('travel_times', []))
                )
                return baseline_data
            else:
                # CACHE MISS
                self.logger.info(
                    "cache_baseline_miss",
                    scenario=scenario_name,
                    cache_key=key
                )
                return None
                
        except Exception as e:
            # Cache corrompu: logging puis retour None (génération baseline nécessaire)
            self.logger.warning(
                "cache_baseline_corrupted",
                scenario=scenario_name,
                error=str(e),
                recovery="Will regenerate baseline from scratch"
            )
            return None
    
    def save_rl_cache(self, scenario_name: str, config_hash: str, rl_data: Dict[str, Any]) -> None:
        """Sauvegarde cache RL pour une configuration spécifique (Innovation 4).
        
        Args:
            scenario_name: Nom du scénario
            config_hash: Hash de la config RL (ex: "a3f7b2c1")
            rl_data: Données RL à cacher
        """
        key = f"rl_{scenario_name}_{config_hash}"
        
        try:
            self.storage.save(key, rl_data)
            
            self.logger.info(
                "cache_rl_saved",
                scenario=scenario_name,
                config_hash=config_hash,
                cache_key=key
            )
        except Exception as e:
            self.logger.error(
                "cache_rl_save_failed",
                scenario=scenario_name,
                config_hash=config_hash,
                error=str(e)
            )
            raise
    
    def load_rl_cache(self, scenario_name: str, config_hash: str) -> Optional[Dict[str, Any]]:
        """Charge cache RL pour une configuration spécifique (Innovation 4).
        
        Args:
            scenario_name: Nom du scénario
            config_hash: Hash de la config RL
            
        Returns:
            Données RL ou None si cache miss
        """
        key = f"rl_{scenario_name}_{config_hash}"
        
        try:
            rl_data = self.storage.load(key)
            
            if rl_data is not None:
                self.logger.info(
                    "cache_rl_hit",
                    scenario=scenario_name,
                    config_hash=config_hash
                )
                return rl_data
            else:
                self.logger.info(
                    "cache_rl_miss",
                    scenario=scenario_name,
                    config_hash=config_hash
                )
                return None
                
        except Exception as e:
            self.logger.warning(
                "cache_rl_corrupted",
                scenario=scenario_name,
                config_hash=config_hash,
                error=str(e)
            )
            return None
    
    def invalidate_baseline(self, scenario_name: str) -> None:
        """Invalide (supprime) cache baseline pour un scénario.
        
        Args:
            scenario_name: Nom du scénario
        """
        key = f"baseline_{scenario_name}"
        
        if self.storage.exists(key):
            self.storage.delete(key)
            self.logger.info(
                "cache_baseline_invalidated",
                scenario=scenario_name
            )
    
    def _validate_baseline_data(self, baseline_data: Dict[str, Any]) -> None:
        """Valide la structure des données baseline.
        
        Args:
            baseline_data: Données à valider
            
        Raises:
            ValueError: Si données invalides
        """
        required_keys = ['travel_times', 'metrics', 'scenario_config']
        
        for key in required_keys:
            if key not in baseline_data:
                raise ValueError(f"Clé manquante dans baseline data: {key}")
        
        # Validation métriques
        metrics = baseline_data['metrics']
        required_metrics = ['mean_travel_time', 'std_travel_time', 'total_vehicles']
        
        for metric in required_metrics:
            if metric not in metrics:
                raise ValueError(f"Métrique manquante: {metric}")
