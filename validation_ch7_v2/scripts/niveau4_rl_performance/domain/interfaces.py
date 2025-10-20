"""
Domain Interfaces - Abstractions pour Dependency Inversion Principle (DIP)

Ce module définit les interfaces abstraites que le code métier utilise.
Les implémentations concrètes sont dans infrastructure/.

Principe: Le domain dépend d'abstractions, pas de détails d'implémentation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path


class CacheStorage(ABC):
    """Interface abstraite pour le stockage du cache (Innovation 1: Cache Additif Baseline)."""
    
    @abstractmethod
    def save(self, key: str, data: Dict[str, Any]) -> None:
        """Sauvegarde données dans le cache.
        
        Args:
            key: Identifiant unique du cache (ex: "baseline_scenario1")
            data: Données à sauvegarder (dict sérialisable)
        """
        pass
    
    @abstractmethod
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Charge données depuis le cache.
        
        Args:
            key: Identifiant unique du cache
            
        Returns:
            Données chargées ou None si cache inexistant
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Vérifie si une clé existe dans le cache.
        
        Args:
            key: Identifiant unique du cache
            
        Returns:
            True si le cache existe, False sinon
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Supprime une entrée du cache.
        
        Args:
            key: Identifiant unique du cache
        """
        pass


class ConfigLoader(ABC):
    """Interface abstraite pour le chargement de configuration (Innovation 6: DRY + Config externe)."""
    
    @abstractmethod
    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """Charge configuration depuis fichier.
        
        Args:
            config_path: Chemin vers fichier config (YAML, JSON, etc.)
            
        Returns:
            Configuration chargée sous forme de dictionnaire
        """
        pass
    
    @abstractmethod
    def get_scenarios(self) -> List[Dict[str, Any]]:
        """Retourne la liste des scénarios configurés."""
        pass
    
    @abstractmethod
    def get_rl_config(self, algorithm: str) -> Dict[str, Any]:
        """Retourne la configuration pour un algorithme RL spécifique.
        
        Args:
            algorithm: Nom de l'algorithme (ex: "DQN", "PPO")
            
        Returns:
            Configuration de l'algorithme
        """
        pass


class Logger(ABC):
    """Interface abstraite pour le logging structuré (Innovation 7: Dual Logging)."""
    
    @abstractmethod
    def info(self, event: str, **kwargs) -> None:
        """Log événement de niveau INFO avec contexte structuré.
        
        Args:
            event: Nom de l'événement (ex: "cache_loaded")
            **kwargs: Contexte structuré (scenario, timestep, reward, etc.)
        """
        pass
    
    @abstractmethod
    def warning(self, event: str, **kwargs) -> None:
        """Log événement de niveau WARNING."""
        pass
    
    @abstractmethod
    def error(self, event: str, **kwargs) -> None:
        """Log événement de niveau ERROR."""
        pass
    
    @abstractmethod
    def exception(self, event: str, **kwargs) -> None:
        """Log exception avec traceback complet."""
        pass


class CheckpointStorage(ABC):
    """Interface abstraite pour le stockage des checkpoints RL (Innovation 2: Config-Hashing)."""
    
    @abstractmethod
    def save_checkpoint(self, checkpoint_path: Path, model: Any) -> None:
        """Sauvegarde checkpoint RL.
        
        Args:
            checkpoint_path: Chemin complet du checkpoint
            model: Modèle RL à sauvegarder (Stable-Baselines3)
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_path: Path, env: Any) -> Any:
        """Charge checkpoint RL.
        
        Args:
            checkpoint_path: Chemin complet du checkpoint
            env: Environnement Gymnasium
            
        Returns:
            Modèle RL chargé
        """
        pass
    
    @abstractmethod
    def list_checkpoints(self, pattern: str) -> List[Path]:
        """Liste tous les checkpoints matchant un pattern.
        
        Args:
            pattern: Pattern glob (ex: "rl_model_a3f7b2c1_iter*.zip")
            
        Returns:
            Liste des chemins de checkpoints
        """
        pass
    
    @abstractmethod
    def delete_checkpoint(self, checkpoint_path: Path) -> None:
        """Supprime un checkpoint.
        
        Args:
            checkpoint_path: Chemin du checkpoint à supprimer
        """
        pass
