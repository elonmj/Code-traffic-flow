"""
PickleCacheStorage - Implémentation concrète du stockage cache avec pickle

Implémente l'interface CacheStorage pour sauvegarder/charger les caches
baseline et RL en utilisant le format pickle (Innovation 1 + 4: Cache Additif + Dual Cache).
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional

from domain.interfaces import CacheStorage


class PickleCacheStorage(CacheStorage):
    """Stockage cache utilisant le format pickle."""
    
    def __init__(self, cache_root_dir: Path):
        """Initialise le storage avec répertoire racine.
        
        Args:
            cache_root_dir: Répertoire racine du cache (contiendra baseline/ et rl/)
        """
        self.cache_root_dir = Path(cache_root_dir)
        self.baseline_dir = self.cache_root_dir / "baseline"
        self.rl_dir = self.cache_root_dir / "rl"
        
        # Création architecture dual cache (Innovation 4)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.rl_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, key: str, data: Dict[str, Any]) -> None:
        """Sauvegarde données en pickle.
        
        Args:
            key: Identifiant du cache (ex: "baseline_scenario1" ou "rl_scenario1_a3f7b2c1")
            data: Données à sauvegarder
        """
        cache_file = self._get_cache_path(key)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise IOError(f"Erreur sauvegarde cache '{key}': {e}")
    
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Charge données depuis pickle.
        
        Args:
            key: Identifiant du cache
            
        Returns:
            Données chargées ou None si inexistant/corrompu
        """
        cache_file = self._get_cache_path(key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            # Cache corrompu: on le supprime et retourne None (recovery automatique)
            cache_file.unlink()
            return None
        except Exception as e:
            raise IOError(f"Erreur chargement cache '{key}': {e}")
    
    def exists(self, key: str) -> bool:
        """Vérifie existence d'une clé cache.
        
        Args:
            key: Identifiant du cache
            
        Returns:
            True si existe, False sinon
        """
        cache_file = self._get_cache_path(key)
        return cache_file.exists()
    
    def delete(self, key: str) -> None:
        """Supprime une entrée cache.
        
        Args:
            key: Identifiant du cache
        """
        cache_file = self._get_cache_path(key)
        if cache_file.exists():
            cache_file.unlink()
    
    def _get_cache_path(self, key: str) -> Path:
        """Détermine le chemin du fichier cache basé sur la clé.
        
        Implémente Innovation 4 (Dual Cache System):
        - Clés commençant par "baseline_" → baseline/
        - Autres clés → rl/
        
        Args:
            key: Identifiant du cache
            
        Returns:
            Chemin complet du fichier cache
        """
        if key.startswith("baseline_"):
            return self.baseline_dir / f"{key}.pkl"
        else:
            return self.rl_dir / f"{key}.pkl"
