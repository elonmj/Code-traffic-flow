"""
SB3CheckpointStorage - Storage checkpoints Stable-Baselines3

Wrapper autour mécanisme sauvegarde/chargement Stable-Baselines3.
Implémente interface CheckpointStorage pour DIP (Dependency Inversion).
"""

from typing import List, Any, Optional
from pathlib import Path
import glob

from stable_baselines3.common.base_class import BaseAlgorithm

from domain.interfaces import CheckpointStorage


class SB3CheckpointStorage(CheckpointStorage):
    """Storage checkpoints pour Stable-Baselines3 models."""
    
    def __init__(self, checkpoints_dir: Path):
        """Initialise storage avec répertoire checkpoints.
        
        Args:
            checkpoints_dir: Répertoire stockage checkpoints
        """
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        checkpoint_path: Path,
        model: BaseAlgorithm
    ) -> None:
        """Sauvegarde modèle Stable-Baselines3.
        
        Args:
            checkpoint_path: Chemin fichier checkpoint (.zip)
            model: Modèle Stable-Baselines3 à sauvegarder
        """
        # Stable-Baselines3 utilise format .zip
        checkpoint_path = Path(checkpoint_path)
        
        # Création répertoire parent si nécessaire
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde modèle
        # Note: SB3 ajoute automatiquement .zip si pas présent
        model.save(str(checkpoint_path.with_suffix('')))
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        env: Any,
        algorithm_class: Optional[type] = None
    ) -> BaseAlgorithm:
        """Charge modèle Stable-Baselines3.
        
        Args:
            checkpoint_path: Chemin fichier checkpoint (.zip)
            env: Environnement Gymnasium
            algorithm_class: Classe algorithme (DQN, PPO, etc.) - optionnel
            
        Returns:
            Modèle Stable-Baselines3 chargé
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint introuvable: {checkpoint_path}"
            )
        
        # Détection automatique classe algorithme si non fournie
        if algorithm_class is None:
            # Import dynamique basé sur nom fichier ou métadonnées
            # Pour l'instant: nécessite classe explicite
            raise ValueError(
                "algorithm_class requis pour chargement checkpoint. "
                "Détection automatique non implémentée."
            )
        
        # Chargement modèle
        model = algorithm_class.load(
            str(checkpoint_path.with_suffix('')),
            env=env
        )
        
        return model
    
    def list_checkpoints(self, pattern: str = "*.zip") -> List[Path]:
        """Liste checkpoints matchant pattern.
        
        Args:
            pattern: Pattern glob (ex: "rl_model_a3f7b2c1_iter*.zip")
            
        Returns:
            Liste chemins checkpoints
        """
        # Recherche dans répertoire checkpoints
        search_pattern = str(self.checkpoints_dir / pattern)
        checkpoint_files = glob.glob(search_pattern)
        
        return [Path(f) for f in sorted(checkpoint_files)]
    
    def delete_checkpoint(self, checkpoint_path: Path) -> None:
        """Supprime checkpoint.
        
        Args:
            checkpoint_path: Chemin fichier checkpoint à supprimer
        """
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    
    def checkpoint_exists(self, checkpoint_path: Path) -> bool:
        """Vérifie existence checkpoint.
        
        Args:
            checkpoint_path: Chemin fichier checkpoint
            
        Returns:
            True si checkpoint existe
        """
        return Path(checkpoint_path).exists()
