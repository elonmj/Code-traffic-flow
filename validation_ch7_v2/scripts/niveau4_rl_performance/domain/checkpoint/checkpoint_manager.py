"""
CheckpointManager - Gestion des checkpoints RL avec config-hashing et rotation

Implémente Innovation 2: Config-Hashing Checkpoints
Garantit que checkpoints ne sont jamais réutilisés avec config incompatible.

Implémente Innovation 5: Checkpoint Rotation
Conserve uniquement les N derniers checkpoints (gain: 50-70% espace disque).
"""

from typing import Optional, List, Any
from pathlib import Path

from domain.checkpoint.config_hasher import ConfigHasher
from domain.interfaces import CheckpointStorage, Logger


class CheckpointManager:
    """Gestionnaire de checkpoints RL avec hashing config et rotation."""
    
    def __init__(
        self,
        checkpoint_storage: CheckpointStorage,
        logger: Logger,
        checkpoints_dir: Path,
        keep_last: int = 3
    ):
        """Initialise le checkpoint manager.
        
        Args:
            checkpoint_storage: Storage pour sauvegarder/charger checkpoints - INJECTION
            logger: Logger structuré - INJECTION
            checkpoints_dir: Répertoire racine des checkpoints
            keep_last: Nombre de checkpoints à conserver par config (Innovation 5)
        """
        self.storage = checkpoint_storage
        self.logger = logger
        self.checkpoints_dir = Path(checkpoints_dir)
        self.keep_last = keep_last
        
        # Création répertoire checkpoints
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    def save_with_rotation(
        self,
        model: Any,
        config: dict,
        iteration: int
    ) -> Path:
        """Sauvegarde checkpoint avec rotation automatique (Innovations 2 + 5).
        
        Args:
            model: Modèle RL (Stable-Baselines3)
            config: Configuration RL complète
            iteration: Numéro d'itération
            
        Returns:
            Chemin du checkpoint sauvegardé
        """
        # 1. Calcul hash config (Innovation 2)
        config_hash = ConfigHasher.compute_hash(config)
        
        # 2. Génération nom checkpoint avec hash
        checkpoint_name = f"rl_model_{config_hash}_iter{iteration}.zip"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        
        try:
            # 3. Sauvegarde checkpoint
            self.storage.save_checkpoint(checkpoint_path, model)
            
            self.logger.info(
                "checkpoint_saved",
                config_hash=config_hash,
                iteration=iteration,
                checkpoint_path=str(checkpoint_path)
            )
            
            # 4. Rotation automatique (Innovation 5)
            self._rotate_checkpoints(config_hash)
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(
                "checkpoint_save_failed",
                config_hash=config_hash,
                iteration=iteration,
                error=str(e)
            )
            raise
    
    def load_if_compatible(
        self,
        config: dict,
        env: Any
    ) -> Optional[Any]:
        """Charge checkpoint uniquement si config compatible (Innovation 2).
        
        Args:
            config: Configuration RL actuelle
            env: Environnement Gymnasium
            
        Returns:
            Modèle chargé ou None si aucun checkpoint compatible
        """
        # 1. Calcul hash config actuelle
        config_hash = ConfigHasher.compute_hash(config)
        
        # 2. Recherche checkpoints avec même hash (compatibles)
        pattern = f"rl_model_{config_hash}_iter*.zip"
        compatible_checkpoints = self.storage.list_checkpoints(pattern)
        
        if not compatible_checkpoints:
            self.logger.info(
                "no_compatible_checkpoint_found",
                config_hash=config_hash,
                pattern=pattern
            )
            return None
        
        # 3. Chargement du checkpoint le plus récent
        latest_checkpoint = sorted(
            compatible_checkpoints,
            key=lambda p: self._extract_iteration(p)
        )[-1]
        
        try:
            model = self.storage.load_checkpoint(latest_checkpoint, env)
            
            self.logger.info(
                "checkpoint_loaded",
                config_hash=config_hash,
                checkpoint_path=str(latest_checkpoint),
                iteration=self._extract_iteration(latest_checkpoint)
            )
            
            return model
            
        except Exception as e:
            self.logger.error(
                "checkpoint_load_failed",
                config_hash=config_hash,
                checkpoint_path=str(latest_checkpoint),
                error=str(e)
            )
            return None
    
    def _rotate_checkpoints(self, config_hash: str) -> None:
        """Rotation des checkpoints: garde uniquement les N derniers (Innovation 5).
        
        Politique de rétention (keep_last=3):
        - Checkpoint N (dernier): Modèle le plus récent
        - Checkpoint N-1: Backup en cas corruption
        - Checkpoint N-2: Analyse régression
        
        Args:
            config_hash: Hash de la configuration
        """
        # Liste tous les checkpoints pour cette config
        pattern = f"rl_model_{config_hash}_iter*.zip"
        checkpoints = self.storage.list_checkpoints(pattern)
        
        # Tri par numéro d'itération
        checkpoints_sorted = sorted(
            checkpoints,
            key=lambda p: self._extract_iteration(p)
        )
        
        # Calcul nombre à supprimer
        num_to_delete = max(0, len(checkpoints_sorted) - self.keep_last)
        
        if num_to_delete > 0:
            # Suppression des plus anciens
            for checkpoint in checkpoints_sorted[:num_to_delete]:
                self.storage.delete_checkpoint(checkpoint)
                
                self.logger.info(
                    "checkpoint_deleted_rotation",
                    config_hash=config_hash,
                    checkpoint_path=str(checkpoint),
                    iteration=self._extract_iteration(checkpoint)
                )
        
        self.logger.info(
            "checkpoint_rotation_complete",
            config_hash=config_hash,
            checkpoints_kept=len(checkpoints_sorted) - num_to_delete,
            checkpoints_deleted=num_to_delete
        )
    
    @staticmethod
    def _extract_iteration(checkpoint_path: Path) -> int:
        """Extrait le numéro d'itération depuis le nom du checkpoint.
        
        Args:
            checkpoint_path: Chemin du checkpoint (ex: "rl_model_a3f7b2c1_iter5.zip")
            
        Returns:
            Numéro d'itération
        """
        # Extraction depuis pattern "rl_model_{hash}_iter{iteration}.zip"
        stem = checkpoint_path.stem  # "rl_model_a3f7b2c1_iter5"
        iteration_str = stem.split('_iter')[1]  # "5"
        return int(iteration_str)
