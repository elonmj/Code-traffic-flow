"""
Tests unitaires pour CheckpointManager.

Teste la gestion des checkpoints avec config-hashing et rotation automatique.
Innovation 2 (Config-Hashing) + Innovation 5 (Checkpoint Rotation) testés ici.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, call
from domain.checkpoint.checkpoint_manager import CheckpointManager
from domain.checkpoint.config_hasher import ConfigHasher


class TestCheckpointManager:
    """Tests pour CheckpointManager avec rotation et config-hashing."""
    
    @pytest.fixture
    def mock_storage(self):
        """Mock de CheckpointStorage interface."""
        storage = Mock()
        storage.save_checkpoint = Mock()
        storage.load_checkpoint = Mock(return_value=MagicMock())
        storage.list_checkpoints = Mock(return_value=[])
        storage.delete_checkpoint = Mock()
        storage.checkpoint_exists = Mock(return_value=False)
        return storage
    
    @pytest.fixture
    def mock_logger(self):
        """Mock de Logger interface."""
        logger = Mock()
        logger.info = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        return logger
    
    @pytest.fixture
    def config_hasher(self):
        """ConfigHasher réel pour tests."""
        return ConfigHasher()
    
    @pytest.fixture
    def checkpoint_manager(self, mock_storage, mock_logger, config_hasher):
        """Instance de CheckpointManager avec dépendances mockées."""
        return CheckpointManager(
            checkpoint_storage=mock_storage,
            logger=mock_logger,
            checkpoints_dir="checkpoints",
            keep_last=3
        )
    
    # Tests: save_with_rotation
    
    def test_save_with_rotation_success(self, checkpoint_manager, mock_storage, mock_logger, config_hasher):
        """Vérifie que save_with_rotation sauvegarde avec config_hash dans le nom."""
        model = MagicMock()
        config = {"learning_rate": 0.0001, "buffer_size": 100000}
        iteration = 5
        
        config_hash = config_hasher.compute_hash(config)
        expected_filename = f"rl_model_{config_hash}_iter{iteration}.zip"
        expected_path = Path("checkpoints") / expected_filename
        
        checkpoint_manager.save_with_rotation(model, config, iteration)
        
        # Vérifie que storage.save_checkpoint appelé avec bon path
        mock_storage.save_checkpoint.assert_called_once()
        call_args = mock_storage.save_checkpoint.call_args
        assert str(call_args[0][0]) == str(expected_path)
        assert call_args[0][1] == model
        
        # Vérifie logging
        mock_logger.info.assert_called()
        log_calls = [call[1] for call in mock_logger.info.call_args_list]
        assert any("checkpoint_saved" in str(c) for c in log_calls)
    
    def test_save_with_rotation_triggers_rotation(self, checkpoint_manager, mock_storage, mock_logger, config_hasher):
        """Vérifie que _rotate_checkpoints est appelé après sauvegarde."""
        # Simule 3 checkpoints existants
        config = {"learning_rate": 0.0001}
        config_hash = config_hasher.compute_hash(config)
        
        existing_checkpoints = [
            Path(f"checkpoints/rl_model_{config_hash}_iter1.zip"),
            Path(f"checkpoints/rl_model_{config_hash}_iter2.zip"),
            Path(f"checkpoints/rl_model_{config_hash}_iter3.zip"),
        ]
        mock_storage.list_checkpoints.return_value = existing_checkpoints
        
        model = MagicMock()
        checkpoint_manager.save_with_rotation(model, config, iteration=4)
        
        # Vérifie que list_checkpoints appelé (rotation détecte anciens checkpoints)
        mock_storage.list_checkpoints.assert_called()
    
    # Tests: load_if_compatible
    
    def test_load_if_compatible_success(self, checkpoint_manager, mock_storage, mock_logger, config_hasher):
        """Config hash matches → model chargé avec succès."""
        config = {"learning_rate": 0.0001, "buffer_size": 50000}
        config_hash = config_hasher.compute_hash(config)
        env = MagicMock()
        
        # Simule checkpoint existant avec bon hash
        checkpoint_path = Path(f"checkpoints/rl_model_{config_hash}_iter5.zip")
        mock_storage.list_checkpoints.return_value = [checkpoint_path]
        mock_storage.checkpoint_exists.return_value = True
        
        mock_model = MagicMock()
        mock_storage.load_checkpoint.return_value = mock_model
        
        result = checkpoint_manager.load_if_compatible(config, env)
        
        assert result == mock_model
        mock_storage.load_checkpoint.assert_called_once_with(checkpoint_path, env)
        
        # Vérifie logging checkpoint_loaded_compatible
        log_calls = [call[1] for call in mock_logger.info.call_args_list]
        assert any("checkpoint_loaded_compatible" in str(c) for c in log_calls)
    
    def test_load_if_compatible_failure_hash_mismatch(self, checkpoint_manager, mock_storage, mock_logger, config_hasher):
        """Config hash ne matche pas → None retourné."""
        config = {"learning_rate": 0.0001}
        config_hash = config_hasher.compute_hash(config)
        env = MagicMock()
        
        # Simule checkpoint avec hash différent
        different_hash = "xxxxxxxx"
        checkpoint_path = Path(f"checkpoints/rl_model_{different_hash}_iter3.zip")
        mock_storage.list_checkpoints.return_value = [checkpoint_path]
        
        result = checkpoint_manager.load_if_compatible(config, env)
        
        assert result is None
        mock_storage.load_checkpoint.assert_not_called()
        
        # Vérifie warning checkpoint_incompatible_config
        log_calls = [call[1] for call in mock_logger.warning.call_args_list]
        assert any("checkpoint_incompatible_config" in str(c) or "no_compatible_checkpoint" in str(c) for c in log_calls)
    
    def test_load_if_compatible_no_checkpoint_found(self, checkpoint_manager, mock_storage, mock_logger):
        """Aucun checkpoint trouvé → None retourné avec logging."""
        config = {"learning_rate": 0.0003}
        env = MagicMock()
        
        mock_storage.list_checkpoints.return_value = []
        
        result = checkpoint_manager.load_if_compatible(config, env)
        
        assert result is None
        mock_storage.load_checkpoint.assert_not_called()
        
        # Vérifie info no_compatible_checkpoint
        log_calls = [call[1] for call in mock_logger.info.call_args_list]
        assert any("no_compatible_checkpoint" in str(c) for c in log_calls)
    
    # Tests: _rotate_checkpoints
    
    def test_rotate_checkpoints_keeps_last_3(self, checkpoint_manager, mock_storage, mock_logger, config_hasher):
        """Crée 5 checkpoints → vérifie que 2 sont supprimés (garde seulement 3)."""
        config = {"learning_rate": 0.0001}
        config_hash = config_hasher.compute_hash(config)
        
        # Simule 5 checkpoints existants (iterations 1,2,3,4,5)
        checkpoints = [
            Path(f"checkpoints/rl_model_{config_hash}_iter1.zip"),
            Path(f"checkpoints/rl_model_{config_hash}_iter2.zip"),
            Path(f"checkpoints/rl_model_{config_hash}_iter3.zip"),
            Path(f"checkpoints/rl_model_{config_hash}_iter4.zip"),
            Path(f"checkpoints/rl_model_{config_hash}_iter5.zip"),
        ]
        mock_storage.list_checkpoints.return_value = checkpoints
        
        checkpoint_manager._rotate_checkpoints(config_hash)
        
        # Vérifie que 2 checkpoints supprimés (garde iter3, iter4, iter5)
        assert mock_storage.delete_checkpoint.call_count == 2
        
        # Vérifie que iter1 et iter2 supprimés
        deleted_paths = [call[0][0] for call in mock_storage.delete_checkpoint.call_args_list]
        assert checkpoints[0] in deleted_paths  # iter1
        assert checkpoints[1] in deleted_paths  # iter2
    
    def test_rotate_checkpoints_logs_deletions(self, checkpoint_manager, mock_storage, mock_logger, config_hasher):
        """Vérifie que checkpoint_deleted_rotation loggé pour chaque suppression."""
        config_hash = "a1b2c3d4"
        checkpoints = [
            Path(f"checkpoints/rl_model_{config_hash}_iter1.zip"),
            Path(f"checkpoints/rl_model_{config_hash}_iter2.zip"),
            Path(f"checkpoints/rl_model_{config_hash}_iter3.zip"),
            Path(f"checkpoints/rl_model_{config_hash}_iter4.zip"),
        ]
        mock_storage.list_checkpoints.return_value = checkpoints
        
        checkpoint_manager._rotate_checkpoints(config_hash)
        
        # Vérifie 1 suppression (garde 3 derniers)
        assert mock_storage.delete_checkpoint.call_count == 1
        
        # Vérifie logging
        log_calls = [call[1] for call in mock_logger.info.call_args_list]
        assert any("checkpoint_deleted_rotation" in str(c) for c in log_calls)
    
    # Tests: _extract_iteration
    
    def test_extract_iteration_from_filename(self, checkpoint_manager):
        """Parse 'rl_model_a3f7b2c1_iter5.zip' → 5."""
        path = Path("checkpoints/rl_model_a3f7b2c1_iter5.zip")
        iteration = checkpoint_manager._extract_iteration(path)
        assert iteration == 5
    
    def test_extract_iteration_invalid_filename(self, checkpoint_manager):
        """Nom invalide → retourne 0."""
        path = Path("checkpoints/invalid_checkpoint.zip")
        iteration = checkpoint_manager._extract_iteration(path)
        assert iteration == 0
    
    # Tests: Error Handling
    
    def test_checkpoint_save_failed_error_handling(self, checkpoint_manager, mock_storage, mock_logger):
        """Storage.save_checkpoint raise exception → erreur loggée."""
        mock_storage.save_checkpoint.side_effect = Exception("Disk full")
        
        model = MagicMock()
        config = {"learning_rate": 0.0001}
        
        with pytest.raises(Exception, match="Disk full"):
            checkpoint_manager.save_with_rotation(model, config, iteration=1)
        
        # Vérifie que exception loggée
        mock_logger.error.assert_called()
    
    def test_checkpoint_load_failed_error_handling(self, checkpoint_manager, mock_storage, mock_logger, config_hasher):
        """Checkpoint corrompu → None retourné avec logging."""
        config = {"learning_rate": 0.0001}
        config_hash = config_hasher.compute_hash(config)
        env = MagicMock()
        
        checkpoint_path = Path(f"checkpoints/rl_model_{config_hash}_iter3.zip")
        mock_storage.list_checkpoints.return_value = [checkpoint_path]
        mock_storage.checkpoint_exists.return_value = True
        mock_storage.load_checkpoint.side_effect = Exception("Corrupted checkpoint")
        
        result = checkpoint_manager.load_if_compatible(config, env)
        
        assert result is None
        
        # Vérifie error logging
        mock_logger.error.assert_called()
