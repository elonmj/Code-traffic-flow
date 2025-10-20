"""
Tests unitaires CacheManager

Démonstration testabilité Clean Architecture:
- Mocks pour CacheStorage et Logger (interfaces)
- Tests rapides (<1s chacun)
- Isolation complète (pas de fichiers réels)
"""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path

# Import du module à tester
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domain.cache.cache_manager import CacheManager


class TestCacheManager:
    """Tests unitaires CacheManager."""
    
    @pytest.fixture
    def mock_cache_storage(self):
        """Mock CacheStorage interface."""
        storage = Mock()
        storage.exists.return_value = False
        storage.load.return_value = None
        return storage
    
    @pytest.fixture
    def mock_logger(self):
        """Mock Logger interface."""
        logger = Mock()
        return logger
    
    @pytest.fixture
    def cache_manager(self, mock_cache_storage, mock_logger):
        """Instance CacheManager avec mocks injectés."""
        return CacheManager(
            cache_storage=mock_cache_storage,
            logger=mock_logger
        )
    
    def test_save_baseline_success(self, cache_manager, mock_cache_storage, mock_logger):
        """Test sauvegarde baseline cache avec succès."""
        # Given
        scenario_name = "test_scenario"
        baseline_data = {
            "metrics": {"average_travel_time": 300},
            "controller_state": {"simulation_step": 100}
        }
        
        # When
        cache_manager.save_baseline(scenario_name, baseline_data)
        
        # Then
        mock_cache_storage.save.assert_called_once_with(
            f"baseline_{scenario_name}",
            baseline_data
        )
        mock_logger.info.assert_called_with(
            "cache_baseline_saved",
            scenario=scenario_name
        )
    
    def test_load_baseline_cache_hit(self, cache_manager, mock_cache_storage, mock_logger):
        """Test chargement baseline cache - cache hit."""
        # Given
        scenario_name = "test_scenario"
        cached_data = {
            "metrics": {"average_travel_time": 300},
            "controller_state": {"simulation_step": 100}
        }
        mock_cache_storage.exists.return_value = True
        mock_cache_storage.load.return_value = cached_data
        
        # When
        result = cache_manager.load_baseline(scenario_name)
        
        # Then
        assert result == cached_data
        mock_cache_storage.load.assert_called_once_with(f"baseline_{scenario_name}")
        mock_logger.info.assert_called_with(
            "cache_baseline_hit",
            scenario=scenario_name
        )
    
    def test_load_baseline_cache_miss(self, cache_manager, mock_cache_storage, mock_logger):
        """Test chargement baseline cache - cache miss."""
        # Given
        scenario_name = "test_scenario"
        mock_cache_storage.exists.return_value = False
        
        # When
        result = cache_manager.load_baseline(scenario_name)
        
        # Then
        assert result is None
        mock_logger.info.assert_called_with(
            "cache_baseline_miss",
            scenario=scenario_name
        )
    
    def test_save_rl_cache_success(self, cache_manager, mock_cache_storage, mock_logger):
        """Test sauvegarde RL cache avec config hash."""
        # Given
        scenario_name = "test_scenario"
        config_hash = "a3f7b2c1"
        rl_data = {
            "eval_metrics": {"mean_reward": 150},
            "training_metrics": {"total_timesteps": 10000}
        }
        
        # When
        cache_manager.save_rl_cache(scenario_name, config_hash, rl_data)
        
        # Then
        expected_key = f"rl_{scenario_name}_{config_hash}"
        mock_cache_storage.save.assert_called_once_with(expected_key, rl_data)
        mock_logger.info.assert_called_with(
            "cache_rl_saved",
            scenario=scenario_name,
            config_hash=config_hash
        )
    
    def test_load_rl_cache_with_hash(self, cache_manager, mock_cache_storage, mock_logger):
        """Test chargement RL cache avec hash spécifique."""
        # Given
        scenario_name = "test_scenario"
        config_hash = "a3f7b2c1"
        cached_rl_data = {"eval_metrics": {"mean_reward": 150}}
        mock_cache_storage.exists.return_value = True
        mock_cache_storage.load.return_value = cached_rl_data
        
        # When
        result = cache_manager.load_rl_cache(scenario_name, config_hash)
        
        # Then
        assert result == cached_rl_data
        expected_key = f"rl_{scenario_name}_{config_hash}"
        mock_cache_storage.load.assert_called_once_with(expected_key)
        mock_logger.info.assert_called_with(
            "cache_rl_hit",
            scenario=scenario_name,
            config_hash=config_hash
        )
    
    def test_invalidate_baseline(self, cache_manager, mock_cache_storage, mock_logger):
        """Test invalidation cache baseline."""
        # Given
        scenario_name = "test_scenario"
        
        # When
        cache_manager.invalidate_baseline(scenario_name)
        
        # Then
        expected_key = f"baseline_{scenario_name}"
        mock_cache_storage.delete.assert_called_once_with(expected_key)
        mock_logger.info.assert_called_with(
            "cache_baseline_invalidated",
            scenario=scenario_name
        )
    
    def test_save_baseline_invalid_data_missing_metrics(self, cache_manager):
        """Test sauvegarde baseline avec données invalides (metrics manquant)."""
        # Given
        scenario_name = "test_scenario"
        invalid_data = {
            "controller_state": {"simulation_step": 100}
            # Missing "metrics"
        }
        
        # When/Then
        with pytest.raises(ValueError, match="metrics"):
            cache_manager.save_baseline(scenario_name, invalid_data)
    
    def test_save_baseline_invalid_data_missing_controller_state(self, cache_manager):
        """Test sauvegarde baseline avec données invalides (controller_state manquant)."""
        # Given
        scenario_name = "test_scenario"
        invalid_data = {
            "metrics": {"average_travel_time": 300}
            # Missing "controller_state"
        }
        
        # When/Then
        with pytest.raises(ValueError, match="controller_state"):
            cache_manager.save_baseline(scenario_name, invalid_data)
    
    def test_cache_corrupted_recovery(self, cache_manager, mock_cache_storage, mock_logger):
        """Test récupération cache corrompu."""
        # Given
        scenario_name = "test_scenario"
        mock_cache_storage.exists.return_value = True
        mock_cache_storage.load.side_effect = Exception("Cache corrupted")
        
        # When
        result = cache_manager.load_baseline(scenario_name)
        
        # Then
        assert result is None
        mock_logger.error.assert_called_once()
        # Vérifier que l'événement "cache_corrupted" est loggé
        assert mock_logger.error.call_args[0][0] == "cache_corrupted"


# Exécution tests si script lancé directement
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
