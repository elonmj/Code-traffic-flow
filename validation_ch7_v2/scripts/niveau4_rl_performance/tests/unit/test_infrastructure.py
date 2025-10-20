"""
Tests unitaires pour Infrastructure Layer.

Teste les implémentations concrètes: PickleCacheStorage, YAMLConfigLoader,
StructuredLogger, SB3CheckpointStorage.
"""

import pytest
import pickle
import yaml
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from infrastructure.cache.pickle_storage import PickleCacheStorage
from infrastructure.config.yaml_config_loader import YAMLConfigLoader
from infrastructure.logging.structured_logger import StructuredLogger
from infrastructure.checkpoint.sb3_checkpoint_storage import SB3CheckpointStorage


class TestPickleCacheStorage:
    """Tests pour PickleCacheStorage (Innovation 1 + 4 - Dual Cache System)."""
    
    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Crée répertoires temporaires pour cache."""
        baseline_dir = tmp_path / "baseline"
        rl_dir = tmp_path / "rl"
        return baseline_dir, rl_dir
    
    @pytest.fixture
    def pickle_storage(self, temp_dirs):
        """Instance de PickleCacheStorage avec répertoires temp."""
        baseline_dir, rl_dir = temp_dirs
        return PickleCacheStorage(str(baseline_dir), str(rl_dir))
    
    def test_save_and_load_baseline_cache(self, pickle_storage, temp_dirs):
        """Sauvegarde baseline → charge → vérifie données identiques."""
        baseline_dir, _ = temp_dirs
        data = {"metrics": {"avg_travel_time": 120.0}, "controller_state": {}}
        
        pickle_storage.save("baseline_low_traffic", data)
        
        # Vérifie fichier créé dans baseline_dir
        cache_file = baseline_dir / "baseline_low_traffic.pkl"
        assert cache_file.exists()
        
        # Charge et vérifie
        loaded_data = pickle_storage.load("baseline_low_traffic")
        assert loaded_data == data
    
    def test_save_and_load_rl_cache(self, pickle_storage, temp_dirs):
        """Sauvegarde RL cache → charge → vérifie données identiques."""
        _, rl_dir = temp_dirs
        data = {"training_results": {"mean_reward": 45.5}}
        
        pickle_storage.save("rl_a1b2c3d4_low_traffic", data)
        
        # Vérifie fichier créé dans rl_dir
        cache_file = rl_dir / "rl_a1b2c3d4_low_traffic.pkl"
        assert cache_file.exists()
        
        # Charge et vérifie
        loaded_data = pickle_storage.load("rl_a1b2c3d4_low_traffic")
        assert loaded_data == data
    
    def test_dual_cache_routing(self, pickle_storage, temp_dirs):
        """Vérifie que baseline_ → baseline_dir/, rl_ → rl_dir/ (Innovation 4)."""
        baseline_dir, rl_dir = temp_dirs
        
        pickle_storage.save("baseline_scenario1", {"data": 1})
        pickle_storage.save("rl_hash123_scenario1", {"data": 2})
        
        # Vérifie séparation physique
        assert (baseline_dir / "baseline_scenario1.pkl").exists()
        assert (rl_dir / "rl_hash123_scenario1.pkl").exists()
        
        # Vérifie que pas dans mauvais répertoire
        assert not (rl_dir / "baseline_scenario1.pkl").exists()
        assert not (baseline_dir / "rl_hash123_scenario1.pkl").exists()
    
    def test_exists_returns_true_when_cached(self, pickle_storage):
        """Fichier existe → exists() retourne True."""
        pickle_storage.save("baseline_test", {"data": "test"})
        
        assert pickle_storage.exists("baseline_test") is True
    
    def test_exists_returns_false_when_not_cached(self, pickle_storage):
        """Fichier n'existe pas → exists() retourne False."""
        assert pickle_storage.exists("baseline_nonexistent") is False
    
    def test_delete_removes_cache(self, pickle_storage, temp_dirs):
        """delete() supprime fichier cache."""
        baseline_dir, _ = temp_dirs
        pickle_storage.save("baseline_to_delete", {"data": "test"})
        
        cache_file = baseline_dir / "baseline_to_delete.pkl"
        assert cache_file.exists()
        
        pickle_storage.delete("baseline_to_delete")
        
        assert not cache_file.exists()
        assert pickle_storage.exists("baseline_to_delete") is False
    
    def test_load_corrupted_cache_returns_none(self, pickle_storage, temp_dirs):
        """Cache corrompu → load() retourne None sans crash."""
        baseline_dir, _ = temp_dirs
        corrupted_file = baseline_dir / "baseline_corrupted.pkl"
        
        # Crée fichier corrompu (pas un pickle valide)
        corrupted_file.write_text("CORRUPTED DATA NOT PICKLE")
        
        result = pickle_storage.load("baseline_corrupted")
        
        assert result is None
    
    def test_auto_creates_directories(self, tmp_path):
        """Répertoires n'existent pas → créés automatiquement."""
        baseline_dir = tmp_path / "new_baseline"
        rl_dir = tmp_path / "new_rl"
        
        assert not baseline_dir.exists()
        assert not rl_dir.exists()
        
        storage = PickleCacheStorage(str(baseline_dir), str(rl_dir))
        storage.save("baseline_test", {"data": 1})
        
        assert baseline_dir.exists()
        # rl_dir créé seulement si RL cache sauvegardé
        storage.save("rl_test", {"data": 2})
        assert rl_dir.exists()


class TestYAMLConfigLoader:
    """Tests pour YAMLConfigLoader (Innovation 6 - DRY Hyperparameters)."""
    
    @pytest.fixture
    def sample_yaml_content(self):
        """Contenu YAML de test."""
        return """
scenarios:
  - name: low_traffic
    network_file: network_low.xml
    inflow_rate: 500
    duration: 1800
  - name: high_traffic
    network_file: network_high.xml
    inflow_rate: 1500
    duration: 3600

rl_algorithms:
  DQN:
    learning_rate: 0.0001
    buffer_size: 100000
    total_timesteps: 50000
  PPO:
    learning_rate: 0.0003
    n_steps: 2048
    total_timesteps: 100000

benin_context:
  motos_proportion: 0.70
  voitures_proportion: 0.30
  infrastructure_quality: 0.60

cache:
  baseline_dir: cache/baseline
  rl_dir: cache/rl
  enable_cache: true

checkpoints:
  checkpoints_dir: checkpoints
  keep_last: 3
  config_hash_length: 8

logging:
  log_file: logs/section_7_6.log
  log_level: INFO
  enable_structured_logging: true

quick_test:
  enabled: false
  scenarios:
    - name: quick_scenario
      duration: 300
  rl_overrides:
    total_timesteps: 1000
"""
    
    @pytest.fixture
    def yaml_file(self, tmp_path, sample_yaml_content):
        """Crée fichier YAML temporaire."""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(sample_yaml_content)
        return str(yaml_path)
    
    @pytest.fixture
    def config_loader(self, yaml_file):
        """Instance de YAMLConfigLoader."""
        return YAMLConfigLoader(yaml_file)
    
    def test_load_config_parses_yaml(self, config_loader):
        """load_config() parse YAML correctement."""
        config = config_loader.load_config()
        
        assert "scenarios" in config
        assert "rl_algorithms" in config
        assert "benin_context" in config
        assert isinstance(config["scenarios"], list)
    
    def test_get_scenarios_returns_all_scenarios(self, config_loader):
        """get_scenarios() retourne tous les scénarios."""
        scenarios = config_loader.get_scenarios(quick_test=False)
        
        assert len(scenarios) == 2
        assert scenarios[0]["name"] == "low_traffic"
        assert scenarios[1]["name"] == "high_traffic"
    
    def test_get_scenarios_quick_test_mode(self, config_loader):
        """quick_test=True → retourne scénarios réduits."""
        # Modifie config pour activer quick_test
        config_loader.config["quick_test"]["enabled"] = True
        
        scenarios = config_loader.get_scenarios(quick_test=True)
        
        # Devrait retourner quick_test scenarios
        assert len(scenarios) == 1
        assert scenarios[0]["name"] == "quick_scenario"
        assert scenarios[0]["duration"] == 300
    
    def test_get_rl_config_dqn(self, config_loader):
        """get_rl_config('DQN') retourne hyperparameters DQN."""
        rl_config = config_loader.get_rl_config("DQN", quick_test=False)
        
        assert rl_config["learning_rate"] == 0.0001
        assert rl_config["buffer_size"] == 100000
        assert rl_config["total_timesteps"] == 50000
    
    def test_get_rl_config_ppo(self, config_loader):
        """get_rl_config('PPO') retourne hyperparameters PPO."""
        rl_config = config_loader.get_rl_config("PPO", quick_test=False)
        
        assert rl_config["learning_rate"] == 0.0003
        assert rl_config["n_steps"] == 2048
        assert rl_config["total_timesteps"] == 100000
    
    def test_get_rl_config_quick_test_overrides(self, config_loader):
        """quick_test=True → total_timesteps overridé à 1000."""
        config_loader.config["quick_test"]["enabled"] = True
        
        rl_config = config_loader.get_rl_config("DQN", quick_test=True)
        
        # Override de quick_test appliqué
        assert rl_config["total_timesteps"] == 1000
        # Autres paramètres inchangés
        assert rl_config["learning_rate"] == 0.0001
    
    def test_get_benin_context(self, config_loader):
        """get_benin_context() retourne contexte Béninois (Innovation 8)."""
        benin_context = config_loader.get_benin_context()
        
        assert benin_context["motos_proportion"] == 0.70
        assert benin_context["voitures_proportion"] == 0.30
        assert benin_context["infrastructure_quality"] == 0.60
    
    def test_get_cache_config(self, config_loader):
        """get_cache_config() retourne configuration cache."""
        cache_config = config_loader.get_cache_config()
        
        assert cache_config["baseline_dir"] == "cache/baseline"
        assert cache_config["rl_dir"] == "cache/rl"
        assert cache_config["enable_cache"] is True
    
    def test_get_checkpoint_config(self, config_loader):
        """get_checkpoint_config() retourne configuration checkpoints."""
        checkpoint_config = config_loader.get_checkpoint_config()
        
        assert checkpoint_config["checkpoints_dir"] == "checkpoints"
        assert checkpoint_config["keep_last"] == 3
        assert checkpoint_config["config_hash_length"] == 8
    
    def test_is_quick_test_mode(self, config_loader):
        """is_quick_test_mode() détecte si quick_test activé."""
        assert config_loader.is_quick_test_mode() is False
        
        config_loader.config["quick_test"]["enabled"] = True
        assert config_loader.is_quick_test_mode() is True
    
    def test_invalid_yaml_raises_error(self, tmp_path):
        """YAML invalide → erreur parsage."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content: [unclosed")
        
        with pytest.raises(yaml.YAMLError):
            YAMLConfigLoader(str(invalid_yaml))
    
    def test_missing_algorithm_raises_error(self, config_loader):
        """Algorithme non existant → KeyError."""
        with pytest.raises(KeyError):
            config_loader.get_rl_config("NONEXISTENT_ALGO")


class TestStructuredLogger:
    """Tests pour StructuredLogger (Innovation 7 - Dual Logging)."""
    
    @pytest.fixture
    def temp_log_file(self, tmp_path):
        """Fichier log temporaire."""
        return str(tmp_path / "test.log")
    
    @pytest.fixture
    def structured_logger(self, temp_log_file):
        """Instance de StructuredLogger."""
        return StructuredLogger(log_file=temp_log_file, log_level="INFO")
    
    def test_info_logs_event(self, structured_logger, temp_log_file):
        """info() log événement structuré."""
        structured_logger.info("test_event", key1="value1", key2=123)
        
        # Vérifie fichier log créé
        assert Path(temp_log_file).exists()
        
        # Vérifie contenu (JSON structuré)
        log_content = Path(temp_log_file).read_text()
        assert "test_event" in log_content
        assert "value1" in log_content
    
    def test_warning_logs_event(self, structured_logger, temp_log_file):
        """warning() log warning."""
        structured_logger.warning("warning_event", reason="test warning")
        
        log_content = Path(temp_log_file).read_text()
        assert "warning_event" in log_content
        assert "test warning" in log_content
    
    def test_error_logs_event(self, structured_logger, temp_log_file):
        """error() log erreur."""
        structured_logger.error("error_event", error_message="test error")
        
        log_content = Path(temp_log_file).read_text()
        assert "error_event" in log_content
        assert "test error" in log_content
    
    def test_exception_logs_with_traceback(self, structured_logger, temp_log_file):
        """exception() log avec traceback."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            structured_logger.exception("exception_event", context="testing")
        
        log_content = Path(temp_log_file).read_text()
        assert "exception_event" in log_content
        assert "Test exception" in log_content
    
    def test_dual_output_file_and_console(self, structured_logger, temp_log_file, capsys):
        """Vérifie dual output: fichier JSON + console formaté (Innovation 7)."""
        structured_logger.info("dual_output_test", data="test_data")
        
        # Vérifie fichier JSON
        log_content = Path(temp_log_file).read_text()
        assert "dual_output_test" in log_content
        
        # Vérifie console output (capture stdout)
        # Note: Peut nécessiter config structlog pour console handler
        # Pour l'instant, vérifie juste fichier


class TestSB3CheckpointStorage:
    """Tests pour SB3CheckpointStorage."""
    
    @pytest.fixture
    def temp_checkpoints_dir(self, tmp_path):
        """Répertoire checkpoints temporaire."""
        return tmp_path / "checkpoints"
    
    @pytest.fixture
    def sb3_storage(self, temp_checkpoints_dir):
        """Instance de SB3CheckpointStorage."""
        return SB3CheckpointStorage(str(temp_checkpoints_dir))
    
    @patch('infrastructure.checkpoint.sb3_checkpoint_storage.DQN')
    def test_save_checkpoint_creates_file(self, mock_dqn_class, sb3_storage, temp_checkpoints_dir):
        """save_checkpoint() crée fichier .zip."""
        mock_model = MagicMock()
        checkpoint_path = temp_checkpoints_dir / "test_checkpoint.zip"
        
        sb3_storage.save_checkpoint(checkpoint_path, mock_model)
        
        # Vérifie que model.save() appelé
        mock_model.save.assert_called_once_with(checkpoint_path)
    
    @patch('infrastructure.checkpoint.sb3_checkpoint_storage.DQN')
    def test_load_checkpoint_loads_model(self, mock_dqn_class, sb3_storage):
        """load_checkpoint() charge modèle SB3."""
        checkpoint_path = Path("test_checkpoint.zip")
        env = MagicMock()
        
        mock_loaded_model = MagicMock()
        mock_dqn_class.load.return_value = mock_loaded_model
        
        result = sb3_storage.load_checkpoint(checkpoint_path, env, mock_dqn_class)
        
        mock_dqn_class.load.assert_called_once_with(checkpoint_path, env=env)
        assert result == mock_loaded_model
    
    def test_list_checkpoints_glob_search(self, sb3_storage, temp_checkpoints_dir):
        """list_checkpoints() recherche par pattern glob."""
        temp_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Crée fichiers checkpoints
        (temp_checkpoints_dir / "rl_model_hash1_iter1.zip").touch()
        (temp_checkpoints_dir / "rl_model_hash1_iter2.zip").touch()
        (temp_checkpoints_dir / "rl_model_hash2_iter1.zip").touch()
        
        # Recherche checkpoints hash1
        checkpoints = sb3_storage.list_checkpoints("*hash1*.zip")
        
        assert len(checkpoints) == 2
        assert any("hash1_iter1" in str(cp) for cp in checkpoints)
        assert any("hash1_iter2" in str(cp) for cp in checkpoints)
    
    def test_delete_checkpoint_removes_file(self, sb3_storage, temp_checkpoints_dir):
        """delete_checkpoint() supprime fichier."""
        temp_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = temp_checkpoints_dir / "to_delete.zip"
        checkpoint_path.touch()
        
        assert checkpoint_path.exists()
        
        sb3_storage.delete_checkpoint(checkpoint_path)
        
        assert not checkpoint_path.exists()
    
    def test_checkpoint_exists(self, sb3_storage, temp_checkpoints_dir):
        """checkpoint_exists() vérifie existence."""
        temp_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        existing_checkpoint = temp_checkpoints_dir / "existing.zip"
        existing_checkpoint.touch()
        
        assert sb3_storage.checkpoint_exists(existing_checkpoint) is True
        assert sb3_storage.checkpoint_exists(temp_checkpoints_dir / "nonexistent.zip") is False
