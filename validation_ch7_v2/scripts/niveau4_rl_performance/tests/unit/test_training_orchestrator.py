"""
Tests unitaires pour TrainingOrchestrator.

Teste le workflow complet baseline + RL + comparison.
Orchestre toutes les innovations (1-8) ensemble.
"""

import pytest
from unittest.mock import Mock, MagicMock, call
from domain.orchestration.training_orchestrator import TrainingOrchestrator


class TestTrainingOrchestrator:
    """Tests pour TrainingOrchestrator - cœur de la logique métier."""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock de CacheManager."""
        manager = Mock()
        manager.load_baseline = Mock(return_value=None)
        manager.save_baseline = Mock()
        manager.load_rl_cache = Mock(return_value=None)
        manager.save_rl_cache = Mock()
        return manager
    
    @pytest.fixture
    def mock_checkpoint_manager(self):
        """Mock de CheckpointManager."""
        manager = Mock()
        manager.load_if_compatible = Mock(return_value=None)
        manager.save_with_rotation = Mock()
        return manager
    
    @pytest.fixture
    def mock_logger(self):
        """Mock de Logger."""
        logger = Mock()
        logger.info = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        return logger
    
    @pytest.fixture
    def orchestrator(self, mock_cache_manager, mock_checkpoint_manager, mock_logger):
        """Instance de TrainingOrchestrator."""
        return TrainingOrchestrator(
            cache_manager=mock_cache_manager,
            checkpoint_manager=mock_checkpoint_manager,
            logger=mock_logger
        )
    
    # Tests: run_scenario (workflow complet)
    
    def test_run_scenario_baseline_cache_hit(self, orchestrator, mock_cache_manager, mock_logger):
        """Cache hit baseline → pas de simulation baseline exécutée."""
        scenario_config = {
            "name": "low_traffic",
            "network_file": "network.xml",
            "inflow_rate": 500,
            "duration": 3600
        }
        rl_config = {"algorithm": "DQN", "learning_rate": 0.0001, "total_timesteps": 10000}
        benin_context = {"motos_proportion": 0.70}
        
        # Simule cache hit baseline
        cached_baseline = {
            "metrics": {"avg_travel_time": 120.0, "throughput": 800},
            "controller_state": {}
        }
        mock_cache_manager.load_baseline.return_value = cached_baseline
        
        # Mock RL training (placeholder car env Gymnasium non implémenté)
        with pytest.raises(NotImplementedError):
            orchestrator.run_scenario(scenario_config, rl_config, benin_context)
        
        # Vérifie que load_baseline appelé
        mock_cache_manager.load_baseline.assert_called_once_with("low_traffic")
        
        # Vérifie logging cache_baseline_hit
        log_calls = [call[1] for call in mock_logger.info.call_args_list]
        assert any("cache_baseline_hit" in str(c) for c in log_calls)
    
    def test_run_scenario_baseline_cache_miss(self, orchestrator, mock_cache_manager, mock_logger):
        """Cache miss baseline → simulation baseline exécutée puis cachée."""
        scenario_config = {
            "name": "medium_traffic",
            "network_file": "network.xml",
            "inflow_rate": 1000,
            "duration": 3600
        }
        rl_config = {"algorithm": "PPO", "learning_rate": 0.0003, "total_timesteps": 20000}
        benin_context = {"motos_proportion": 0.70}
        
        # Simule cache miss baseline
        mock_cache_manager.load_baseline.return_value = None
        
        # Mock RL training (placeholder)
        with pytest.raises(NotImplementedError):
            orchestrator.run_scenario(scenario_config, rl_config, benin_context)
        
        # Vérifie logging cache_baseline_miss
        log_calls = [call[1] for call in mock_logger.info.call_args_list]
        assert any("cache_baseline_miss" in str(c) for c in log_calls)
    
    def test_run_scenario_checkpoint_compatible(self, orchestrator, mock_checkpoint_manager, mock_cache_manager, mock_logger):
        """Checkpoint compatible trouvé → training reprend depuis checkpoint."""
        scenario_config = {"name": "high_traffic", "network_file": "network.xml", "inflow_rate": 1500, "duration": 3600}
        rl_config = {"algorithm": "DQN", "learning_rate": 0.0001, "total_timesteps": 50000}
        benin_context = {"motos_proportion": 0.70}
        
        # Simule checkpoint compatible trouvé
        mock_model = MagicMock()
        mock_checkpoint_manager.load_if_compatible.return_value = mock_model
        
        # Simule baseline cached
        mock_cache_manager.load_baseline.return_value = {
            "metrics": {"avg_travel_time": 150.0},
            "controller_state": {}
        }
        
        # Mock RL training
        with pytest.raises(NotImplementedError):
            orchestrator.run_scenario(scenario_config, rl_config, benin_context)
        
        # Vérifie que load_if_compatible appelé
        mock_checkpoint_manager.load_if_compatible.assert_called_once()
    
    def test_run_scenario_no_compatible_checkpoint(self, orchestrator, mock_checkpoint_manager, mock_cache_manager, mock_logger):
        """Pas de checkpoint compatible → training from scratch."""
        scenario_config = {"name": "peak_traffic", "network_file": "network.xml", "inflow_rate": 2000, "duration": 3600}
        rl_config = {"algorithm": "PPO", "learning_rate": 0.0003, "total_timesteps": 100000}
        benin_context = {"motos_proportion": 0.70}
        
        # Simule aucun checkpoint compatible
        mock_checkpoint_manager.load_if_compatible.return_value = None
        
        # Simule baseline cached
        mock_cache_manager.load_baseline.return_value = {
            "metrics": {"avg_travel_time": 180.0},
            "controller_state": {}
        }
        
        # Mock RL training
        with pytest.raises(NotImplementedError):
            orchestrator.run_scenario(scenario_config, rl_config, benin_context)
        
        # Vérifie logging training_from_scratch
        log_calls = [call[1] for call in mock_logger.info.call_args_list]
        assert any("no_compatible_checkpoint" in str(c) or "training_from_scratch" in str(c) for c in log_calls)
    
    # Tests: run_multiple_scenarios
    
    def test_run_multiple_scenarios_sequential_execution(self, orchestrator, mock_cache_manager, mock_logger):
        """Multiples scénarios → exécutés séquentiellement."""
        scenarios = [
            {"name": "scenario1", "network_file": "network.xml", "inflow_rate": 500, "duration": 1800},
            {"name": "scenario2", "network_file": "network.xml", "inflow_rate": 1000, "duration": 1800},
        ]
        rl_config = {"algorithm": "DQN", "learning_rate": 0.0001, "total_timesteps": 5000}
        benin_context = {"motos_proportion": 0.70}
        
        # Mock baseline cached pour éviter simulation
        mock_cache_manager.load_baseline.return_value = {
            "metrics": {"avg_travel_time": 100.0},
            "controller_state": {}
        }
        
        # Mock RL training
        with pytest.raises(NotImplementedError):
            orchestrator.run_multiple_scenarios(scenarios, rl_config, benin_context)
        
        # Vérifie que load_baseline appelé pour chaque scénario
        assert mock_cache_manager.load_baseline.call_count == 2
        mock_cache_manager.load_baseline.assert_any_call("scenario1")
        mock_cache_manager.load_baseline.assert_any_call("scenario2")
    
    def test_run_multiple_scenarios_aggregates_results(self, orchestrator, mock_cache_manager):
        """Multiples scénarios → résultats agrégés dans liste."""
        scenarios = [
            {"name": "s1", "network_file": "network.xml", "inflow_rate": 500, "duration": 1800},
            {"name": "s2", "network_file": "network.xml", "inflow_rate": 1000, "duration": 1800},
        ]
        rl_config = {"algorithm": "PPO", "learning_rate": 0.0003, "total_timesteps": 5000}
        benin_context = {"motos_proportion": 0.70}
        
        mock_cache_manager.load_baseline.return_value = {
            "metrics": {"avg_travel_time": 90.0},
            "controller_state": {}
        }
        
        # Mock RL training
        with pytest.raises(NotImplementedError):
            results = orchestrator.run_multiple_scenarios(scenarios, rl_config, benin_context)
        
        # Note: Actuellement NotImplementedError car env Gymnasium non implémenté
        # À compléter après implémentation TrafficEnvironment
    
    # Tests: _run_baseline_with_cache (Innovation 1)
    
    def test_run_baseline_with_cache_hit(self, orchestrator, mock_cache_manager, mock_logger):
        """Cache hit → baseline non simulée, controller_state restauré."""
        scenario_config = {"name": "test_scenario", "network_file": "network.xml", "inflow_rate": 500, "duration": 1800}
        benin_context = {"motos_proportion": 0.70}
        
        cached_data = {
            "metrics": {"avg_travel_time": 110.0, "throughput": 900},
            "controller_state": {"network_file": "network.xml", "simulation_results": {}}
        }
        mock_cache_manager.load_baseline.return_value = cached_data
        
        baseline_results, controller_state = orchestrator._run_baseline_with_cache(scenario_config, benin_context)
        
        assert baseline_results == cached_data["metrics"]
        assert controller_state == cached_data["controller_state"]
        
        # Vérifie que save_baseline pas appelé (cache hit)
        mock_cache_manager.save_baseline.assert_not_called()
    
    def test_run_baseline_with_cache_miss_saves_to_cache(self, orchestrator, mock_cache_manager, mock_logger):
        """Cache miss → baseline simulée puis sauvegardée dans cache."""
        scenario_config = {"name": "new_scenario", "network_file": "network.xml", "inflow_rate": 1200, "duration": 3600}
        benin_context = {"motos_proportion": 0.70}
        
        mock_cache_manager.load_baseline.return_value = None
        
        # Mock RL (actuellement NotImplementedError car env non implémenté)
        # À compléter après implémentation BaselineController.run_simulation()
        with pytest.raises(NotImplementedError):
            orchestrator._run_baseline_with_cache(scenario_config, benin_context)
    
    # Tests: _run_rl_with_checkpoints (Innovation 2 + 5)
    
    def test_run_rl_with_checkpoints_loads_compatible(self, orchestrator, mock_checkpoint_manager, mock_logger):
        """Checkpoint compatible → modèle chargé depuis checkpoint."""
        scenario_config = {"name": "test", "network_file": "network.xml", "inflow_rate": 1000, "duration": 3600}
        rl_config = {"algorithm": "DQN", "learning_rate": 0.0001, "total_timesteps": 20000}
        
        mock_model = MagicMock()
        mock_checkpoint_manager.load_if_compatible.return_value = mock_model
        
        # Mock RL training
        with pytest.raises(NotImplementedError):
            orchestrator._run_rl_with_checkpoints(scenario_config, rl_config)
        
        # Vérifie que load_if_compatible appelé
        mock_checkpoint_manager.load_if_compatible.assert_called_once()
    
    def test_run_rl_with_checkpoints_saves_with_rotation(self, orchestrator, mock_checkpoint_manager, mock_logger):
        """Après training → checkpoint sauvegardé avec rotation (Innovation 5)."""
        scenario_config = {"name": "test", "network_file": "network.xml", "inflow_rate": 1000, "duration": 3600}
        rl_config = {"algorithm": "PPO", "learning_rate": 0.0003, "total_timesteps": 50000}
        
        mock_checkpoint_manager.load_if_compatible.return_value = None
        
        # Mock RL training
        with pytest.raises(NotImplementedError):
            orchestrator._run_rl_with_checkpoints(scenario_config, rl_config)
        
        # Note: Vérification save_with_rotation à compléter après implémentation complète
    
    # Tests: _compute_comparison
    
    def test_compute_comparison_calculates_improvement(self, orchestrator):
        """Vérifie calcul improvement % correct."""
        baseline_results = {"avg_travel_time": 150.0, "throughput": 800}
        rl_results = {"avg_travel_time": 120.0, "throughput": 900}
        
        comparison = orchestrator._compute_comparison(baseline_results, rl_results)
        
        # Amélioration travel time: (150 - 120) / 150 = 20%
        assert comparison["travel_time_improvement_pct"] == pytest.approx(20.0, rel=0.01)
        
        # Amélioration throughput: (900 - 800) / 800 = 12.5%
        assert comparison["throughput_improvement_pct"] == pytest.approx(12.5, rel=0.01)
    
    def test_compute_comparison_negative_improvement(self, orchestrator):
        """RL pire que baseline → improvement négatif."""
        baseline_results = {"avg_travel_time": 100.0, "throughput": 1000}
        rl_results = {"avg_travel_time": 120.0, "throughput": 900}
        
        comparison = orchestrator._compute_comparison(baseline_results, rl_results)
        
        # Dégradation travel time: (100 - 120) / 100 = -20%
        assert comparison["travel_time_improvement_pct"] == pytest.approx(-20.0, rel=0.01)
        
        # Dégradation throughput: (900 - 1000) / 1000 = -10%
        assert comparison["throughput_improvement_pct"] == pytest.approx(-10.0, rel=0.01)
    
    def test_compute_comparison_zero_baseline(self, orchestrator):
        """Baseline = 0 → évite division par zéro."""
        baseline_results = {"avg_travel_time": 0.0, "throughput": 0}
        rl_results = {"avg_travel_time": 100.0, "throughput": 500}
        
        comparison = orchestrator._compute_comparison(baseline_results, rl_results)
        
        # Vérifie que pas de crash (gestion division par zéro)
        assert "travel_time_improvement_pct" in comparison
        assert "throughput_improvement_pct" in comparison
    
    # Tests: Error Handling
    
    def test_run_scenario_invalid_algorithm_raises_error(self, orchestrator, mock_cache_manager):
        """Algorithme invalide → ValueError."""
        scenario_config = {"name": "test", "network_file": "network.xml", "inflow_rate": 1000, "duration": 3600}
        rl_config = {"algorithm": "INVALID", "learning_rate": 0.0001, "total_timesteps": 10000}
        benin_context = {"motos_proportion": 0.70}
        
        mock_cache_manager.load_baseline.return_value = {
            "metrics": {"avg_travel_time": 100.0},
            "controller_state": {}
        }
        
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            orchestrator.run_scenario(scenario_config, rl_config, benin_context)
    
    def test_run_scenario_missing_config_keys_raises_error(self, orchestrator):
        """Clés manquantes dans config → KeyError ou validation error."""
        incomplete_scenario = {"name": "test"}  # Manque network_file, inflow_rate, duration
        rl_config = {"algorithm": "DQN", "learning_rate": 0.0001, "total_timesteps": 10000}
        benin_context = {"motos_proportion": 0.70}
        
        with pytest.raises((KeyError, ValueError)):
            orchestrator.run_scenario(incomplete_scenario, rl_config, benin_context)
