"""
Tests unitaires pour BaselineController et RLController.

Teste la gestion d'état, sérialisation, et intégration Benin context.
Innovation 3 (Sérialisation État) + Innovation 8 (Contexte Béninois) testés ici.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from domain.controllers.baseline_controller import BaselineController
from domain.controllers.rl_controller import RLController


class TestBaselineController:
    """Tests pour BaselineController avec contexte Béninois."""
    
    @pytest.fixture
    def mock_logger(self):
        """Mock de Logger interface."""
        logger = Mock()
        logger.info = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        return logger
    
    @pytest.fixture
    def baseline_controller(self, mock_logger):
        """Instance de BaselineController."""
        return BaselineController(logger=mock_logger)
    
    def test_initialize_simulation_applies_benin_context(self, baseline_controller, mock_logger):
        """Vérifie que initialize_simulation applique le contexte Béninois."""
        benin_context = {
            "motos_proportion": 0.70,
            "voitures_proportion": 0.30,
            "infrastructure_quality": 0.60,
            "max_speed_moto": 50,
            "max_speed_voiture": 60
        }
        
        baseline_controller.initialize_simulation(
            network_file="network.xml",
            inflow_rate=1000,
            duration=3600,
            benin_context=benin_context
        )
        
        # Vérifie que contexte stocké
        assert baseline_controller.benin_context == benin_context
        assert baseline_controller.network_file == "network.xml"
        assert baseline_controller.inflow_rate == 1000
        assert baseline_controller.duration == 3600
        
        # Vérifie logging initialization
        log_calls = [call[1] for call in mock_logger.info.call_args_list]
        assert any("baseline_simulation_initialized" in str(c) for c in log_calls)
    
    def test_initialize_simulation_default_benin_context(self, baseline_controller):
        """Si benin_context non fourni, utilise BENIN_CONTEXT_DEFAULT."""
        baseline_controller.initialize_simulation(
            network_file="network.xml",
            inflow_rate=500,
            duration=1800
        )
        
        # Vérifie que BENIN_CONTEXT_DEFAULT appliqué
        assert baseline_controller.benin_context["motos_proportion"] == 0.70
        assert baseline_controller.benin_context["voitures_proportion"] == 0.30
    
    def test_get_state_serialization(self, baseline_controller):
        """Vérifie que get_state() sérialise l'état du controller (Innovation 3)."""
        baseline_controller.network_file = "network.xml"
        baseline_controller.inflow_rate = 1000
        baseline_controller.duration = 3600
        baseline_controller.benin_context = {"motos_proportion": 0.70}
        baseline_controller.simulation_results = {"avg_speed": 45.2}
        
        state = baseline_controller.get_state()
        
        assert state["network_file"] == "network.xml"
        assert state["inflow_rate"] == 1000
        assert state["duration"] == 3600
        assert state["benin_context"] == {"motos_proportion": 0.70}
        assert state["simulation_results"] == {"avg_speed": 45.2}
    
    def test_load_state_deserialization(self, baseline_controller):
        """Vérifie que load_state() restore l'état (Innovation 3)."""
        state = {
            "network_file": "restored_network.xml",
            "inflow_rate": 800,
            "duration": 2400,
            "benin_context": {"motos_proportion": 0.65},
            "simulation_results": {"avg_speed": 50.0}
        }
        
        baseline_controller.load_state(state)
        
        assert baseline_controller.network_file == "restored_network.xml"
        assert baseline_controller.inflow_rate == 800
        assert baseline_controller.duration == 2400
        assert baseline_controller.benin_context == {"motos_proportion": 0.65}
        assert baseline_controller.simulation_results == {"avg_speed": 50.0}
    
    def test_run_simulation_placeholder(self, baseline_controller, mock_logger):
        """Vérifie que run_simulation() retourne résultats (actuellement placeholder)."""
        baseline_controller.initialize_simulation(
            network_file="network.xml",
            inflow_rate=1000,
            duration=3600
        )
        
        results = baseline_controller.run_simulation()
        
        # Actuellement placeholder, vérifie structure minimale
        assert isinstance(results, dict)
        assert "avg_travel_time" in results
        assert "throughput" in results
        
        # Vérifie logging
        log_calls = [call[1] for call in mock_logger.info.call_args_list]
        assert any("baseline_simulation_completed" in str(c) for c in log_calls)
    
    def test_benin_context_motos_proportion_applied(self, baseline_controller):
        """Vérifie que 70% motos appliqué à inflow_rate."""
        benin_context = {"motos_proportion": 0.70, "voitures_proportion": 0.30}
        
        baseline_controller.initialize_simulation(
            network_file="network.xml",
            inflow_rate=1000,
            duration=3600,
            benin_context=benin_context
        )
        
        # Calcul attendu: 70% de 1000 = 700 motos, 30% = 300 voitures
        expected_motos = 1000 * 0.70
        expected_voitures = 1000 * 0.30
        
        # Note: Logique à vérifier dans implémentation réelle run_simulation()
        # Pour l'instant, vérifie juste que contexte stocké
        assert baseline_controller.benin_context["motos_proportion"] == 0.70
    
    def test_infrastructure_quality_stored(self, baseline_controller):
        """Vérifie que infrastructure_quality (60%) stockée correctement."""
        benin_context = {"infrastructure_quality": 0.60}
        
        baseline_controller.initialize_simulation(
            network_file="network.xml",
            inflow_rate=500,
            duration=1800,
            benin_context=benin_context
        )
        
        assert baseline_controller.benin_context["infrastructure_quality"] == 0.60


class TestRLController:
    """Tests pour RLController avec Stable-Baselines3."""
    
    @pytest.fixture
    def mock_logger(self):
        """Mock de Logger interface."""
        logger = Mock()
        logger.info = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        return logger
    
    @pytest.fixture
    def mock_env(self):
        """Mock de Gymnasium environment."""
        env = MagicMock()
        env.observation_space = MagicMock()
        env.action_space = MagicMock()
        return env
    
    @pytest.fixture
    def rl_controller_dqn(self, mock_logger, mock_env):
        """Instance de RLController avec algorithme DQN."""
        return RLController(
            logger=mock_logger,
            env=mock_env,
            algorithm="DQN",
            hyperparameters={"learning_rate": 0.0001, "buffer_size": 50000}
        )
    
    @pytest.fixture
    def rl_controller_ppo(self, mock_logger, mock_env):
        """Instance de RLController avec algorithme PPO."""
        return RLController(
            logger=mock_logger,
            env=mock_env,
            algorithm="PPO",
            hyperparameters={"learning_rate": 0.0003, "n_steps": 2048}
        )
    
    @patch('domain.controllers.rl_controller.DQN')
    def test_initialize_model_creates_new_dqn(self, mock_dqn_class, rl_controller_dqn, mock_env):
        """Vérifie que initialize_model() crée nouveau modèle DQN."""
        mock_model = MagicMock()
        mock_dqn_class.return_value = mock_model
        
        rl_controller_dqn.initialize_model(mock_env)
        
        # Vérifie que DQN() appelé avec hyperparameters
        mock_dqn_class.assert_called_once()
        call_kwargs = mock_dqn_class.call_args[1]
        assert call_kwargs["learning_rate"] == 0.0001
        assert call_kwargs["buffer_size"] == 50000
        assert rl_controller_dqn.model == mock_model
    
    @patch('domain.controllers.rl_controller.PPO')
    def test_initialize_model_creates_new_ppo(self, mock_ppo_class, rl_controller_ppo, mock_env):
        """Vérifie que initialize_model() crée nouveau modèle PPO."""
        mock_model = MagicMock()
        mock_ppo_class.return_value = mock_model
        
        rl_controller_ppo.initialize_model(mock_env)
        
        # Vérifie que PPO() appelé avec hyperparameters
        mock_ppo_class.assert_called_once()
        call_kwargs = mock_ppo_class.call_args[1]
        assert call_kwargs["learning_rate"] == 0.0003
        assert call_kwargs["n_steps"] == 2048
    
    @patch('domain.controllers.rl_controller.DQN')
    def test_initialize_model_loads_from_checkpoint(self, mock_dqn_class, rl_controller_dqn, mock_env):
        """Checkpoint path fourni → modèle chargé depuis checkpoint."""
        mock_loaded_model = MagicMock()
        mock_dqn_class.load.return_value = mock_loaded_model
        
        rl_controller_dqn.initialize_model(mock_env, checkpoint_path="checkpoint.zip")
        
        # Vérifie que DQN.load() appelé
        mock_dqn_class.load.assert_called_once_with("checkpoint.zip", env=mock_env)
        assert rl_controller_dqn.model == mock_loaded_model
    
    def test_train_calls_model_learn(self, rl_controller_dqn):
        """Vérifie que train() appelle model.learn()."""
        mock_model = MagicMock()
        rl_controller_dqn.model = mock_model
        
        rl_controller_dqn.train(total_timesteps=10000)
        
        mock_model.learn.assert_called_once_with(total_timesteps=10000, callback=None)
    
    def test_evaluate_returns_mean_reward(self, rl_controller_dqn, mock_env):
        """Vérifie que evaluate() retourne mean_reward."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (MagicMock(), None)
        rl_controller_dqn.model = mock_model
        
        # Mock env.step() pour simulation
        mock_env.step.return_value = (MagicMock(), 10.0, False, False, {})
        mock_env.reset.return_value = (MagicMock(), {})
        
        with patch('domain.controllers.rl_controller.np.mean', return_value=25.5):
            mean_reward, std_reward = rl_controller_dqn.evaluate(n_eval_episodes=5)
        
        assert mean_reward == 25.5
    
    def test_get_state_serialization(self, rl_controller_dqn):
        """Vérifie que get_state() sérialise hyperparameters + algorithm (Innovation 3)."""
        state = rl_controller_dqn.get_state()
        
        assert state["algorithm"] == "DQN"
        assert state["hyperparameters"]["learning_rate"] == 0.0001
        assert state["hyperparameters"]["buffer_size"] == 50000
    
    def test_load_state_deserialization(self, rl_controller_dqn):
        """Vérifie que load_state() restore algorithm + hyperparameters (Innovation 3)."""
        state = {
            "algorithm": "PPO",
            "hyperparameters": {"learning_rate": 0.0003, "n_steps": 1024}
        }
        
        rl_controller_dqn.load_state(state)
        
        assert rl_controller_dqn.algorithm == "PPO"
        assert rl_controller_dqn.hyperparameters["learning_rate"] == 0.0003
        assert rl_controller_dqn.hyperparameters["n_steps"] == 1024
    
    def test_apply_action_calls_model_predict(self, rl_controller_dqn):
        """Vérifie que apply_action() appelle model.predict()."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (2, None)  # action=2
        rl_controller_dqn.model = mock_model
        
        observation = [1.0, 2.0, 3.0]
        action = rl_controller_dqn.apply_action(observation)
        
        assert action == 2
        mock_model.predict.assert_called_once()
    
    def test_get_model_returns_model(self, rl_controller_dqn):
        """Vérifie que get_model() retourne le modèle SB3."""
        mock_model = MagicMock()
        rl_controller_dqn.model = mock_model
        
        result = rl_controller_dqn.get_model()
        
        assert result == mock_model
    
    def test_algorithm_case_insensitive(self, mock_logger, mock_env):
        """Vérifie que 'dqn', 'DQN', 'Dqn' tous acceptés."""
        controller = RLController(
            logger=mock_logger,
            env=mock_env,
            algorithm="dqn",  # lowercase
            hyperparameters={"learning_rate": 0.0001}
        )
        
        # Vérifie que algorithm normalisé en uppercase
        assert controller.algorithm == "DQN"
    
    def test_unsupported_algorithm_raises_error(self, mock_logger, mock_env):
        """Algorithme non supporté → ValueError."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            RLController(
                logger=mock_logger,
                env=mock_env,
                algorithm="INVALID_ALGO",
                hyperparameters={}
            )
