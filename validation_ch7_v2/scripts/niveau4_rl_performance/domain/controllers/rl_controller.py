"""
RLController - Entraînement RL avec Code_RL Integration

✅ CORRECTION ARCHITECTURALE: Utilise Code_RL comme source de vérité
- Réutilise TrafficSignalEnvDirect (validé sur Kaggle)
- Réutilise train_dqn.py (testé et fonctionnel)
- Préserve Bug #6, #7, #27 fixes

Implémente Innovation 3: Sérialisation État Controller
Permet reprise entraînement sans recommencer (~2-3h gagnées).
"""

from typing import Dict, Any, Optional
from pathlib import Path

from domain.interfaces import Logger


class RLController:
    """
    Controller RL utilisant Code_RL adapters (Infrastructure Layer).
    
    Architecture Pattern: Dependency Injection
    - training_adapter: CodeRLTrainingAdapter (injected)
    - logger: Logger (injected)
    
    This controller delegates ALL RL operations to Code_RL adapters,
    ensuring 100% code reuse and 0% duplication.
    
    Attributes:
        training_adapter: CodeRLTrainingAdapter from Infrastructure
        logger: Structured logger
        training_state: Dict tracking training progress
    """
    
    def __init__(
        self,
        training_adapter,  # CodeRLTrainingAdapter from infrastructure.rl
        logger: Logger
    ):
        """Initialise RL controller avec dependency injection.
        
        Args:
            training_adapter: CodeRLTrainingAdapter (Infrastructure Layer)
            logger: Logger structuré (Infrastructure dependency)
        """
        self.training_adapter = training_adapter
        self.logger = logger
        
        # État entraînement
        self.training_state = {
            "initialized": False,
            "total_timesteps": 0,
            "episodes_completed": 0,
            "best_reward": float('-inf'),
            "trained_model": None
        }
        
        self.logger.info(
            "rl_controller_initialized",
            adapter="CodeRLTrainingAdapter (Code_RL integration)"
        )
    
    def train(
        self,
        env_adapter,  # BeninTrafficEnvironmentAdapter
        algorithm: str,
        hyperparameters: Dict[str, Any],
        total_timesteps: int,
        checkpoint_dir: str,
        model_name: str = "rl_model"
    ) -> Dict[str, Any]:
        """
        Entraîne modèle RL en utilisant Code_RL training adapter.
        
        Workflow:
        1. Délégation à training_adapter.train() (Code_RL)
        2. Logging métriques entraînement
        3. Mise à jour training_state
        
        Args:
            env_adapter: BeninTrafficEnvironmentAdapter (Infrastructure)
            algorithm: 'dqn', 'ppo', or 'a2c'
            hyperparameters: Dict with RL hyperparameters
            total_timesteps: Total timesteps to train
            checkpoint_dir: Directory for checkpoints
            model_name: Base name for model
        
        Returns:
            Training results dict
        """
        self.logger.info(
            "rl_training_delegating_to_code_rl",
            algorithm=algorithm,
            total_timesteps=total_timesteps
        )
        
        # Delegate to Code_RL training adapter
        trained_model = self.training_adapter.train(
            env=env_adapter,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            total_timesteps=total_timesteps,
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            resume_training=True  # Always try to resume
        )
        
        # Update training state
        self.training_state["initialized"] = True
        self.training_state["total_timesteps"] = total_timesteps
        self.training_state["trained_model"] = trained_model
        
        self.logger.info(
            "rl_training_complete_via_code_rl",
            total_timesteps=total_timesteps,
            algorithm=algorithm
        )
        
        return {
            "total_timesteps": total_timesteps,
            "algorithm": algorithm,
            "model": trained_model
        }
    
    def evaluate(
        self,
        env_adapter,  # BeninTrafficEnvironmentAdapter
        n_eval_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Évalue performance modèle entraîné.
        
        Args:
            env_adapter: BeninTrafficEnvironmentAdapter
            n_eval_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation metrics (mean_reward, std_reward)
        """
        if not self.training_state["initialized"]:
            raise RuntimeError(
                "Modèle non entraîné. Appeler train() d'abord."
            )
        
        model = self.training_state.get("trained_model")
        if model is None:
            raise RuntimeError("Trained model not found in training_state")
        
        # Delegate to Code_RL training adapter
        eval_results = self.training_adapter.evaluate(
            model=model,
            env=env_adapter,
            n_eval_episodes=n_eval_episodes
        )
        
        # Update best reward
        mean_reward = eval_results["mean_reward"]
        if mean_reward > self.training_state["best_reward"]:
            self.training_state["best_reward"] = mean_reward
        
        self.logger.info(
            "rl_evaluation_complete",
            mean_reward=mean_reward,
            std_reward=eval_results["std_reward"],
            best_reward=self.training_state["best_reward"]
        )
        
        return eval_results
    
    def get_state(self) -> Dict[str, Any]:
        """Récupère état complet controller (Innovation 3).
        
        Returns:
            État sérialisable pour cache (EXCLUDING trained model)
        """
        return {
            "adapter": "CodeRLTrainingAdapter",
            "training_state": {
                "initialized": self.training_state["initialized"],
                "total_timesteps": self.training_state["total_timesteps"],
                "episodes_completed": self.training_state["episodes_completed"],
                "best_reward": self.training_state["best_reward"]
                # Note: trained_model NOT serialized (use checkpoints instead)
            },
            "version": "2.0.0"  # New version with Code_RL integration
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Restaure état controller depuis cache (Innovation 3).
        
        Args:
            state: État sérialisé depuis cache
        """
        training_state_cached = state["training_state"]
        self.training_state["initialized"] = training_state_cached["initialized"]
        self.training_state["total_timesteps"] = training_state_cached["total_timesteps"]
        self.training_state["episodes_completed"] = training_state_cached["episodes_completed"]
        self.training_state["best_reward"] = training_state_cached["best_reward"]
        
        self.logger.info(
            "rl_state_loaded",
            version=state.get("version", "unknown"),
            adapter=state.get("adapter", "unknown"),
            total_timesteps=self.training_state["total_timesteps"],
            best_reward=self.training_state["best_reward"]
        )
