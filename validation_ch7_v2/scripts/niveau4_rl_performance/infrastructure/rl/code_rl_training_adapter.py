"""
Code_RL Training Adapter - Infrastructure Layer

Adapte la boucle d'entraînement validée de Code_RL pour notre workflow
de validation Section 7.6 RL Performance.

Ce module est un WRAPPER autour de train_dqn.py, garantissant:
- ✅ Checkpoint resume intelligent
- ✅ Rotating checkpoint callbacks
- ✅ Training progress tracking
- ✅ Kaggle compatibility

Principe: RÉUTILISER train_dqn.py comme source de vérité, ADAPTER pour notre workflow.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Optional, Any
import time

# Import Code_RL training functions
# Path calculation: file → rl/ → infrastructure/ → niveau4_rl_performance/ → scripts/ → validation_ch7_v2/ → Code project/
CODE_RL_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "Code_RL"
if not CODE_RL_PATH.exists():
    raise RuntimeError(
        f"Code_RL not found at {CODE_RL_PATH}. "
        f"Please ensure Code_RL directory is at project root level."
    )

# Add both Code_RL root and src to path for proper imports
sys.path.insert(0, str(CODE_RL_PATH))
sys.path.insert(0, str(CODE_RL_PATH / "src"))

try:
    from src.rl.train_dqn import train_dqn_agent, find_latest_checkpoint
    from src.rl.callbacks import RotatingCheckpointCallback, TrainingProgressCallback
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CallbackList
except ImportError as e:
    raise ImportError(
        f"Failed to import from Code_RL: {e}. "
        f"Ensure Code_RL/src/rl/ modules exist."
    )


class CodeRLTrainingAdapter:
    """
    Wrapper autour de train_dqn.py qui adapte pour notre workflow Clean Architecture
    tout en préservant 100% de la logique d'entraînement validée sur Kaggle.
    
    Architecture Pattern: Adapter Pattern
    - Delegate: train_dqn_agent (Code_RL)
    - Adaptation: Integration with our CheckpointManager and Logger
    
    Features Preserved (from Code_RL):
    - Intelligent checkpoint resume
    - Rotating checkpoint saves (disk space optimization)
    - Training progress callbacks
    - Kaggle compatibility
    
    Attributes:
        checkpoint_manager: Our domain CheckpointManager (for rotation policy)
        logger: Structured logger (Infrastructure dependency)
    """
    
    def __init__(self, 
                 checkpoint_manager,
                 logger):
        """
        Initialize training adapter.
        
        Args:
            checkpoint_manager: Domain CheckpointManager for rotation policy
            logger: Structured logger interface
        """
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
    
    def train(self,
              env,
              algorithm: str,
              hyperparameters: Dict,
              total_timesteps: int,
              checkpoint_dir: str,
              model_name: str = "rl_model",
              checkpoint_freq: int = 10000,
              resume_training: bool = True) -> DQN:
        """
        Entraîne un agent RL en utilisant train_dqn.py de Code_RL.
        
        Workflow:
        1. Chercher checkpoint existant (si resume_training=True)
        2. Charger modèle existant OU créer nouveau
        3. Calculer timesteps restants
        4. Déléguer à train_dqn_agent (Code_RL)
        5. Logger métriques d'entraînement
        
        Args:
            env: Gymnasium environment (BeninTrafficEnvironmentAdapter)
            algorithm: 'dqn', 'ppo', or 'a2c'
            hyperparameters: Dict with learning_rate, buffer_size, etc.
            total_timesteps: Total timesteps to train
            checkpoint_dir: Directory for checkpoints
            model_name: Base name for model checkpoints
            checkpoint_freq: Save checkpoint every N timesteps
            resume_training: Whether to resume from latest checkpoint
        
        Returns:
            Trained DQN model
        
        Raises:
            ValueError: If algorithm not supported
            RuntimeError: If training fails
        """
        if algorithm.lower() != 'dqn':
            raise ValueError(
                f"Algorithm '{algorithm}' not supported. "
                f"Currently only 'dqn' is supported (via Code_RL/train_dqn.py). "
                f"For PPO/A2C, implement additional adapters."
            )
        
        self.logger.info("rl_training_starting",
                        algorithm=algorithm,
                        total_timesteps=total_timesteps,
                        checkpoint_dir=checkpoint_dir,
                        hyperparameters=hyperparameters)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Step 1: Find latest checkpoint (if resume enabled)
        checkpoint_path = None
        num_timesteps_done = 0
        
        if resume_training:
            checkpoint_path, num_timesteps_done = find_latest_checkpoint(
                checkpoint_dir, model_name
            )
            
            if checkpoint_path:
                self.logger.info("checkpoint_found_resuming",
                               checkpoint_path=checkpoint_path,
                               timesteps_done=num_timesteps_done,
                               timesteps_remaining=total_timesteps - num_timesteps_done)
            else:
                self.logger.info("no_checkpoint_found_training_from_scratch")
        
        # Step 2: Load existing model OR create new
        model = None
        remaining_timesteps = total_timesteps - num_timesteps_done
        
        if checkpoint_path and remaining_timesteps > 0:
            try:
                model = DQN.load(checkpoint_path, env=env)
                self.logger.info("checkpoint_loaded_successfully",
                               checkpoint_path=checkpoint_path)
            except Exception as e:
                self.logger.error("checkpoint_load_failed",
                                error=str(e),
                                checkpoint_path=checkpoint_path)
                # Fallback: train from scratch
                model = None
                remaining_timesteps = total_timesteps
        
        elif remaining_timesteps <= 0:
            # Already trained enough timesteps
            self.logger.info("training_already_complete",
                           timesteps_done=num_timesteps_done,
                           total_timesteps=total_timesteps)
            return DQN.load(checkpoint_path, env=env)
        
        # Step 3: Delegate to Code_RL train_dqn_agent
        start_time = time.time()
        
        try:
            trained_model = train_dqn_agent(
                env=env,
                total_timesteps=remaining_timesteps,
                model=model,  # None if from scratch, existing if resume
                checkpoint_dir=checkpoint_dir,
                checkpoint_freq=checkpoint_freq,
                model_name=model_name,
                
                # Hyperparameters forwarding
                learning_rate=hyperparameters.get('learning_rate', 1e-4),
                buffer_size=hyperparameters.get('buffer_size', 100000),
                learning_starts=hyperparameters.get('learning_starts', 1000),
                batch_size=hyperparameters.get('batch_size', 32),
                tau=hyperparameters.get('tau', 1.0),
                gamma=hyperparameters.get('gamma', 0.99),
                train_freq=hyperparameters.get('train_freq', 4),
                gradient_steps=hyperparameters.get('gradient_steps', 1),
                target_update_interval=hyperparameters.get('target_update_interval', 1000),
                exploration_fraction=hyperparameters.get('exploration_fraction', 0.1),
                exploration_initial_eps=hyperparameters.get('exploration_initial_eps', 1.0),
                exploration_final_eps=hyperparameters.get('exploration_final_eps', 0.05),
                
                # Training config
                verbose=1,
                tensorboard_log=None,  # Can be enabled if needed
                seed=None
            )
            
            training_duration = time.time() - start_time
            
            self.logger.info("rl_training_completed",
                           algorithm=algorithm,
                           total_timesteps=total_timesteps,
                           training_duration_seconds=training_duration,
                           training_duration_minutes=training_duration / 60.0)
            
            return trained_model
            
        except Exception as e:
            self.logger.error("rl_training_failed",
                            error=str(e),
                            algorithm=algorithm,
                            timesteps=remaining_timesteps)
            raise RuntimeError(f"RL training failed: {e}")
    
    def evaluate(self,
                 model: DQN,
                 env,
                 n_eval_episodes: int = 10) -> Dict[str, Any]:
        """
        Évalue un modèle entraîné sur l'environnement.
        
        Args:
            model: Trained DQN model
            env: Gymnasium environment
            n_eval_episodes: Number of evaluation episodes
        
        Returns:
            Dict with evaluation metrics (mean_reward, std_reward, episode_lengths)
        """
        self.logger.info("rl_evaluation_starting",
                        n_eval_episodes=n_eval_episodes)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_eval_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                # Predict action (deterministic for evaluation)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            self.logger.debug("eval_episode_complete",
                            episode=episode,
                            reward=episode_reward,
                            length=episode_length)
        
        # Calculate statistics
        import numpy as np
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        eval_results = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_episode_length': mean_length,
            'n_eval_episodes': n_eval_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        self.logger.info("rl_evaluation_completed",
                        mean_reward=mean_reward,
                        std_reward=std_reward,
                        mean_episode_length=mean_length)
        
        return eval_results
    
    def save_model(self,
                   model: DQN,
                   save_path: str):
        """
        Sauvegarde un modèle entraîné.
        
        Args:
            model: DQN model to save
            save_path: Path for saving (.zip extension)
        """
        try:
            model.save(save_path)
            self.logger.info("model_saved",
                           save_path=save_path)
        except Exception as e:
            self.logger.error("model_save_failed",
                            error=str(e),
                            save_path=save_path)
            raise
    
    def load_model(self,
                   load_path: str,
                   env) -> DQN:
        """
        Charge un modèle sauvegardé.
        
        Args:
            load_path: Path to saved model (.zip)
            env: Gymnasium environment
        
        Returns:
            Loaded DQN model
        """
        try:
            model = DQN.load(load_path, env=env)
            self.logger.info("model_loaded",
                           load_path=load_path)
            return model
        except Exception as e:
            self.logger.error("model_load_failed",
                            error=str(e),
                            load_path=load_path)
            raise
    
    def __repr__(self) -> str:
        """String representation."""
        return "CodeRLTrainingAdapter(algorithm=dqn, source=Code_RL/train_dqn.py)"
