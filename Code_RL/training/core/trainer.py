"""
RL Training Orchestrator

Module principal pour orchestrer l'entraînement RL avec:
- Sanity checks automatiques (BUG #37, #33, #27, #36, Reward)
- Gestion des checkpoints (rotation, best, final)
- Logging structuré
- Évaluation périodique
- Reprise d'entraînement

Architecture:
    RLConfigBuilder (env config) + TrainingConfig (training config) → Trainer
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
import json

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Imports locaux
from Code_RL.src.utils.config import RLConfigBuilder
from Code_RL.src.rl.callbacks import RotatingCheckpointCallback, TrainingProgressCallback
from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect

from ..config import TrainingConfig
from .sanity_checker import SanityChecker, run_sanity_checks

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Orchestrateur d'entraînement RL
    
    Usage:
        # Setup configs
        rl_config = RLConfigBuilder.for_training("lagos")
        training_config = production_config("lagos_v1")
        
        # Train
        trainer = RLTrainer(rl_config, training_config)
        model = trainer.train()
    """
    
    def __init__(
        self,
        rl_config: RLConfigBuilder,
        training_config: TrainingConfig
    ):
        """
        Initialise le trainer
        
        Args:
            rl_config: Configuration de l'environnement (ARZ + RL env)
            training_config: Configuration de l'entraînement (hyperparams, checkpoints)
        """
        self.rl_config = rl_config
        self.training_config = training_config
        
        # Chemins
        self.output_dir = Path(training_config.output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        self.eval_dir = self.output_dir / "eval"
        
        # État
        self.model: Optional[DQN] = None
        self.env = None
        self.eval_env = None
        
        # Métriques
        self.start_time: Optional[float] = None
        self.total_steps_trained: int = 0
        
        logger.info(f"RLTrainer initialized for experiment: {training_config.experiment_name}")
    
    def train(self) -> DQN:
        """
        Entraîne le modèle RL de bout en bout
        
        Returns:
            Modèle DQN entraîné
        
        Raises:
            RuntimeError: Si sanity checks échouent
        """
        logger.info("=" * 80)
        logger.info(f"TRAINING: {self.training_config.experiment_name}")
        logger.info(f"Mode: {self.training_config.mode}")
        logger.info(f"Total timesteps: {self.training_config.total_timesteps}")
        logger.info(f"Device: {self.training_config.device}")
        logger.info("=" * 80)
        
        # 1. Setup directories
        self._setup_directories()
        
        # 2. Save configs
        self._save_configs()
        
        # 3. Sanity checks
        if self.training_config.sanity_check.enabled:
            logger.info("\n[STAGE 1/5] Running sanity checks...")
            run_sanity_checks(self.rl_config, self.training_config.sanity_check)
            logger.info("✅ All sanity checks passed!\n")
        else:
            logger.warning("⚠️ Sanity checks DISABLED - training at your own risk!\n")
        
        # 4. Create environments
        logger.info("[STAGE 2/5] Creating environments...")
        self._create_environments()
        logger.info("✅ Environments created\n")
        
        # 5. Create or load model
        logger.info("[STAGE 3/5] Creating/loading model...")
        self._create_or_load_model()
        logger.info(f"✅ Model ready: {type(self.model).__name__}\n")
        
        # 6. Setup callbacks
        logger.info("[STAGE 4/5] Setting up callbacks...")
        callbacks = self._create_callbacks()
        logger.info(f"✅ {len(callbacks.callbacks)} callbacks configured\n")
        
        # 7. Train
        logger.info("[STAGE 5/5] Starting training...")
        logger.info("=" * 80)
        
        self.start_time = time.time()
        
        try:
            self.model.learn(
                total_timesteps=self.training_config.total_timesteps,
                callback=callbacks,
                log_interval=10,
                reset_num_timesteps=not self.training_config.resume_training
            )
            
            training_time = time.time() - self.start_time
            logger.info("=" * 80)
            logger.info("✅ TRAINING COMPLETE!")
            logger.info(f"Total time: {training_time/60:.1f} minutes")
            logger.info(f"Steps/second: {self.training_config.total_timesteps/training_time:.1f}")
            logger.info("=" * 80)
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Training interrupted by user")
            self._save_final_model("interrupted")
        
        except Exception as e:
            logger.error(f"\n❌ Training failed: {str(e)}")
            self._save_final_model("failed")
            raise
        
        # 8. Save final model
        self._save_final_model("final")
        
        return self.model
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Évalue le modèle actuel
        
        Args:
            n_episodes: Nombre d'épisodes d'évaluation
        
        Returns:
            Dictionnaire de métriques
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet - call train() first")
        
        logger.info(f"Evaluating model over {n_episodes} episodes...")
        
        # Créer eval env si nécessaire
        if self.eval_env is None:
            self.eval_env = self._create_single_env()
        
        # Collecte des métriques
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            ep_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                ep_reward += reward
                ep_length += 1
            
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
        
        # Statistiques
        metrics = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "n_episodes": n_episodes
        }
        
        logger.info(f"Evaluation results: {metrics}")
        
        return metrics
    
    # =========================================================================
    # SETUP METHODS
    # =========================================================================
    
    def _setup_directories(self):
        """Crée la structure de dossiers pour l'entraînement"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.eval_dir.mkdir(exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
    
    def _save_configs(self):
        """Sauvegarde les configurations pour reproducibilité"""
        config_file = self.output_dir / "training_config.json"
        
        # Helper function to convert complex objects to JSON-serializable format
        def make_json_serializable(obj):
            """Recursively convert Pydantic models and Path objects for JSON serialization"""
            from pathlib import Path, PurePath
            from pydantic import BaseModel
            
            # Check Path first (before BaseModel, as some paths might be in pydantic models)
            if isinstance(obj, (Path, PurePath)):
                return str(obj)
            # Handle Pydantic models
            elif isinstance(obj, BaseModel):
                # Convert Pydantic model to dict using model_dump, then recursively clean
                return make_json_serializable(obj.model_dump())
            # Handle dictionaries
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            # Handle lists and tuples
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            # Handle other objects that might have __dict__
            elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
                # Try to convert to dict and clean
                try:
                    return make_json_serializable(vars(obj))
                except:
                    # If fails, try to convert to string
                    return str(obj)
            # Return as-is for JSON-serializable primitives
            else:
                return obj
        
        # Don't even try to save rl_env_params if it's causing issues
        # Just save the essential configs
        try:
            config_dict = {
                "training_config": self.training_config.to_dict(),
                "experiment_name": self.training_config.experiment_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Try to add rl_env_params, but skip if it fails
            try:
                config_dict["rl_env_params"] = make_json_serializable(self.rl_config.rl_env_params)
            except Exception as e:
                logger.warning(f"Could not serialize rl_env_params, skipping: {e}")
                config_dict["rl_env_params_error"] = str(e)
            
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configs saved to: {config_file}")
        except Exception as e:
            logger.error(f"Failed to save configs: {e}")
            # Don't fail the training just because config save failed
            pass
    
    def _create_environments(self):
        """Crée les environnements d'entraînement et d'évaluation"""
        # Training env (vectorized)
        self.env = make_vec_env(
            self._create_single_env,
            n_envs=1,  # Pour l'instant 1 seul env
            seed=self.training_config.seed
        )
        
        # Eval env (non-vectorized)
        self.eval_env = self._create_single_env()
        
        logger.info(f"Training env: {type(self.env).__name__}")
        logger.info(f"Eval env: {type(self.eval_env).__name__}")
    
    def _create_single_env(self):
        """Factory pour créer un environnement individuel"""
        return TrafficSignalEnvDirect(
            arz_simulation_config=self.rl_config.arz_simulation_config,
            endpoint_params=self.rl_config.endpoint_params,
            signal_params=self.rl_config.signal_params,
            **self.rl_config.rl_env_params
        )
    
    def _create_or_load_model(self):
        """Crée un nouveau modèle ou charge un checkpoint existant"""
        # Vérifier si reprise d'entraînement
        if self.training_config.resume_training:
            checkpoint = self._find_latest_checkpoint()
            if checkpoint is not None:
                logger.info(f"Resuming from checkpoint: {checkpoint}")
                self.model = DQN.load(
                    checkpoint,
                    env=self.env,
                    device=self.training_config.device
                )
                return
        
        # Créer nouveau modèle
        logger.info("Creating new DQN model...")
        
        hyperparams = self.training_config.dqn_hyperparams
        
        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=hyperparams.learning_rate,
            buffer_size=hyperparams.buffer_size,
            learning_starts=hyperparams.learning_starts,
            batch_size=hyperparams.batch_size,
            tau=hyperparams.tau,
            gamma=hyperparams.gamma,
            train_freq=hyperparams.train_freq,
            gradient_steps=hyperparams.gradient_steps,
            target_update_interval=hyperparams.target_update_interval,
            exploration_fraction=hyperparams.exploration_fraction,
            exploration_initial_eps=hyperparams.exploration_initial_eps,
            exploration_final_eps=hyperparams.exploration_final_eps,
            verbose=1,
            seed=self.training_config.seed,
            device=self.training_config.device
        )
    
    def _create_callbacks(self) -> CallbackList:
        """Crée la liste des callbacks pour l'entraînement"""
        callbacks = []
        
        # 1. Rotating checkpoint callback
        checkpoint_callback = RotatingCheckpointCallback(
            save_freq=self.training_config.get_checkpoint_freq(),
            save_path=str(self.checkpoints_dir / "latest"),
            name_prefix="dqn_checkpoint",
            max_checkpoints=self.training_config.checkpoint_strategy.max_checkpoints,
            save_replay_buffer=self.training_config.checkpoint_strategy.save_replay_buffer,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # 2. Training progress callback
        progress_callback = TrainingProgressCallback(
            check_freq=1000,
            verbose=1
        )
        callbacks.append(progress_callback)
        
        # 3. Eval callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.checkpoints_dir / "best"),
            log_path=str(self.eval_dir),
            eval_freq=self.training_config.evaluation_strategy.eval_freq,
            n_eval_episodes=self.training_config.evaluation_strategy.n_eval_episodes,
            deterministic=self.training_config.evaluation_strategy.deterministic,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)
        
        return CallbackList(callbacks)
    
    # =========================================================================
    # CHECKPOINT MANAGEMENT
    # =========================================================================
    
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Trouve le checkpoint le plus récent"""
        # Chercher dans latest/
        latest_dir = self.checkpoints_dir / "latest"
        if not latest_dir.exists():
            return None
        
        # Chercher fichiers .zip
        checkpoints = list(latest_dir.glob("dqn_checkpoint_*.zip"))
        if not checkpoints:
            return None
        
        # Trier par timestamp (dans le nom)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        return checkpoints[0]
    
    def _save_final_model(self, suffix: str = "final"):
        """Sauvegarde le modèle final"""
        if self.model is None:
            return
        
        final_path = self.checkpoints_dir / f"dqn_model_{suffix}"
        self.model.save(final_path)
        
        logger.info(f"Final model saved: {final_path}.zip")
        
        # Sauvegarder aussi le replay buffer
        if self.training_config.checkpoint_strategy.save_replay_buffer:
            buffer_path = self.checkpoints_dir / f"replay_buffer_{suffix}.pkl"
            self.model.save_replay_buffer(buffer_path)
            logger.info(f"Replay buffer saved: {buffer_path}")
    
    def cleanup(self):
        """Nettoie les ressources"""
        if self.env is not None:
            self.env.close()
        if self.eval_env is not None:
            self.eval_env.close()
        
        logger.info("Resources cleaned up")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def train_model(
    rl_config: RLConfigBuilder,
    training_config: TrainingConfig
) -> DQN:
    """
    Fonction utilitaire pour entraîner un modèle
    
    Args:
        rl_config: Configuration de l'environnement
        training_config: Configuration de l'entraînement
    
    Returns:
        Modèle DQN entraîné
    """
    trainer = RLTrainer(rl_config, training_config)
    model = trainer.train()
    trainer.cleanup()
    
    return model
