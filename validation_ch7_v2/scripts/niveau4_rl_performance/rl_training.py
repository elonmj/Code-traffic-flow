"""RL Training - REAL Integration with Code_RL TrafficSignalEnvDirect"""
import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

# Setup paths
CODE_RL_PATH = Path(__file__).parent.parent.parent.parent / "Code_RL"
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(CODE_RL_PATH) not in sys.path:
    sys.path.insert(0, str(CODE_RL_PATH))
    sys.path.insert(0, str(CODE_RL_PATH / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# ✅ REAL CODE_RL IMPORTS
from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Code_RL callbacks & config
try:
    from Code_RL.src.rl.callbacks import RotatingCheckpointCallback, TrainingProgressCallback
except ImportError:
    # Fallback: use standard CheckpointCallback if Code_RL callbacks not available
    RotatingCheckpointCallback = CheckpointCallback
    TrainingProgressCallback = None

# ✅ CODE_RL HYPERPARAMETERS (source of truth)
CODE_RL_HYPERPARAMETERS = {
    "learning_rate": 1e-3,  # Code_RL default (NOT 1e-4)
    "buffer_size": 50000,
    "learning_starts": 1000,
    "batch_size": 32,  # Code_RL default (NOT 64)
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 1000,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05
}


class RLTrainer:
    """RL Agent Trainer - REAL Code_RL integration with DQN."""
    
    def __init__(self, config_name: str = "lagos_master", algorithm: str = "DQN", device: str = "cpu"):
        self.config_name = config_name
        self.algorithm = algorithm
        self.device = device if device != "auto" else ("gpu" if _detect_gpu() else "cpu")
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)
    
    def train_agent(self, total_timesteps: int = 100000, use_mock: bool = False) -> Tuple[Any, Dict[str, Any]]:
        """Train RL agent using REAL Code_RL DQN."""
        
        if use_mock:
            # Quick test: mock training
            print(f"Training {self.algorithm} agent: {total_timesteps} timesteps (MOCK MODE)")
            return None, {
                "total_timesteps": total_timesteps,
                "algorithm": self.algorithm,
                "config": self.config_name,
                "device": self.device,
                "mode": "mock"
            }
        
        print(f"\n[TRAINING] Starting REAL DQN training")
        print(f"  Config: {self.config_name}")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Device: {self.device}")
        print(f"  Algorithm: {self.algorithm}")
        
        try:
            # ✅ CREATE ENVIRONMENT (TrafficSignalEnvDirect from Code_RL)
            print(f"\n[ENV] Creating TrafficSignalEnvDirect environment...")
            
            # Minimal scenario config (will use defaults if file not found)
            scenario_config = PROJECT_ROOT / "Code_RL" / "configs" / "env_lagos.yaml"
            
            env = TrafficSignalEnvDirect(
                scenario_config_path=str(scenario_config),
                decision_interval=15.0,  # ✅ Bug #27 fix: 15s (4x improvement)
                episode_max_time=3600.0,
                observation_segments={'upstream': [3, 4, 5], 'downstream': [6, 7, 8]},
                device=self.device,
                quiet=False
            )
            
            print(f"[ENV] ✅ Environment created:")
            print(f"       Observation space: {env.observation_space.shape}")
            print(f"       Action space: {env.action_space.n}")
            
            # ✅ INITIALIZE DQN AGENT (Code_RL hyperparameters)
            print(f"\n[AGENT] Initializing DQN agent...")
            model = DQN(
                'MlpPolicy',
                env,
                verbose=1,
                device=self.device,
                **CODE_RL_HYPERPARAMETERS  # ✅ Use Code_RL defaults
            )
            print(f"[AGENT] ✅ DQN agent initialized with Code_RL hyperparameters")
            
            # ✅ SETUP CALLBACKS
            callbacks = []
            checkpoint_freq = max(total_timesteps // 10, 100)  # 10 checkpoints
            
            # Checkpoint callback
            checkpoint_dir = self.models_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix=f"{self.config_name}_dqn"
            )
            callbacks.append(checkpoint_callback)
            
            # Evaluation callback
            eval_callback = EvalCallback(
                eval_env=env,
                best_model_save_path=str(self.models_dir / "best"),
                log_path=str(self.models_dir / "logs"),
                eval_freq=checkpoint_freq,
                n_eval_episodes=3,
                deterministic=True,
                verbose=1
            )
            callbacks.append(eval_callback)
            
            # ✅ TRAIN AGENT
            print(f"\n[TRAIN] Training DQN for {total_timesteps} timesteps...")
            print(f"         Checkpoint every {checkpoint_freq} steps")
            
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=True
            )
            
            # ✅ SAVE MODEL
            model_path = self.models_dir / f"dqn_{self.config_name}_{total_timesteps}_steps"
            model.save(str(model_path))
            print(f"\n[TRAIN] ✅ Training complete!")
            print(f"       Model saved: {model_path}.zip")
            
            env.close()
            
            training_history = {
                "total_timesteps": total_timesteps,
                "algorithm": self.algorithm,
                "config": self.config_name,
                "device": self.device,
                "model_path": str(model_path),
                "hyperparameters": CODE_RL_HYPERPARAMETERS,
                "mode": "real"
            }
            
            return model, training_history
            
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def _detect_gpu() -> bool:
    """Detect if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def train_rl_agent_for_validation(config_name: str = "lagos_master", total_timesteps: int = 100000,
                                   algorithm: str = "DQN", device: str = "cpu", use_mock: bool = False):
    """Convenience function for training RL agent in validation context."""
    trainer = RLTrainer(config_name=config_name, algorithm=algorithm, device=device)
    return trainer.train_agent(total_timesteps=total_timesteps, use_mock=use_mock)
