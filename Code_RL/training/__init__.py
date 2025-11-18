"""
RL Training Package

Package complet pour l'entraînement RL avec:
- Sanity checks automatiques (BUG #37, #33, #27, #36, Reward)
- Configuration Pydantic (TrainingConfig)
- Orchestration d'entraînement (RLTrainer)
- Point d'entrée CLI (train.py)

Usage rapide:
    from Code_RL.training import train_model, production_config
    from Code_RL.src.utils.config import RLConfigBuilder
    
    rl_config = RLConfigBuilder.for_training("lagos")
    training_config = production_config("lagos_v1")
    
    model = train_model(rl_config, training_config)

Usage CLI:
    python -m Code_RL.training.train --mode production --scenario lagos
"""

from .config import (
    TrainingConfig,
    DQNHyperparameters,
    CheckpointStrategy,
    EvaluationStrategy,
    SanityCheckConfig,
    sanity_check_config,
    quick_test_config,
    production_config,
    kaggle_gpu_config
)

from .core import (
    RLTrainer,
    train_model,
    SanityChecker,
    SanityCheckResult,
    run_sanity_checks
)

__all__ = [
    # Config
    "TrainingConfig",
    "DQNHyperparameters",
    "CheckpointStrategy",
    "EvaluationStrategy",
    "SanityCheckConfig",
    "sanity_check_config",
    "quick_test_config",
    "production_config",
    "kaggle_gpu_config",
    
    # Core
    "RLTrainer",
    "train_model",
    "SanityChecker",
    "SanityCheckResult",
    "run_sanity_checks"
]

__version__ = "1.0.0"
