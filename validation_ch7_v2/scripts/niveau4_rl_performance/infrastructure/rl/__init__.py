"""
Infrastructure - RL Adapters

Adapters pour intégrer Code_RL (source de vérité) dans notre Clean Architecture.

Modules:
    - code_rl_environment_adapter: Adapter TrafficSignalEnvDirect pour contexte Béninois
    - code_rl_training_adapter: Adapter train_dqn.py pour notre workflow
"""

from .code_rl_environment_adapter import BeninTrafficEnvironmentAdapter
from .code_rl_training_adapter import CodeRLTrainingAdapter

__all__ = [
    'BeninTrafficEnvironmentAdapter',
    'CodeRLTrainingAdapter',
]
