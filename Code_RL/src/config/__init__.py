"""
Configuration Pydantic pour Code_RL.

Ce module fournit des wrappers RL-specific autour du système de configuration
Pydantic de arz_model, optimisés pour l'entraînement d'agents RL.
"""
from .rl_network_config import create_rl_training_config

__all__ = ['create_rl_training_config']
