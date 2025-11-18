"""
Training Configuration System

Séparation des concerns:
- RLConfigBuilder (src/utils/config.py): Config de l'ENVIRONNEMENT (ARZ + RL env)
- TrainingConfig (ici): Config de l'ENTRAÎNEMENT (hyperparams, checkpoints, etc.)

Cette séparation permet de:
- Réutiliser le même environnement avec différents hyperparams
- Tester différentes stratégies d'entraînement
- Faciliter le tuning des hyperparamètres
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from pathlib import Path


class DQNHyperparameters(BaseModel):
    """
    Hyperparamètres DQN (source: Code_RL/src/rl/train_dqn.py lignes 151-167)
    
    Ces valeurs sont le SOURCE OF TRUTH extrait du code existant.
    """
    learning_rate: float = Field(default=1e-3, description="Learning rate (Code_RL default)")
    buffer_size: int = Field(default=50000, description="Replay buffer size")
    learning_starts: int = Field(default=1000, description="Steps before training starts")
    batch_size: int = Field(default=32, description="Batch size (Code_RL default)")
    tau: float = Field(default=1.0, description="Soft update coefficient")
    gamma: float = Field(default=0.99, description="Discount factor")
    train_freq: int = Field(default=4, description="Training frequency")
    gradient_steps: int = Field(default=1, description="Gradient steps per update")
    target_update_interval: int = Field(default=1000, description="Target network update frequency")
    exploration_fraction: float = Field(default=0.1, description="Fraction of timesteps for exploration")
    exploration_initial_eps: float = Field(default=1.0, description="Initial epsilon")
    exploration_final_eps: float = Field(default=0.05, description="Final epsilon")


class CheckpointStrategy(BaseModel):
    """
    Stratégie de sauvegarde des checkpoints
    
    Basé sur les leçons de train_dqn.py (système à 3 niveaux):
    1. Latest: Rotation automatique (2 plus récents)
    2. Best: 1 meilleur modèle (évaluation)
    3. Final: État à la fin de l'entraînement
    """
    save_freq: int = Field(default=1000, description="Fréquence de sauvegarde (steps)")
    max_checkpoints: int = Field(default=2, description="Nombre max de checkpoints latest (rotation)")
    save_replay_buffer: bool = Field(default=True, description="Sauvegarder le replay buffer (requis pour DQN)")
    
    @validator('save_freq')
    def validate_save_freq(cls, v, values):
        if v <= 0:
            raise ValueError("save_freq must be > 0")
        return v


class EvaluationStrategy(BaseModel):
    """Stratégie d'évaluation pendant l'entraînement"""
    eval_freq: int = Field(default=1000, description="Fréquence d'évaluation (steps)")
    n_eval_episodes: int = Field(default=5, description="Nombre d'épisodes pour évaluation")
    deterministic: bool = Field(default=True, description="Actions déterministes pour évaluation")
    
    @validator('eval_freq')
    def validate_eval_freq(cls, v):
        if v <= 0:
            raise ValueError("eval_freq must be > 0")
        return v


class SanityCheckConfig(BaseModel):
    """
    Configuration pour les tests de sanité pré-entraînement
    
    Basé sur RL_TRAINING_SURVIVAL_GUIDE.md - vérifie BUG #37, #33, #27
    """
    enabled: bool = Field(default=True, description="Activer les sanity checks")
    num_steps: int = Field(default=100, description="Nombre de steps pour le test")
    min_unique_rewards: int = Field(default=5, description="Minimum de rewards uniques requis")
    min_max_queue: float = Field(default=5.0, description="Queue maximale minimale requise")
    check_action_mapping: bool = Field(default=True, description="Vérifier round() vs int()")
    check_flux_config: bool = Field(default=True, description="Vérifier q_inflow >> q_initial")
    check_control_interval: bool = Field(default=True, description="Vérifier interval = 15s (pas 60s)")


class TrainingMode(str):
    """Modes d'entraînement prédéfinis"""
    SANITY = "sanity"  # 100 steps - vérification rapide
    QUICK = "quick"    # 5000 steps - validation apprentissage
    PRODUCTION = "production"  # 100k steps - entraînement complet


class TrainingConfig(BaseModel):
    """
    Configuration complète pour l'entraînement RL
    
    Sépare les concerns:
    - Environnement (ARZ): Géré par RLConfigBuilder
    - Entraînement (DQN): Géré par cette classe
    """
    
    # === Identité de l'expérience ===
    experiment_name: str = Field(default="dqn_traffic_control", description="Nom de l'expérience")
    description: Optional[str] = Field(default=None, description="Description de l'expérience")
    
    # === Mode d'entraînement ===
    mode: Literal["sanity", "quick", "production"] = Field(
        default="production",
        description="Mode d'entraînement (définit timesteps par défaut)"
    )
    total_timesteps: Optional[int] = Field(
        default=None,
        description="Nombre total de timesteps (None = auto selon mode)"
    )
    
    # === Hyperparamètres DQN ===
    dqn_hyperparams: DQNHyperparameters = Field(default_factory=DQNHyperparameters)
    
    # === Stratégies ===
    checkpoint_strategy: CheckpointStrategy = Field(default_factory=CheckpointStrategy)
    evaluation_strategy: EvaluationStrategy = Field(default_factory=EvaluationStrategy)
    sanity_check: SanityCheckConfig = Field(default_factory=SanityCheckConfig)
    
    # === Environnement ===
    seed: int = Field(default=42, description="Random seed")
    device: str = Field(default="cpu", description="Device (cpu/cuda)")
    
    # === Output ===
    output_dir: Optional[Path] = Field(default=None, description="Dossier de sortie")
    
    # === Reprise d'entraînement ===
    resume_training: bool = Field(default=True, description="Reprendre depuis dernier checkpoint")
    checkpoint_path: Optional[Path] = Field(default=None, description="Chemin checkpoint spécifique")
    
    class Config:
        use_enum_values = True
    
    @validator('total_timesteps', always=True)
    def set_timesteps_from_mode(cls, v, values):
        """Définit automatiquement timesteps selon le mode si non spécifié"""
        if v is not None:
            return v
        
        mode = values.get('mode', 'production')
        timesteps_map = {
            'sanity': 100,
            'quick': 5000,
            'production': 100000
        }
        return timesteps_map.get(mode, 100000)
    
    @validator('output_dir', always=True)
    def set_output_dir(cls, v, values):
        """Crée output_dir depuis experiment_name si non spécifié"""
        if v is not None:
            return v
        
        experiment_name = values.get('experiment_name', 'dqn_traffic_control')
        return Path(f"results/{experiment_name}")
    
    def get_checkpoint_freq(self) -> int:
        """Retourne la fréquence de checkpoint adaptative selon le mode"""
        # Logique de train_dqn.py lignes 267-277
        if self.checkpoint_strategy.save_freq is not None:
            return self.checkpoint_strategy.save_freq
        
        if self.total_timesteps < 5000:
            return 100  # Quick test
        elif self.total_timesteps < 20000:
            return 500  # Small run
        else:
            return 1000  # Production
    
    def to_dict(self) -> dict:
        """Convertit en dictionnaire pour logging"""
        return self.dict()


# === Scénarios Prédéfinis ===

def sanity_check_config() -> TrainingConfig:
    """Configuration pour sanity check rapide (5 min)"""
    return TrainingConfig(
        experiment_name="sanity_check",
        mode="sanity",
        total_timesteps=100,
        sanity_check=SanityCheckConfig(enabled=True),
        checkpoint_strategy=CheckpointStrategy(save_freq=50, max_checkpoints=1),
        evaluation_strategy=EvaluationStrategy(eval_freq=50, n_eval_episodes=2)
    )


def quick_test_config() -> TrainingConfig:
    """Configuration pour test rapide (15 min)"""
    return TrainingConfig(
        experiment_name="quick_test",
        mode="quick",
        total_timesteps=5000,
        checkpoint_strategy=CheckpointStrategy(save_freq=500, max_checkpoints=2),
        evaluation_strategy=EvaluationStrategy(eval_freq=500, n_eval_episodes=5)
    )


def production_config(experiment_name: str = "lagos_production") -> TrainingConfig:
    """Configuration pour entraînement production (2-4h)"""
    return TrainingConfig(
        experiment_name=experiment_name,
        mode="production",
        total_timesteps=100000,
        checkpoint_strategy=CheckpointStrategy(save_freq=1000, max_checkpoints=2),
        evaluation_strategy=EvaluationStrategy(eval_freq=5000, n_eval_episodes=10)
    )


def kaggle_gpu_config(experiment_name: str = "lagos_kaggle_gpu") -> TrainingConfig:
    """Configuration optimisée pour Kaggle GPU (9h limit)"""
    return TrainingConfig(
        experiment_name=experiment_name,
        mode="production",
        total_timesteps=200000,  # Plus de steps avec GPU
        device="cuda",
        checkpoint_strategy=CheckpointStrategy(
            save_freq=2000,  # Moins fréquent (GPU plus rapide)
            max_checkpoints=2,
            save_replay_buffer=True
        ),
        evaluation_strategy=EvaluationStrategy(
            eval_freq=10000,
            n_eval_episodes=10
        )
    )
