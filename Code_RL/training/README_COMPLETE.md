# RL Training System - Guide Complet

Training system architecture for RL-based traffic signal control with **automatic bug detection** and **production-ready** orchestration.

## 🚀 Quick Start

```bash
# Sanity check (5 min) - TOUJOURS faire ça d'abord!
python -m Code_RL.training.train --mode sanity --scenario lagos

# Quick test (15 min)
python -m Code_RL.training.train --mode quick --scenario lagos

# Production (2-4h)
python -m Code_RL.training.train --mode production --scenario lagos

# Kaggle GPU (9h limit)
python -m Code_RL.training.train --mode kaggle --scenario lagos --device cuda
```

## 🏗️ Architecture Overview

```
Code_RL/training/
├── config/                      # Pydantic configurations
│   ├── __init__.py
│   └── training_config.py      # TrainingConfig, DQNHyperparameters, etc.
│
├── core/                        # Core training logic
│   ├── __init__.py
│   ├── trainer.py              # RLTrainer orchestrator
│   └── sanity_checker.py       # Pre-training validation (BUG #37, #33, #27)
│
├── __init__.py                  # Package API
├── train.py                     # CLI entry point
├── quick_start.py              # Minimal example
└── README_COMPLETE.md           # This file
```

## 📦 Key Components

### 1. TrainingConfig (Pydantic) - Séparation des Concerns

**IMPORTANT**: `TrainingConfig` est SÉPARÉ de `RLConfigBuilder`:
- `RLConfigBuilder` (src/utils/config.py): Configuration de l'**ENVIRONNEMENT** (ARZ + RL env)
- `TrainingConfig` (training/config/): Configuration de l'**ENTRAÎNEMENT** (hyperparams DQN)

```python
from Code_RL.training.config import production_config

# Scénarios prédéfinis
config = production_config("lagos_v1")

# Configuration manuelle
from Code_RL.training.config import TrainingConfig, DQNHyperparameters

config = TrainingConfig(
    experiment_name="lagos_v1",
    mode="production",
    total_timesteps=100000,
    device="cuda",
    dqn_hyperparams=DQNHyperparameters(
        learning_rate=1e-3,
        buffer_size=50000,
        batch_size=32
    ),
    checkpoint_strategy=CheckpointStrategy(
        save_freq=1000,
        max_checkpoints=2
    )
)
```

### 2. Trainer - Orchestrateur Principal

```python
from Code_RL.training import train_model, production_config
from Code_RL.src.utils.config import RLConfigBuilder

# Config environnement (ARZ + RL env)
rl_config = RLConfigBuilder.for_training("lagos")

# Config entraînement (DQN hyperparams)
training_config = production_config("lagos_v1")

# Train!
model = train_model(rl_config, training_config)
```

**Ou avec plus de contrôle:**

```python
from Code_RL.training.core import RLTrainer

trainer = RLTrainer(rl_config, training_config)
model = trainer.train()
metrics = trainer.evaluate(n_episodes=10)
trainer.cleanup()
```

### 3. Sanity Checker - Validation Pré-Entraînement

**CRITIQUE**: Vérifie automatiquement les 5 BUGS MORTELS avant l'entraînement:

| Bug    | Symptôme                        | Fix                                    |
|--------|---------------------------------|----------------------------------------|
| #37    | Action truncation               | `round(float(action))` vs `int()`      |
| #33    | Queue toujours zéro             | `rho_inflow >> rho_initial` (15:1)     |
| #27    | Pas d'apprentissage             | `dt_decision = 15s` (pas 60s)          |
| #36    | Erreur GPU/CPU                  | Vérifier device consistency            |
| Reward | Reward constant                 | Au moins 5 valeurs uniques sur 100 steps|

```python
from Code_RL.training.core import run_sanity_checks
from Code_RL.training.config import SanityCheckConfig

sanity_config = SanityCheckConfig(
    enabled=True,
    num_steps=100,
    min_unique_rewards=5
)

# Lève RuntimeError si les checks échouent
run_sanity_checks(rl_config, sanity_config)
```

## 📊 Modes d'Entraînement

| Mode       | Steps   | Time    | Purpose                        | Checkpoint Freq |
|------------|---------|---------|--------------------------------|-----------------|
| **sanity** | 100     | 5 min   | Validate setup, check bugs     | 50              |
| **quick**  | 5,000   | 15 min  | Test learning, verify rewards  | 500             |
| **production** | 100,000 | 2-4h | Full training              | 1,000           |
| **kaggle** | 200,000 | 9h      | GPU-optimized for Kaggle       | 2,000           |

## 📁 Output Structure

```
results/
└── {experiment_name}/
    ├── checkpoints/
    │   ├── latest/                    # Rotating checkpoints (max 2)
    │   │   ├── dqn_checkpoint_1000_steps.zip
    │   │   └── replay_buffer_1000_steps.pkl
    │   ├── best/                      # Best model (eval callback)
    │   │   └── best_model.zip
    │   ├── dqn_model_final.zip       # Final model
    │   └── replay_buffer_final.pkl
    ├── logs/
    │   └── training.log
    ├── eval/
    │   └── evaluations.npz
    └── training_config.json           # Reproducibility
```

## 🔧 Configuration Pydantic - Détails

### DQNHyperparameters

```python
DQNHyperparameters(
    learning_rate=1e-3,           # Learning rate (default from Code_RL)
    buffer_size=50000,            # Replay buffer size
    learning_starts=1000,         # Steps before training starts
    batch_size=32,                # Batch size
    tau=1.0,                      # Soft update coefficient
    gamma=0.99,                   # Discount factor
    train_freq=4,                 # Training frequency
    gradient_steps=1,             # Gradient steps per update
    target_update_interval=1000,  # Target network update freq
    exploration_fraction=0.1,     # Fraction of timesteps for exploration
    exploration_initial_eps=1.0,  # Initial epsilon
    exploration_final_eps=0.05    # Final epsilon
)
```

### CheckpointStrategy

```python
CheckpointStrategy(
    save_freq=1000,              # Fréquence de sauvegarde (steps)
    max_checkpoints=2,           # Nombre max de checkpoints (rotation)
    save_replay_buffer=True      # Sauvegarder replay buffer (requis pour DQN)
)
```

### EvaluationStrategy

```python
EvaluationStrategy(
    eval_freq=5000,              # Fréquence d'évaluation (steps)
    n_eval_episodes=10,          # Nombre d'épisodes pour évaluation
    deterministic=True           # Actions déterministes pour évaluation
)
```

### SanityCheckConfig

```python
SanityCheckConfig(
    enabled=True,                # Activer les sanity checks
    num_steps=100,               # Nombre de steps pour le test
    min_unique_rewards=5,        # Minimum de rewards uniques requis
    min_max_queue=5.0,          # Queue maximale minimale requise
    check_action_mapping=True,   # Vérifier round() vs int()
    check_flux_config=True,      # Vérifier q_inflow >> q_initial
    check_control_interval=True  # Vérifier interval = 15s
)
```

## 🔌 Integration with Existing Code

Le système réutilise l'infrastructure moderne de `Code_RL/`:

| Module Existant                  | Usage dans Training                    |
|----------------------------------|----------------------------------------|
| `RLConfigBuilder`                | Configuration de l'environnement ARZ   |
| `TrafficSignalEnvDirect`         | Environnement RL                       |
| `RotatingCheckpointCallback`     | Gestion des checkpoints (rotation)     |
| `TrainingProgressCallback`       | Logging des progrès                    |
| `train_dqn.py`                   | Source of truth pour hyperparamètres   |

## 🎯 Usage Examples

### Example 1: Quick Sanity Check

```python
from Code_RL.src.utils.config import RLConfigBuilder
from Code_RL.training import train_model, sanity_check_config

rl_config = RLConfigBuilder.for_training("lagos")
training_config = sanity_check_config()

model = train_model(rl_config, training_config)
# Output: results/sanity_check/
```

### Example 2: Production Training

```python
from Code_RL.src.utils.config import RLConfigBuilder
from Code_RL.training import train_model, production_config

rl_config = RLConfigBuilder.for_training("lagos")
training_config = production_config("lagos_v1")

model = train_model(rl_config, training_config)
# Output: results/lagos_v1/
```

### Example 3: Custom Configuration

```python
from Code_RL.src.utils.config import RLConfigBuilder
from Code_RL.training import TrainingConfig, DQNHyperparameters, train_model

rl_config = RLConfigBuilder.for_training("lagos")

training_config = TrainingConfig(
    experiment_name="lagos_custom",
    mode="production",
    total_timesteps=150000,  # Custom timesteps
    device="cuda",
    dqn_hyperparams=DQNHyperparameters(
        learning_rate=5e-4,  # Custom learning rate
        buffer_size=100000   # Larger buffer
    )
)

model = train_model(rl_config, training_config)
```

### Example 4: Resume Training

```bash
# CLI
python -m Code_RL.training.train --mode production --scenario lagos --resume

# Python
training_config.resume_training = True
training_config.checkpoint_path = Path("results/lagos_v1/checkpoints/latest/...")
```

## 📝 CLI Reference

```bash
python -m Code_RL.training.train [OPTIONS]

Options:
  --mode {sanity,quick,production,kaggle}
                        Training mode (default: production)
  --scenario {simple,lagos,riemann}
                        Training scenario (default: lagos)
  --device {cpu,cuda}   Device (default: cpu)
  --timesteps INT       Total timesteps (overrides mode default)
  --name STR            Experiment name (default: auto-generated)
  --resume              Resume from latest checkpoint
  --no-sanity-checks    Disable sanity checks (NOT RECOMMENDED)
```

## 🐛 Troubleshooting

### Issue: Sanity checks fail

**Solution**: Vérifier les bugs documentés dans `RL_TRAINING_SURVIVAL_GUIDE.md`:
- BUG #37: Action mapping uses `round()` not `int()`
- BUG #33: `rho_inflow >> rho_initial` (ratio 15:1)
- BUG #27: `dt_decision = 15.0` (not 60.0)

### Issue: Training doesn't resume

**Solution**: Vérifier que `--resume` est spécifié ET qu'un checkpoint existe dans `results/{experiment_name}/checkpoints/latest/`

### Issue: Out of memory on GPU

**Solution**: Réduire `buffer_size` ou `batch_size` dans DQNHyperparameters

### Issue: Reward constant / no learning

**Solution**: Sanity checker détectera cela automatiquement. Vérifier reward function.

## 📚 See Also

- `RL_TRAINING_SURVIVAL_GUIDE.md`: Lessons learned from 351 commits
- `Code_RL/src/rl/train_dqn.py`: Source of truth for hyperparameters
- `.copilot-tracking/analysis/rl_history_analysis.md`: Full commit analysis

## 🎓 Philosophy

Ce système incarne les leçons des 351 commits analysés:
1. **Separation of Concerns**: Environment config ≠ Training config
2. **Fail Fast**: Sanity checks detect bugs BEFORE wasting hours
3. **Reproducibility**: All configs saved as JSON
4. **Modularity**: Reuse existing Code_RL infrastructure
5. **Safety**: Resume training, rotating checkpoints, best model tracking
