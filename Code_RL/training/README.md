# Training System - Architecture Moderne

Ce dossier contient le **système unique d'entraînement RL** pour le contrôle de signaux de trafic.

## 🎯 Philosophie

**Une seule tâche: Entraîner l'agent RL de manière robuste et reproductible**

- ✅ Applique TOUTES les leçons du `RL_TRAINING_SURVIVAL_GUIDE.md`
- ✅ Utilise le code moderne de `Code_RL/src/`
- ✅ Architecture modulaire inspirée de `niveau4_rl_performance/`
- ✅ Décisions mathématiques validées de `test_section_7_6_rl_performance.py`

## 📁 Structure

```
training/
├── README.md                 # Ce fichier
├── train.py                  # 🚀 POINT D'ENTRÉE UNIQUE
├── config/
│   ├── __init__.py
│   ├── training_config.py    # Configuration d'entraînement (Pydantic)
│   └── scenarios.py          # Scénarios prédéfinis (Lagos, etc.)
├── core/
│   ├── __init__.py
│   ├── trainer.py            # Orchestrateur d'entraînement
│   ├── evaluator.py          # Évaluation baseline vs RL
│   └── sanity_checker.py     # Tests pré-entraînement (BUG #37, #33, #27)
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py      # Logs microscopiques
│   └── checkpoint_manager.py # Gestion checkpoints avec rotation
└── notebooks/
    └── analyze_results.ipynb # Analyse post-entraînement
```

## 🚀 Usage

### Quick Start (Test Rapide)

```bash
# Test de sanité (5 min)
python training/train.py --mode sanity --timesteps 100

# Quick test (15 min, 5000 steps)
python training/train.py --mode quick --timesteps 5000

# Production (2-4h, 100k steps)
python training/train.py --mode production --timesteps 100000
```

### Avec Configuration Personnalisée

```python
# training/config/scenarios.py
from training.config.training_config import TrainingConfig

lagos_config = TrainingConfig(
    scenario_name="lagos_victoria_island",
    control_interval=15.0,  # BUG #27: PAS 60s!
    episode_length=3600.0,
    
    # BUG #33: Flux entrant >> Flux initial
    initial_density_ratio=0.1,  # Route vide
    inflow_density_ratio=0.8,   # Forte demande
    
    # BUG #37: round() utilisé automatiquement dans env
    
    timesteps=100000,
    checkpoint_freq=1000,
    eval_freq=5000,
)
```

```bash
python training/train.py --config lagos_victoria_island
```

## ✅ Checklist Automatique Pré-Entraînement

Le système vérifie automatiquement (via `sanity_checker.py`):

1. **Actions mapping** → `round()` utilisé (pas `int()`)
2. **Flux configuration** → `q_inflow >> q_initial`
3. **Intervalle contrôle** → 15s (pas 60s)
4. **Reward variance** → Actions aléatoires génèrent rewards variés
5. **Queue formation** → Queue > 0 dans les 100 premiers steps

**Si un check échoue → Entraînement s'arrête avec diagnostic clair**

## 📊 Outputs

```
Code_RL/results/
├── {experiment_name}/
│   ├── checkpoints/
│   │   ├── checkpoint_1000_steps.zip
│   │   ├── checkpoint_5000_steps.zip
│   │   └── best_model.zip  # Meilleure évaluation
│   ├── logs/
│   │   ├── training.log        # Logs microscopiques
│   │   ├── tensorboard/        # TensorBoard events
│   │   └── sanity_check.log    # Résultats pré-entraînement
│   ├── eval/
│   │   ├── baseline_results.json
│   │   ├── rl_results.json
│   │   └── comparison.json
│   └── metadata.json  # Config, hyperparams, timestamps
```

## 🎓 Leçons Appliquées

### De `niveau4_rl_performance/` (Modularité)

- ✅ Séparation `core/` (business logic) vs `infrastructure/` (technical)
- ✅ Cache intelligent avec config hashing
- ✅ Checkpoint rotation automatique

### De `test_section_7_6_rl_performance.py` (Math)

- ✅ Métriques: efficiency, delay, throughput
- ✅ Baseline fixed-time (60s GREEN/RED) comme référence
- ✅ Évaluation sur MÊME fenêtre temporelle (BUG 0%)

### De `RL_TRAINING_SURVIVAL_GUIDE.md` (Bugs)

- ✅ `round(action)` au lieu de `int(action)` (BUG #37)
- ✅ `q_inflow >> q_initial` vérifié (BUG #33)
- ✅ `control_interval = 15s` (BUG #27)
- ✅ Logs microscopiques pour debug reward (BUG Reward)
- ✅ Config identique baseline vs RL (BUG 0%)

## 🔧 Intégration avec Code_RL Existant

```python
# Le système utilise DIRECTEMENT le code moderne de Code_RL/src/
from Code_RL.src.env.traffic_signal_env import TrafficSignalEnv
from Code_RL.src.rl.callbacks import RotatingCheckpointCallback
from Code_RL.src.utils.config import RLConfigBuilder

# Pas de duplication - on réutilise ce qui existe!
```

## 📝 Exemple de Run Complet

```bash
# 1. Sanity check (OBLIGATOIRE avant entraînement long)
$ python training/train.py --mode sanity
✅ Sanity check PASSED:
   - Action mapping: round() verified
   - Flux: q_inflow (1780) >> q_initial (222) ✓
   - Control interval: 15.0s ✓
   - Reward variance: 23 unique values ✓
   - Queue formation: max=45.2 vehicles ✓

# 2. Quick test (valider apprentissage)
$ python training/train.py --mode quick --timesteps 5000
🚀 Starting QUICK training: 5000 timesteps
📊 Episode 10/50: reward=0.15 (improving!)
💾 Checkpoint saved: checkpoint_1000_steps.zip
✅ Training completed in 12.5 minutes

# 3. Production run
$ python training/train.py --mode production --timesteps 100000
🚀 Starting PRODUCTION training: 100000 timesteps
📊 Progress: 10000/100000 (10%)
   - Mean reward (last 100 episodes): 0.42
   - Best evaluation reward: 0.58
💾 Checkpoint saved: checkpoint_10000_steps.zip
...
✅ Training completed in 3.2 hours
📊 Best model: results/lagos/best_model.zip (eval_reward=0.65)
```

## 🎯 Prochaines Étapes

1. Implémenter `trainer.py` (orchestrateur principal)
2. Implémenter `sanity_checker.py` (checks automatiques)
3. Implémenter `evaluator.py` (baseline vs RL comparison)
4. Tester sur Lagos Victoria Island scenario
5. Déployer sur Kaggle pour GPU training

---

**Règle d'Or**: Si reward = 0.0 après 1000 steps → STOP, debug, ne pas perdre de temps !
