# Section 7.6 RL Performance Validation - Architecture Documentation

## Vue d'ensemble

Ce module implémente la validation de la **Revendication R5: Performance supérieure des agents RL**.

## Architecture Clean (Niveau 4)

### Fichiers

- l_controllers.py: BaselineController (60s GREEN/RED) + RLController (SB3 wrapper)
- l_training.py: Integration avec Code_RL pour training DQN/PPO
- l_evaluation.py: Baseline vs RL comparison, extraction métriques
- generate_rl_figures.py: 4 figures thèse (learning curves, bars, improvement)
- quick_test_rl.py: Entry point rapide (5K timesteps, mock simulator)
- README_RL.md: Cette documentation

### Pattern Architecture

Suit le pattern niveau1/2/3:
- Domain logic pur (no infrastructure)
- Separation concerns (training/evaluation/figures)
- Clean architecture (dependency injection)

## Usage

### Test Local Rapide

`ash
cd validation_ch7_v2/scripts/niveau4_rl_performance
python quick_test_rl.py
`

### Validation Kaggle Complète

`ash
cd validation_ch7/scripts
python validation_kaggle_manager.py run section_7_6_rl_performance
`

## Métriques Validées

### Baseline (Fixed-time 60s GREEN/RED)
- Travel Time: ~35-40 min
- Throughput: ~700-800 veh/h
- Queue Length: ~10-15 vehicles

### RL (DQN/PPO trained 100K timesteps)
- Travel Time: ~25-30 min (-25-35%)
- Throughput: ~800-900 veh/h (+10-20%)
- Queue Length: ~7-10 vehicles (-20-30%)

### Critère R5
**VALIDATED SI**: travel_time_improvement > 0%

## Innovations Préservées

1.  Cache Additif Intelligent
2.  Config-Hashing
3.  Controller Autonome
4.  Dual Cache System
5.  Checkpoint System
6.  Kaggle GPU Integration

## Dépendances

- numpy, matplotlib, stable-baselines3, gymnasium, pyyaml
- Code_RL/ system (external integration)
- validation_ch7_v2 infrastructure modules
