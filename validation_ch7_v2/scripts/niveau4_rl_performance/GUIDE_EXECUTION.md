# Guide d'Exécution Rapide - Section 7.6 RL Performance

## ✅ Système Maintenant Exécutable!

---

## Commandes Disponibles

### 1. Aide CLI
```bash
cd "d:\Projets\Alibi\Code project\validation_ch7_v2\scripts\niveau4_rl_performance"
python entry_points/cli.py --help
```

**Output attendu**:
```
Usage: cli.py [OPTIONS] COMMAND [ARGS]...
  Section 7.6 RL Performance Validation CLI.

Commands:
  info  Affiche informations architecture...
  run   Exécute validation Section 7.6 RL...
```

### 2. Information Architecture
```bash
python entry_points/cli.py info
```

Affiche:
- Innovations implémentées
- Architecture Clean Architecture
- Composants et responsabilités

### 3. Mode Quick Test (Recommandé pour premiers tests)
```bash
python entry_points/cli.py run --section section_7_6 --quick-test
```

**Durée**: ~5 minutes
**Ce que ça fait**:
- Charge scénario quick_low_traffic (5 min simulation)
- Entraîne DQN avec 1000 timesteps seulement
- Teste workflow complet sans GPU intensif
- Vérifie cache, checkpoints, logging

### 4. Exécution Complète
```bash
python entry_points/cli.py run --section section_7_6
```

**Durée**: ~2-3 heures (avec GPU)
**Ce que ça fait**:
- Charge tous les scénarios (low, medium, high, peak traffic)
- Entraîne DQN avec 100,000 timesteps
- Sauvegarde checkpoints rotatifs
- Cache baseline pour réutilisation
- Logs structurés JSON + console

### 5. Avec Config Personnalisé
```bash
python entry_points/cli.py run --section section_7_6 --config-file mon_config.yaml
```

---

## Structure Fichiers Générés

### Après Exécution

```
niveau4_rl_performance/
├── logs/
│   └── section_7_6_rl_performance.log  # Logs structurés JSON
│
├── cache/
│   ├── baseline/
│   │   └── baseline_*.pkl  # Cache trajectoires baseline
│   └── rl/
│       └── rl_*.pkl  # Cache résultats RL
│
├── checkpoints/
│   ├── dqn_checkpoint_1_*.zip  # Checkpoint RL 1
│   ├── dqn_checkpoint_2_*.zip  # Checkpoint RL 2
│   └── dqn_checkpoint_3_*.zip  # Checkpoint RL 3 (rotation keep_last=3)
│
└── config/
    └── section_7_6_rl_performance.yaml  # Configuration source
```

---

## Vérifications Post-Exécution

### 1. Vérifier Logs
```bash
# Voir dernières lignes logs
Get-Content logs/section_7_6_rl_performance.log -Tail 50

# Chercher événements spécifiques
Select-String -Path logs/section_7_6_rl_performance.log -Pattern "cache_hit"
Select-String -Path logs/section_7_6_rl_performance.log -Pattern "checkpoint_saved"
Select-String -Path logs/section_7_6_rl_performance.log -Pattern "training_complete"
```

### 2. Vérifier Cache
```bash
# Lister fichiers cache
Get-ChildItem cache/baseline/ -Recurse
Get-ChildItem cache/rl/ -Recurse

# Taille cache
(Get-ChildItem cache/ -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
```

### 3. Vérifier Checkpoints
```bash
# Lister checkpoints
Get-ChildItem checkpoints/ | Sort-Object LastWriteTime -Descending

# Vérifier rotation (max 3 checkpoints normalement)
(Get-ChildItem checkpoints/*.zip).Count
```

---

## Debugging

### Import Errors
Si erreurs d'import:
```bash
# Vérifier sys.path
python -c "import sys; print('\n'.join(sys.path))"

# Tester imports individuels
python -c "from infrastructure.rl import BeninTrafficEnvironmentAdapter; print('OK')"
python -c "from domain.controllers.rl_controller import RLController; print('OK')"
```

### Path Errors (Code_RL not found)
Si Code_RL introuvable:
```bash
# Vérifier que Code_RL existe
Test-Path "d:\Projets\Alibi\Code project\Code_RL"

# Vérifier structure Code_RL
Get-ChildItem "d:\Projets\Alibi\Code project\Code_RL"
```

### Dépendances Manquantes
Si module introuvable:
```bash
# Installer toutes dépendances
cd "d:\Projets\Alibi\Code project\validation_ch7_v2\scripts\niveau4_rl_performance"
pip install -r requirements.txt

# Ou installer individuellement
pip install stable-baselines3 gymnasium pyyaml structlog click
```

---

## Workflow Complet Recommandé

### Première Utilisation

```bash
# 1. Aller dans répertoire
cd "d:\Projets\Alibi\Code project\validation_ch7_v2\scripts\niveau4_rl_performance"

# 2. Vérifier aide
python entry_points/cli.py --help

# 3. Premier test rapide (5 min)
python entry_points/cli.py run --section section_7_6 --quick-test

# 4. Vérifier logs
Get-Content logs/section_7_6_rl_performance.log -Tail 20

# 5. Si OK, exécution complète
python entry_points/cli.py run --section section_7_6
```

### Utilisation Continue (avec cache)

```bash
# Le cache baseline sera réutilisé automatiquement
# Gain: ~60% temps (pas besoin recalculer baseline)

python entry_points/cli.py run --section section_7_6

# Si changement config RL seulement:
# - Cache baseline réutilisé ✅
# - Nouvel entraînement RL lancé ✅
# - Checkpoints avec nouveau hash config ✅
```

---

## Métriques Attendues

### Quick Test Mode (~5 min)
- **Baseline simulation**: 30-60 secondes
- **RL training (1000 steps)**: 2-3 minutes
- **Evaluation**: 30 seconds
- **Total**: 3-5 minutes

### Full Run (~2-3h avec GPU)
- **Baseline simulations** (4 scenarios): 5-10 minutes
- **RL training DQN** (100k timesteps): 1.5-2 hours
- **Evaluations**: 10-15 minutes
- **Total**: 2-3 hours

### Performance GPU
- **Environment step**: 0.2-0.6ms (Code_RL optimized)
- **DQN training**: ~40-50 steps/sec (GPU)
- **Cache hit**: <1ms (pickle load)

---

## Prochaines Étapes

### Après Validation Locale Réussie

1. **Tests Unitaires** (optionnel, 3-5h)
   - `tests/unit/test_code_rl_environment_adapter.py`
   - `tests/unit/test_code_rl_training_adapter.py`

2. **Déploiement Kaggle** (1h)
   - Package complet
   - Upload kernel
   - Run validation

3. **Analyse Résultats**
   - Comparaison baseline vs RL
   - Métriques performance
   - Visualisations

---

## Contact / Support

**Documentation complète**:
- `CORRECTION_ARCHITECTURALE_CODE_RL.md` - Analyse erreur + correction
- `CORRECTION_EXECUTEE_FINALE.md` - Résumé exécution
- `INTEGRATION_COMPLETE.md` - Status intégration complet

**Pour debugging**:
- Vérifier logs structurés: `logs/section_7_6_rl_performance.log`
- Activer mode verbose: Modifier `log_level: "DEBUG"` dans config YAML
- Utiliser `--quick-test` pour tests rapides

---

**Dernière mise à jour**: 2025-10-19 08:02
**Status**: ✅ SYSTÈME OPÉRATIONNEL
