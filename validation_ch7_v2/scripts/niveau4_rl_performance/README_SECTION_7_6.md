# Section 7.6: Validation RL Performance

## 🎯 Objectif

Produire les résultats pour la **Section 7.6** de la thèse (Niveau 4 de la Pyramide de Validation):
- **Revendication R5**: L'agent RL surpasse les stratégies de contrôle traditionnelles

## 📁 Architecture

```
niveau4_rl_performance/
├── run_section_7_6.py          ⭐ SCRIPT FINAL UNIQUE
├── rl_training.py              (implémentation DQN training)
├── rl_evaluation.py            (implémentation evaluation)
├── section_7_6_results/        (généré à l'exécution)
│   ├── figures/
│   │   ├── rl_performance_comparison.png
│   │   └── rl_learning_curve_revised.png
│   ├── latex/
│   │   ├── tab_rl_performance_gains_revised.tex
│   │   └── section_7_6_content.tex   ⭐ pour \input
│   └── data/
│       └── section_7_6_results.json
```

## 🚀 Usage

### Quick Test (5 minutes, CPU)

```bash
cd validation_ch7_v2/scripts/niveau4_rl_performance
python run_section_7_6.py --quick --device cpu
```

- Timesteps: 100
- Episodes: 1
- Durée: ~5 minutes
- **Usage**: Vérifier que tout fonctionne avant Kaggle GPU

### Full Validation (3 heures, GPU Kaggle)

```bash
python run_section_7_6.py --device gpu
```

- Timesteps: 5000
- Episodes: 3
- Durée: ~3-4 heures sur GPU
- **Usage**: Résultats finaux pour la thèse

### Custom Configuration

```bash
python run_section_7_6.py --timesteps 10000 --episodes 5 --device gpu
```

## 📊 Outputs Générés

### 1. Figures PNG (pour thèse)

- `rl_performance_comparison.png`: Comparaison 3 métriques (Travel Time, Throughput, Queue)
- `rl_learning_curve_revised.png`: Courbe d'apprentissage DQN

### 2. Tableaux LaTeX

- `tab_rl_performance_gains_revised.tex`: Tableau gains quantitatifs

### 3. Contenu LaTeX Complet

- `section_7_6_content.tex`: Prêt pour `\input` dans la thèse

### 4. Données Brutes

- `section_7_6_results.json`: Résultats complets au format JSON

## 📝 Intégration Thèse

Dans `section7_validation_nouvelle_version.tex`:

```latex
\subsection{Niveau 4 : Validation de l'Optimisation par Apprentissage par Renforcement}
\label{sec:validation_rl}

% ✅ Inclure le contenu auto-généré
\input{validation_output/section_7_6/latex/section_7_6_content.tex}
```

## ⚙️ Pipeline Complet

Le script `run_section_7_6.py` exécute:

1. **Phase 1: Training** (60-180 min sur GPU)
   - Entraîne agent DQN sur `TrafficSignalEnvDirect`
   - Sauvegarde modèle entraîné
   
2. **Phase 2: Evaluation** (5-20 min)
   - Compare RL vs Baseline (Fixed-time 60s)
   - Simulations ARZ réelles (pas de mocks!)
   - Calcule métriques: Travel Time, Throughput, Queue Length
   
3. **Phase 3: Outputs** (<1 min)
   - Génère figures PNG haute résolution (300 DPI)
   - Génère tableaux LaTeX formatés
   - Génère fichier `.tex` prêt pour `\input`

## ✅ Validation Résultats

### Critères de Succès

- ✅ Amélioration Travel Time > 0%
- ✅ Amélioration Throughput > 0%
- ✅ Réduction Queue Length > 0%
- ✅ Significativité statistique (p < 0.001)

### Exemple Résultats Attendus

```
📊 RÉSUMÉ:
   ✅ Travel Time: +28.7%
   ✅ Throughput: +15.2%
   ✅ Queue Reduction: +22.3%
```

## 🐛 Debugging

### Problème: GPU non détecté

```bash
# Forcer CPU
python run_section_7_6.py --quick --device cpu
```

### Problème: Imports échouent

```bash
# Vérifier Python path
cd d:\Projets\Alibi\Code project
python -c "import sys; print('\n'.join(sys.path))"

# Vérifier Code_RL disponible
python -c "from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect; print('OK')"
```

### Problème: Training trop long

```bash
# Réduire timesteps pour test
python run_section_7_6.py --timesteps 50 --episodes 1 --device cpu
```

## 📚 Références Code

- **Training**: `rl_training.py` → `train_rl_agent_for_validation()`
- **Evaluation**: `rl_evaluation.py` → `evaluate_traffic_performance()`
- **Code_RL**: `Code_RL/src/env/traffic_signal_env_direct.py`
- **ARZ Model**: `arz_model/` (simulateur trafic)

## 🎓 Contexte Thèse

### Revendication R5

> "Les agents RL entraînés sur le jumeau numérique surpassent les stratégies de contrôle traditionnelles."

### Baseline

- **Contrôle à temps fixe 60s** (GREEN/RED alternance)
- Représente la pratique actuelle à Lagos (Bénin)
- Pas de systèmes adaptatifs/actuated déployés

### Agent RL

- **Algorithme**: DQN (Deep Q-Network)
- **Environment**: TrafficSignalEnvDirect (couplage direct ARZ)
- **Decision Interval**: 15s
- **Hyperparameters**: Code_RL defaults (lr=1e-3, batch=32, tau=1.0)

## ⏱️ Estimations Durée

| Mode | Timesteps | Episodes | CPU | GPU (Kaggle) |
|------|-----------|----------|-----|--------------|
| Quick | 100 | 1 | 5 min | <1 min |
| Normal | 5000 | 3 | ~6 heures | 3-4 heures |
| Full | 10000 | 5 | ~12 heures | 6-8 heures |

## ✨ Points Clés

1. **UN SEUL fichier final**: `run_section_7_6.py` (plus de quick_test_* séparés)
2. **Mode --quick intégré**: Paramètre pour tests rapides
3. **Outputs prêts pour thèse**: Figures PNG + LaTeX + JSON
4. **REAL implementation**: TrainingSignalEnvDirect + DQN (pas de mocks!)
5. **Reproductible**: Même script local (CPU) et Kaggle (GPU)

---

**Auteur**: Josaphat ADJE  
**Date**: 2025-10-19  
**Version**: 1.0 (Script Final Unique)
