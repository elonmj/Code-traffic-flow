# Kaggle RL Training Deployment

## 🎯 Objectif

Entraîner un agent DQN sur le réseau complet Victoria Island (70 segments, 8 feux de circulation) en utilisant le GPU Kaggle.

## 🚀 Quick Start

### Test rapide (300 steps, ~5 minutes)
```bash
python launch_kaggle_training.py --timesteps 300 --timeout 600
```

### Training court (10k steps, ~30 minutes)
```bash
python launch_kaggle_training.py --timesteps 10000 --timeout 1800
```

### Training complet (100k steps, ~3 heures)
```bash
python launch_kaggle_training.py --timesteps 100000 --timeout 10800
```

## 📋 Architecture

### Réseau Victoria Island
- **70 segments** de route
- **60 nœuds** total
  - 8 nœuds **signalisés** avec feux de circulation 🚦
  - 8 nœuds **boundary** (entrée/sortie)
  - 44 nœuds **junction** (intersections simples)

### Configuration Traffic Lights (West Africa)
- Cycle time: 90s
- Green time: 35s
- Amber time: 3s
- Red time: 52s

### Résolution de grille
- 4 cellules / 100m (dx = 25m)
- Équilibre performance/précision pour Kaggle

## 🔧 Workflow Automatisé

Le système `kaggle_runner` gère automatiquement :

1. **Git Sync** : Commit et push des changements locaux
2. **Kernel Update** : Création/mise à jour du kernel Kaggle
3. **Execution** : Lance l'entraînement sur GPU Kaggle
4. **Monitoring** : Suit l'exécution en temps réel
5. **Artifacts** : Télécharge automatiquement les résultats

## 📦 Artifacts

Les résultats sont sauvegardés dans `/kaggle/working/` :

```
/kaggle/working/
├── final_model.zip          # Modèle DQN entraîné
├── training_metrics.json    # Métriques d'entraînement
├── tensorboard/             # Logs TensorBoard
│   └── events.out.tfevents.*
└── checkpoints/             # Checkpoints intermédiaires
    ├── best_model.zip
    └── model_<timestep>.zip
```

## 🔍 Monitoring

Le script affiche en temps réel :
- Configuration du réseau (segments, nœuds, feux)
- Progression de l'entraînement
- Métriques (reward, loss, exploration)
- Temps écoulé et estimation de fin

## ⚙️ Paramètres Avancés

### Scénarios disponibles
- `quick_test` : 2 min, grille grossière (2 cells/100m)
- `victoria_island` : 7.5 min, grille standard (4 cells/100m)
- `extended` : 1h, grille fine (6 cells/100m)

### Commande manuelle via executor
```bash
python kaggle_runner/executor.py \
  --target kaggle_runner/experiments/rl_training_victoria_island.py \
  --timeout 3600 \
  --commit-message "Test RL training 10k steps"
```

## 📊 Résultats Attendus

### Test rapide (300 steps)
- Durée : ~5 minutes
- But : Vérifier que tout fonctionne
- Reward : Pas encore convergé

### Training court (10k steps)
- Durée : ~30 minutes
- But : Premiers signes d'apprentissage
- Reward : Amélioration visible

### Training complet (100k+ steps)
- Durée : 3h+
- But : Convergence complète
- Reward : Performance stable

## 🐛 Debugging

### Vérification locale

Test de la configuration :
```bash
python test_full_network.py
```

Test de l'entraînement (local, CPU) :
```bash
python kaggle_runner/experiments/rl_training_victoria_island.py \
  --timesteps 100 \
  --device cpu
```

### Logs Kaggle

Les logs sont automatiquement téléchargés dans :
```
kaggle_runner/results/<kernel-slug>/
├── output.log
└── errors.log
```

## ✅ Checklist Avant Déploiement

- [ ] Branche `experiment/kaggle-rl-training` créée
- [ ] Changements committés et pushés
- [ ] Test local réussi (`test_full_network.py`)
- [ ] Kaggle API configurée (`~/.kaggle/kaggle.json`)
- [ ] Timeout suffisant pour le nombre de steps

## 🎉 Prochaines Étapes

Après un training réussi :

1. Analyser les métriques dans TensorBoard
2. Évaluer le modèle sur le réseau complet
3. Visualiser les décisions de l'agent aux feux
4. Comparer avec baseline (feux fixes)
5. Itérer sur les hyperparamètres

---

**Note** : Le système utilise maintenant le RÉSEAU COMPLET avec tous les feux de circulation, exactement comme `main_network_simulation.py`. L'agent RL apprend à contrôler 8 feux simultanément pour optimiser le trafic sur les 70 segments.
