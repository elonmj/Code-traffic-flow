# Guide de Reproductibilité - Système RL Lagos

## 🚀 Installation Rapide

```bash
# 1. Cloner/télécharger le projet
cd Code_RL

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Tester l'installation
python test_lagos.py
```

## 📋 Checklist de Reproductibilité

### ✅ Environnement Python Requis
- [ ] Python 3.8+ installé
- [ ] pip à jour (`python -m pip install --upgrade pip`)
- [ ] Virtual environment recommandé

### ✅ Installation Dépendances
```bash
pip install stable-baselines3==2.0.0
pip install gymnasium==0.29.0  
pip install torch>=1.13.0
pip install numpy pandas scipy
pip install pyyaml tqdm matplotlib pytest
```

### ✅ Structure Fichiers Critiques
```
Code_RL/
├── src/                    ✅ Code source complet
├── configs/                ✅ Toutes configurations YAML
├── data/                   ✅ Données CSV Victoria Island
├── tests/                  ✅ Tests unitaires
├── requirements.txt        ✅ Dépendances exactes  
├── README.md              ✅ Documentation complète
├── demo.py                ✅ Démonstrations
├── train.py               ✅ Entraînement principal
├── test_lagos.py          ✅ Tests Lagos
├── analyze_corridor.py    ✅ Analyse données
└── adapt_lagos.py         ✅ Génération configs
```

## 🧪 Tests de Validation

### Test 1: Configuration Lagos
```bash
python test_lagos.py
# Attendu: "🎉 CONFIGURATION LAGOS OPÉRATIONNELLE !"
```

### Test 2: Démonstration Système
```bash
python demo.py 1
# Attendu: "✓ All components working correctly!"
```

### Test 3: Entraînement Court
```bash
python train.py --config lagos --use-mock --timesteps 100
# Attendu: Training completed, evaluation results
```

### Test 4: Tests Unitaires
```bash
pytest tests/ -v
# Attendu: All tests pass
```

## 🔧 Reproductibilité Exacte

### Versions Testées
- **OS**: Windows 10/11, Ubuntu 20.04+
- **Python**: 3.8.10, 3.9.16, 3.10.11
- **PyTorch**: 1.13.0, 2.0.0  
- **Stable-Baselines3**: 2.0.0

### Seeds Reproductibles
```python
# Dans les scripts, les seeds sont fixées:
env.reset(seed=42)  # Environnement
np.random.seed(42)  # NumPy
torch.manual_seed(42)  # PyTorch
```

### Données Exactes
- `data/fichier_de_travail_corridor.csv`: 70 segments Victoria Island
- `data/donnees_vitesse_historique.csv`: Données historiques vitesses
- Configs générées automatiquement par `adapt_lagos.py`

## 🎯 Résultats Attendus

### Performance Baseline DQN Lagos
```
Agent DQN:
- Récompense moyenne: -0.01 ± 0.00
- Changements phase/épisode: ~90
- Temps entraînement: ~1.6s pour 1000 steps

Baseline Fixe:
- Changements phase/épisode: ~59  
- Cycles fixes 60s/phase
```

### Métriques Reproductibles
1. **test_lagos.py**: 8 branches réseau réel, obs shape (43,)
2. **demo.py 1**: 4 composants validés
3. **train.py**: Convergence <2000 timesteps

## 🐛 Résolution Problèmes

### Erreur Import
```bash
# Solution: Ajouter src au PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:./src"
# Ou sur Windows:
set PYTHONPATH=%PYTHONPATH%;.\src
```

### Erreur Configuration
```bash
# Regénérer toutes les configs Lagos
python adapt_lagos.py
python analyze_corridor.py
```

### Erreur Dépendances
```bash
# Réinstaller dépendances exactes
pip install -r requirements.txt --force-reinstall
```

## 📊 Validation Complète

### Script de Validation Globale
```bash
# Test complet du système (prend ~2 minutes)
python test_lagos.py && \
python demo.py 1 && \
python train.py --config lagos --use-mock --timesteps 500 && \
echo "✅ Système entièrement validé !"
```

### Outputs Attendus
1. **Configuration chargée**: Tous les YAML sans erreur
2. **Réseau généré**: 8 branches Victoria Island  
3. **Environnement créé**: Observation space (43,)
4. **Agent entraîné**: Récompense stable autour -0.01

---

**Note**: Ce guide garantit la reproductibilité exacte des résultats Lagos sur tout environnement compatible.
