# 🗺️ Checkpoint System - Visual Reference Guide

## 📂 Structure des Fichiers - Où Tout se Trouve

```
Code project/
│
├── validation_ch7/
│   └── scripts/
│       ├── validation_kaggle_manager.py    ← 🔧 CORE: Checkpoint logic ici!
│       ├── test_section_7_6_rl_performance.py    ← Training script
│       └── run_kaggle_validation_section_7_6.py  ← Launch script
│   
│   └── section_7_6_rl_performance/
│       └── data/
│           └── models/
│               ├── checkpoints/          ← 📍 DESTINATION: Checkpoints restaurés ICI
│               │   ├── *_checkpoint_50000_steps.zip
│               │   └── *_checkpoint_100000_steps.zip
│               ├── best_model/           ← Meilleur modèle
│               │   └── best_model.zip
│               └── training_metadata.json ← Métadonnées
│
├── validation_output/
│   └── results/
│       └── elonmj_arz-validation-*/      ← 📥 SOURCE: Kaggle télécharge ICI
│           └── section_7_6_rl_performance/
│               └── data/
│                   └── models/
│                       └── checkpoints/  ← Checkpoints de Kaggle
│
├── docs/
│   ├── CHECKPOINT_SYSTEM.md              ← 📖 Documentation complète
│   ├── CHECKPOINT_QUICKSTART.md          ← 🚀 Quick start
│   └── CHECKPOINT_IMPLEMENTATION_SUMMARY.md  ← ✅ Ce qui a été fait
│
└── verify_checkpoint_system.py           ← 🧪 Script de vérification
```

---

## 🔄 Flux de Données - Comment ça Marche

```
┌────────────────────────────────────────────────────────────────────┐
│                        KAGGLE GPU (Remote)                         │
│                                                                    │
│  Training → Checkpoint tous les 10k steps                         │
│                                                                    │
│  data/models/checkpoints/                                         │
│    ├── scenario_checkpoint_50000_steps.zip                        │
│    └── scenario_checkpoint_100000_steps.zip                       │
│                                                                    │
│  data/models/best_model/                                          │
│    └── best_model.zip                                             │
└────────────────────────────────────────────────────────────────────┘
                                ↓
                    📥 DOWNLOAD (automatic)
                    _retrieve_and_analyze_logs()
                                ↓
┌────────────────────────────────────────────────────────────────────┐
│                    LOCAL: validation_output/                       │
│                                                                    │
│  validation_output/results/{kernel}/section_7_6_rl_performance/   │
│    └── data/models/                                               │
│        ├── checkpoints/*.zip                                      │
│        ├── best_model/best_model.zip                              │
│        └── training_metadata.json                                 │
└────────────────────────────────────────────────────────────────────┘
                                ↓
                    🔄 RESTORE (automatic) 🆕
                    _restore_checkpoints_for_next_run()
                                ↓
┌────────────────────────────────────────────────────────────────────┐
│              LOCAL: validation_ch7/ (Training Location)            │
│                                                                    │
│  validation_ch7/section_7_6_rl_performance/data/models/           │
│    ├── checkpoints/                                               │
│    │   ├── *_checkpoint_50000_steps.zip  ← Training cherche ICI! │
│    │   └── *_checkpoint_100000_steps.zip                         │
│    ├── best_model/                                                │
│    │   └── best_model.zip                                         │
│    └── training_metadata.json                                     │
└────────────────────────────────────────────────────────────────────┘
                                ↓
                    ♻️ NEXT RUN
                    find_latest_checkpoint() → LOAD → CONTINUE!
```

---

## 🎯 Points Clés à Retenir

### 1. Deux Locations Différentes

| Location | Usage | Créé par |
|----------|-------|----------|
| `validation_output/results/` | Téléchargement depuis Kaggle | `_retrieve_and_analyze_logs()` |
| `validation_ch7/section_7_6/` | Training local/futur | `_restore_checkpoints_for_next_run()` 🆕 |

### 2. Workflow Automatique

```
Run 1 → Kaggle saves → Download → Restore → Ready for Run 2
Run 2 → Load latest → Continue → Save new → Download → Restore → Ready for Run 3
...
```

**TOUT est automatique!** Aucune action manuelle.

### 3. Fichiers Importants

| Fichier | Quand Utiliser |
|---------|---------------|
| `verify_checkpoint_system.py` | Avant premier run |
| `run_kaggle_validation_section_7_6.py` | Pour lancer training |
| `CHECKPOINT_SYSTEM.md` | Pour documentation complète |
| `CHECKPOINT_QUICKSTART.md` | Pour démarrage rapide |

---

## 🎬 Commandes Rapides

### Vérification
```powershell
# Vérifier l'implémentation
python verify_checkpoint_system.py

# Vérifier les checkpoints
Get-ChildItem -Path "validation_ch7\section_7_6_rl_performance\data\models\checkpoints" -Recurse
```

### Lancement
```powershell
# Quick test (15 min)
python validation_ch7\scripts\run_kaggle_validation_section_7_6.py --quick

# Full training (3-4h)
python validation_ch7\scripts\run_kaggle_validation_section_7_6.py
```

### Maintenance
```powershell
# Voir métadonnées
Get-Content "validation_ch7\section_7_6_rl_performance\data\models\training_metadata.json" | ConvertFrom-Json

# Supprimer checkpoints (fresh start)
Remove-Item -Path "validation_ch7\section_7_6_rl_performance\data\models\checkpoints\*" -Recurse -Force
```

---

## 🔍 Logs à Surveiller

### Succès de Restauration
```
[CHECKPOINT] ====== CHECKPOINT RESTORATION ======
[CHECKPOINT] Found 2 checkpoint(s) to restore:
[CHECKPOINT]   ✓ traffic_light_control_checkpoint_50000_steps.zip (15.3 MB)
[CHECKPOINT]   ✓ traffic_light_control_checkpoint_100000_steps.zip (15.4 MB)
[CHECKPOINT] ✅ Successfully restored 2 file(s)
```

### Premier Run (Normal)
```
[CHECKPOINT] No checkpoints found in validation_output/results/...
[CHECKPOINT] This is normal for first run or if training didn't reach checkpoint threshold
```

### Incompatibilité (Attention!)
```
[CHECKPOINT] ⚠️  INCOMPATIBILITY DETECTED:
[CHECKPOINT]   - Architecture: checkpoint=[64, 64], current=[128, 128]
[CHECKPOINT] Recommendation: Delete checkpoints to start fresh
```

---

## 🆘 Troubleshooting Rapide

### Problème → Solution

| Symptôme | Diagnostic | Solution |
|----------|-----------|----------|
| "No checkpoints found" | Premier run OU training < 10k steps | Normal, continuez |
| "Incompatibility detected" | Architecture changée | Supprimer checkpoints |
| Training ne reprend pas | Checkpoints pas dans bon dossier | Vérifier chemins |
| Erreur au chargement | Fichier corrompu | Re-télécharger |

### Debug Steps

1. **Vérifier présence:**
   ```powershell
   Test-Path "validation_ch7\section_7_6_rl_performance\data\models\checkpoints\*.zip"
   ```

2. **Vérifier pattern:**
   Doit être: `{scenario}_checkpoint_{steps}_steps.zip`

3. **Vérifier logs Kaggle:**
   Chercher "Loading checkpoint" ou "Starting fresh training"

---

## 📊 Taille Typique des Fichiers

| Type | Taille Typique | Fréquence |
|------|----------------|-----------|
| Checkpoint | 15-20 MB | Tous les 10k steps |
| Best Model | 7-10 MB | Quand amélioration |
| Metadata | < 1 KB | Avec checkpoints |

**Total par run:** ~40-50 MB

---

## 🎓 Scénarios d'Usage

### Scénario 1: Quick Test → Full

```powershell
# 1. Quick test (100 steps, 15 min)
python validation_ch7\scripts\run_kaggle_validation_section_7_6.py --quick

# 2. Full (5000 steps, 3h) - reprend à 100
python validation_ch7\scripts\run_kaggle_validation_section_7_6.py
```

### Scénario 2: Continue Training

```powershell
# 1. Run 1: 5000 steps
python validation_ch7\scripts\run_kaggle_validation_section_7_6.py

# 2. Modifier total_timesteps=10000 dans test script

# 3. Run 2: Continue 5000→10000
python validation_ch7\scripts\run_kaggle_validation_section_7_6.py
```

### Scénario 3: Fresh Start

```powershell
# Supprimer anciens checkpoints
Remove-Item -Path "validation_ch7\section_7_6_rl_performance\data\models\checkpoints\*" -Recurse -Force

# Nouveau training
python validation_ch7\scripts\run_kaggle_validation_section_7_6.py
```

---

## ✅ Quick Checklist

Avant chaque run:
- [ ] `python verify_checkpoint_system.py` → ✅
- [ ] Checkpoints présents? (si continuation)
- [ ] Compatibilité OK? (si changements)

Après chaque run:
- [ ] Logs montrent restauration? 
- [ ] Checkpoints dans `validation_ch7/`?
- [ ] Metadata JSON valide?

---

## 🎉 Résumé en 3 Points

1. **AUTOMATIQUE** - Aucune action manuelle requise
2. **ROBUSTE** - Validation de compatibilité intégrée  
3. **SIMPLE** - Lance et oublie!

**Le système gère tout pour vous!** ✨

---

Pour plus de détails, consultez `docs/CHECKPOINT_SYSTEM.md`
