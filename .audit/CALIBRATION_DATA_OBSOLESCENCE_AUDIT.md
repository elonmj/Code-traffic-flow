# Audit d'Obsolescence : Dossier `arz_model/calibration/data`

**Date**: October 22, 2025  
**Status**: LEGACY (2024-2025 Calibration Phase) ⚠️  
**Action Required**: ARCHIVAGE PRIORITAIRE

---

## Executive Summary

Le dossier **`arz_model/calibration/data`** est **majoritairement obsolète** et incompatible avec la nouvelle architecture de validation Ch7 (2025+). Cependant, il existe **une dépendance active** via `test_section_7_4_calibration.py` qui implémente la **calibration temps-réel**.

### Recommandation
🔴 **ARCHIVER 95% du contenu, CONSERVER avec cautèle `real_data_loader.py`**

---

## Analyse Détaillée des Modules

### 1. **digital_twin_calibrator.py** ❌ LEGACY
**Status**: Obsolète - Phase 1.3 Calibration  
**Size**: 668 lignes

**Fonction**:
- Calibration 2 phases (paramètres ARZ → R(x) route)
- Intégration TomTom  
- Multi-objectif: MAPE, RMSE, GEH

**Pourquoi Legacy**:
- ✅ Phase de calibration terminée (2024-2025)
- ✅ Données Victoria Island calibrées
- ✅ ARZ model a ses paramètres optimisés
- ❌ Ch7 validation ne calibre RIEN
- ❌ Ch7 teste la PHYSIQUE véhicule abstraite
- ❌ Ch7 utilise multi-scénarios, pas corridors réels

**Utilisation**:
```python
# Seulement dans test_phase13_calibration.py (même fichier legacy)
from ..calibration.data.digital_twin_calibrator import DigitalTwinCalibrator
```

**Action**: ➡️ ARCHIVER → `_archive/2024_phase13_calibration/`

---

### 2. **spatiotemporal_validator.py** ❌ LEGACY
**Status**: Obsolète - Validation spatio-temporelle ARZ  
**Size**: 922 lignes

**Fonction**:
- Validation croisée temporelle (TimeSeriesSplit)
- Analyse spatiale par segment highway
- Heatmaps de performance
- Critères GEH < 5, MAPE < 15%, RMSE < 10 km/h

**Pourquoi Legacy**:
- ✅ Métriques routières (GEH, MAPE) ≠ Physique véhicule
- ❌ Ch7 valide trajectoires, pas débits routiers
- ❌ Ch7 utilise niveaux 1,2,3 génériques
- ❌ Ch7 pas intéressé par corridors spécifiques
- ❌ Ch7 multi-kernel GPU, pas cross-validation locale

**Utilisation**:
```python
# Seulement dans test_phase13_calibration.py
from ..calibration.data.spatiotemporal_validator import SpatioTemporalValidator
```

**Action**: ➡️ ARCHIVER → `_archive/2024_phase13_calibration/`

---

### 3. **real_data_loader.py** ⚠️ PARTIALLY USED
**Status**: Actif - Dépendance réelle  
**Size**: 398 lignes

**Fonction**:
- Charge données TomTom CSV (donnees_trafic_75_segments.csv)
- Mappe à corridors Victoria Island
- Validation qualité données

**Pourquoi Actif**:
- ✅ Importé et utilisé dans `test_section_7_4_calibration.py`
- ✅ Fichier CSV existe: `donnees_trafic_75_segments.csv`
- ✅ Réseau JSON existe: `groups/victoria_island_corridor.json`

**Utilisation Active**:
```python
# test_section_7_4_calibration.py (ligne 28, 64)
from arz_model.calibration.data.real_data_loader import RealDataLoader

self.data_loader = RealDataLoader(
    csv_file=str(self.csv_data_file),
    network_json=str(self.network_json_file),
    min_confidence=0.8
)
```

**Problème Identifié**:
- `test_section_7_4_calibration.py` est un **test de calibration temps-réel**
- Il n'est PAS clairement intégré à la nouvelle architecture Ch7
- Deux chemins possibles:
  - 🔴 Il est lui-même **legacy** et non utilisé
  - 🟡 Il est **utilisé mais jamais mis à jour** dans la validation Ch7

**Action**: ➡️ VÉRIFIER d'abord si `test_section_7_4_calibration.py` est toujours actif dans le pipeline

---

### 4. **tomtom_collector.py** ❌ LEGACY
**Status**: Obsolète - Collecte de données  
**Size**: Unknown

**Fonction**: Collecte données TomTom

**Pourquoi Legacy**:
- ✅ Données Victoria Island déjà collectées
- ❌ Ch7 ne collecte pas de données
- ❌ Ch7 utilise scénarios abstraits

**Action**: ➡️ ARCHIVER → `_archive/2024_phase13_calibration/`

---

### 5. **speed_processor.py** ❌ LIKELY LEGACY
**Status**: Probablement obsolète

**Fonction**: Traitement données vitesse

**Pourquoi Likely Legacy**:
- Pas d'import détecté hors de `calibration/data`
- Nom spécifique aux vitesses routières, pas trajectoires véhicule

**Action**: ➡️ ARCHIVER avec vérification

---

### 6. **group_manager.py** ❌ LEGACY
**Status**: Obsolète - Gestion groupes segments

**Utilisation**:
```python
# Uniquement dans __init__.py pour imports
from .group_manager import GroupManager, NetworkGroup, SegmentInfo
```

**Action**: ➡️ ARCHIVER → `_archive/2024_phase13_calibration/`

---

### 7. **victoria_island_config.py** ❌ LEGACY
**Status**: Obsolète - Configuration spécifique corridor

**Utilisation**:
- Seulement dans calibration/data
- Référence corridor Victoria Island

**Action**: ➡️ ARCHIVER → `_archive/2024_phase13_calibration/`

---

### 8. **corridor_loader.py** ❌ LEGACY
**Status**: Obsolète - NI MÊME UTILISÉ

**Utilisation**:
```python
# Dans __init__.py - COMMENTÉ
# from .corridor_loader import CorridorLoader  # <- TODO!
```

**Action**: ➡️ SUPPRIMER (jamais implémenté)

---

### 9. **calibration_results_manager.py** ❌ LEGACY
**Status**: Obsolète - Gestion résultats calibration

**Action**: ➡️ ARCHIVER → `_archive/2024_phase13_calibration/`

---

### 10. **test_real_data_loader.py** ❌ LEGACY
**Status**: Obsolète - Tests calibration

**Action**: ➡️ ARCHIVER → `_archive/2024_phase13_calibration/`

---

### 11. **groups/victoria_island_corridor.json** ⚠️ ASSET PARTAGÉ
**Status**: Utilisé si data_loader actif

**Fichier JSON**:
- Configuration corridor Victoria Island
- 75 segments routiers

**Action**: Ne SUPPRIMER que si `real_data_loader.py` archivé

---

### 12. **results/** ⚠️ RÉSULTATS HISTORIQUES
**Status**: Archive d'exécution

**Fichiers**:
- `test_group_calibration_20250908_124713.json`
- `victoria_island_corridor_calibration_20250908_162605.json`

**Action**: ➡️ ARCHIVER → `_archive/2024_phase13_calibration/results/`

---

## Dépendances Externes

### Qui importe `calibration/data` ?

#### Active ✅
```python
# validation_ch7/scripts/test_section_7_4_calibration.py
from arz_model.calibration.data.real_data_loader import RealDataLoader
```

#### Legacy (non exécuté) ❌
```python
# arz_model/tests/test_phase13_calibration.py
from ..calibration.data.tomtom_collector import TomTomDataCollector
from ..calibration.data.digital_twin_calibrator import DigitalTwinCalibrator
from ..calibration.data.spatiotemporal_validator import SpatioTemporalValidator
```

#### Historique 📝
```python
# validation_output/results/ (logs)
# victoria_island_corridor.json utilisé dans log 2025-10-04
```

---

## Nouvelle Architecture (Ch7) vs Ancienne (Calibration)

### Comparaison Critique

| Aspect | Calibration (2024-2025) | Ch7 Validation (2025+) |
|--------|-------|------|
| **Données** | Réelles TomTom Victoria Island | Scénarios abstraits synthétiques |
| **Validé** | Débits, vitesses par corridor | Physique véhicule, trajectoires |
| **Métriques** | GEH, MAPE, RMSE | Erreurs position, vélocité, accélération |
| **Géographie** | Corridor spécifique 75 segments | Générique, n'importe quel réseau |
| **Exécution** | Séquentielle, calibration lente | Multi-kernel GPU parallelisée |
| **Cache** | Simple, spécifique | Versionné, distribué |
| **Objectif** | Optimiser paramètres ARZ | Valider équations différentielles |

---

## Plan d'Action Recommandé

### Phase 1: Vérification Urgente
```bash
# 1. Est-ce que test_section_7_4_calibration.py est utilisé ?
grep -r "test_section_7_4" validation_ch7/scripts/
grep -r "section_7_4" validation_kaggle_manager.py

# 2. Est-ce que donnees_trafic_75_segments.csv est utilisé?
grep -r "donnees_trafic_75_segments" .
grep -r "victoria_island_corridor.json" .
```

### Phase 2: Archivage Conditionnel
```bash
# Si test_section_7_4 n'est PAS utilisé:
mkdir -p _archive/2024_phase13_calibration/
mv arz_model/calibration/data/* _archive/2024_phase13_calibration/
rm -rf arz_model/calibration/data/
rm -rf donnees_trafic_75_segments.csv
git rm -r arz_model/calibration/data/
git commit -m "Archive: Move 2024 calibration phase to legacy storage"

# Si real_data_loader EST utilisé:
mkdir -p _archive/2024_calibration_legacy/
mv arz_model/calibration/data/{calibration_results_manager,corridor_loader,digital_twin_calibrator,group_manager,spatiotemporal_validator,speed_processor,test_real_data_loader,tomtom_collector,victoria_island_config,results}.py _archive/
mv arz_model/calibration/data/groups _archive/
# Conserver seulement: real_data_loader.py, __init__.py (minimaliste)
```

### Phase 3: Nettoyage
```bash
# Supprimer imports morte dans arz_model/calibration/data/__init__.py
# Mettre à jour test_section_7_4_calibration.py si encore actif
# Vérifier que validation_ch7 fonctionne toujours
```

---

## Risques & Dépendances

### 🔴 Risque: test_section_7_4_calibration.py Actif
**Impact**: Cannot archive `real_data_loader.py`  
**Solution**: 
1. Vérifier si test_section_7_4 est exécuté dans validation_kaggle_manager
2. Si NON: archiver tout
3. Si OUI: documenter dépendance, planifier migration

### 🟡 Risque: Données CSV Manquantes
**Impact**: RealDataLoader fail si CSV absent  
**Solution**: Vérifier que `donnees_trafic_75_segments.csv` existe

### 🟢 No Risk: Isolation
**Impact**: L'archivage n'affecte pas Ch7 Niveau 3  
**Raison**: Niveau 3 ne dépend pas de calibration/data

---

## Conclusion

### Verdict
🔴 **Le dossier `arz_model/calibration/data` EST OBSOLÈTE**

**95% du contenu** est du legacy 2024-2025 et peut être archivé sans impact sur la nouvelle architecture Ch7.

**5% du contenu** (`real_data_loader.py`) a une dépendance active potentielle qui doit être vérifiée.

### Prochaine Étape
1. ✅ Vérifier si `test_section_7_4_calibration.py` est actif dans le pipeline
2. ➡️ Si NON → ARCHIVER TOUT
3. ➡️ Si OUI → Documenter dépendance et planifier migration vers Ch7

---

## Fichiers d'Archive Suggérés

```
_archive/2024_phase13_calibration/
├── digital_twin_calibrator.py
├── spatiotemporal_validator.py
├── tomtom_collector.py
├── speed_processor.py
├── group_manager.py
├── victoria_island_config.py
├── calibration_results_manager.py
├── test_real_data_loader.py
├── groups/
│   └── victoria_island_corridor.json
├── results/
│   ├── test_group_calibration_20250908_124713.json
│   └── victoria_island_corridor_calibration_20250908_162605.json
└── README_LEGACY.md (explique ce que c'était)
```

---

**Auteur**: Architecture Audit  
**Classification**: LEGACY ANALYSIS  
**Archivable**: YES (95%), WITH CAUTION (5%)
