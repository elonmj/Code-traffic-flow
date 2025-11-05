# GPU Stability Test - Kaggle Execution Ready

## 📋 Contexte

**Problème identifié**: Instabilité numérique avec inflow BC à haute vitesse
- CPU + dt=0.001s → Explosion v→367 m/s quand v_m ≥ 6.5 m/s
- Hypothèse: GPU + dt plus petit pourrait résoudre

**Recherche web effectuée**: ✅
- Springer, ETH Zurich: GPU permet dt plus petits grâce au speedup
- WENO5-GPU optimisé pour équations hyperboliques
- Stabilité numérique améliorée avec dt réduit

## 🚀 Architecture Mise en Place

### Fichiers Créés

1. **`run_gpu_stability_test.py`** - Launcher (style section 7.6)
   - Location: `validation_ch7/scripts/`
   - Usage: `python run_gpu_stability_test.py` ou `--quick`
   - Délègue à `validation_cli.py`

2. **`test_gpu_stability.py`** - Test principal exécuté sur Kaggle
   - Location: `validation_ch7/scripts/`
   - Config: v_m=10.0 m/s, dt=0.0001s, GPU mode
   - Outputs: PNG + JSON metrics + session_summary.json

3. **Enregistré dans `validation_kaggle_manager.py`**
   - Section: `gpu_stability_test`
   - Durée estimée: 15 minutes (full), 5 minutes (quick)

## ⚙️ Configuration du Test

### Test Complet
```bash
python validation_ch7/scripts/run_gpu_stability_test.py
```
- Duration: 15s simulées (150,000 timesteps!)
- Timestep: dt=0.0001s (10x plus petit)
- BC inflow: v_m=10.0 m/s
- Runtime estimé: ~15 minutes sur Kaggle GPU

### Quick Test
```bash
python validation_ch7/scripts/run_gpu_stability_test.py --quick
```
- Duration: 5s simulées (50,000 timesteps)
- Timestep: dt=0.0001s (identique)
- Runtime estimé: ~5 minutes sur Kaggle GPU

## 📊 Critères de Succès

✅ **SUCCÈS** si:
- v_max final < 20 m/s (stable)
- rho_max final > 0.08 (congestion formée)
- Pas d'explosion numérique

❌ **ÉCHEC** si:
- v_max > 100 m/s (explosion détectée)
- Instabilité persiste même avec GPU + dt petit

## 🔧 Prochaines Étapes

### Immédiat (Recommandé)
```bash
cd "d:\Projets\Alibi\Code project\validation_ch7\scripts"
python run_gpu_stability_test.py --quick
```

Ceci va:
1. ✅ Pousser le code sur GitHub
2. ✅ Créer un kernel Kaggle avec GPU
3. ✅ Cloner le repo sur Kaggle
4. ✅ Exécuter test_gpu_stability.py avec GPU
5. ✅ Télécharger les résultats (PNG + metrics)

### Analyse des Résultats

Après exécution, vérifier:
- `validation_output/.../gpu_stability_test/gpu_stability_evolution.png`
- `validation_output/.../gpu_stability_test/gpu_stability_metrics.json`

Métriques clés:
```json
{
  "final_v_max": 12.5,  // <20 = stable ✅
  "final_rho_max": 0.15, // >0.08 = congestion ✅
  "stable": true,
  "success": true
}
```

## 🎯 Hypothèse à Vérifier

**H0**: GPU + dt=0.0001s résout l'instabilité v_m=10.0 m/s

**Résultats attendus**:
- ✅ **Favorable**: v_max reste < 20 m/s → Solution confirmée!
- ❌ **Défavorable**: Instabilité persiste → Problème plus profond (BC? Schéma numérique?)

## 📝 Notes Techniques

### Blocage GPU Local
- **Status**: GPU local non-fonctionnel (CUDA 13.0 + Numba 0.62 + WDDM)
- **Solution**: Utiliser Kaggle GPU (CUDA 11.x compatible)
- **Architecture**: Code prêt (pure Numba CUDA, pas de CuPy)

### Code Refactoring Effectué
- ✅ NetworkGrid: `cuda.to_device()` au lieu de `cp.asarray()`
- ✅ time_integration: `strang_splitting_step_gpu()` pure Numba
- ✅ Suppression CuPy (causait conflit CUDA)
- ✅ GPU kernels: `@cuda.jit` (Numba native)

## 🚨 Rappel Important

**Ne pas exécuter en local** - GPU non-fonctionnel sur ce système
**Utiliser Kaggle** - Architecture prête, juste lancer le script!

---

**Date**: 2025-11-04
**Status**: PRÊT POUR KAGGLE ✅
**Commande**: `python validation_ch7/scripts/run_gpu_stability_test.py --quick`
