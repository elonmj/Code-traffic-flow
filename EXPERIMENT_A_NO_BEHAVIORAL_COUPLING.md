# 🔬 EXPERIMENT A: Test sans Coupling Comportemental θ_k

**Date**: 4 novembre 2025  
**Branch**: `experiment/no-behavioral-coupling`  
**Commit**: 25be633  
**Status**: ⏳ EN ATTENTE EXÉCUTION KAGGLE

---

## 🎯 OBJECTIF

Tester l'hypothèse que la **variable Lagrangienne θ_k** (mémoire comportementale) cause l'instabilité à haute vitesse en **violant la causalité** dans le solveur Eulérien.

---

## 🔧 MODIFICATIONS APPORTÉES

### 1. **NetworkGrid.step() - Désactivation du coupling**

**Fichier**: `arz_model/network/network_grid.py` ligne 590

```python
# AVANT (Kernel 10 - AVEC coupling):
self._resolve_node_coupling(current_time)

# APRÈS (Kernel 11 - SANS coupling):
# self._resolve_node_coupling(current_time)  # ← DISABLED FOR EXPERIMENT A
```

**Impact**: 
- ❌ **Désactive** la transmission de mémoire θ_k entre jonctions
- ❌ **Désactive** le coupling phénoménologique (Section 4.2.2 thèse)
- ✅ **Conserve** l'évolution des segments (Step 1: strang_splitting_step_gpu)
- ✅ **Conserve** les feux de circulation (Step 3: _update_traffic_lights)

### 2. **test_gpu_stability.py - Documentation expérience**

**Modifications**:
- Header détaillé avec hypothèse et littérature
- Messages de verdict interprétant les résultats
- Guidance claire sur "next steps" selon outcome

---

## 📚 SUPPORT LITTÉRAIRE

### Papers clés trouvés:

1. **Mojgani et al. (2022)** - "Lagrangian PINNs: A causality-conforming solution"
   - **Finding**: Variables Lagrangiennes peuvent violer causalité dans solveurs Eulériens
   - **Relevance**: ⭐⭐⭐⭐⭐ HAUTE - Exactement notre situation!

2. **Wang et al. (2022)** - "Respecting causality is all you need"
   - **Finding**: Structure causale spatio-temporelle critique pour stabilité
   - **Relevance**: ⭐⭐⭐⭐ HAUTE - Confirme importance de causalité

3. **Bremer et al. (2021, ACM)** - "Performance Analysis of Speculative Parallel ALTS"
   - **Finding**: Rollback possible pour conservation laws (1.5% rollback rate)
   - **Relevance**: ⭐⭐⭐ MOYENNE - Pour Option B si Experiment A échoue

### Gap de recherche identifié:

🔍 **AUCUNE littérature trouvée sur**:
- Instabilité des jonctions dans modèles de trafic
- Coupling Lagrangien/Eulérien dans ARZ
- Variable θ_k et causalité

**Implication**: Potentiel de **recherche originale** et **publication académique**!

---

## 🧪 CONFIGURATION DU TEST

### Test identique à Kernel 10 (pour comparaison):

```python
v_m = 10.0 m/s          # BC inflow (haute vitesse)
dt = 0.0001 s           # 10x plus petit que standard
duration = 15 s         # 150,000 timesteps
device = GPU            # Tesla P100 (Kaggle)
```

### Résultat précédent (Kernel 10 - AVEC coupling):

```
❌ FAILURE: v_max=172.11 m/s at t=1.0s (only 6.7% of 15s)
→ Instability persisted even with GPU + dt=0.0001s
```

---

## 📊 RÉSULTATS ATTENDUS

### Cas 1: θ_k EST la cause racine ✅

```
✅ SUCCESS: v_max < 20 m/s for full 15s
✅ Congestion develops: rho > 0.08

INTERPRÉTATION:
→ Lagrangian θ_k coupling confirmed as root cause
→ Violates causality in Eulerian framework
→ Literature hypothesis validated (Mojgani 2022)

NEXT STEPS:
1. Analyze mathematical causality violation
2. Design causality-preserving junction model
3. Publication: Novel finding in traffic flow!
```

### Cas 2: θ_k N'EST PAS la cause racine ❌

```
❌ FAILURE: v_max explodes again (like Kernel 10)

INTERPRÉTATION:
→ θ coupling not the root cause
→ Instability elsewhere:
  • BC inflow formulation?
  • Numerical scheme issues?
  • ODE solver stability?

NEXT STEPS:
1. Implement rollback/checkpoint (Bremer 2021)
2. Analyze BC velocity prescription
3. Test entropy-fix or WENO5 schemes
```

---

## 🚀 EXÉCUTION

### Comment lancer sur Kaggle:

1. **Créer nouveau notebook Kaggle**
2. **Titre**: `ARZ Validation - GPU Stability Test - Experiment A (No θ Coupling)`
3. **Accélérateur**: GPU P100
4. **Code**:

```python
# Clone repository
!git clone https://github.com/elonmj/Code-traffic-flow.git
%cd Code-traffic-flow

# Checkout experiment branch
!git checkout experiment/no-behavioral-coupling

# Install dependencies
!pip install numba matplotlib

# Run test
%cd validation_ch7/scripts
!python test_gpu_stability.py
```

### Fichiers de sortie attendus:

```
/kaggle/working/results/
├── gpu_stability_evolution.png      # Graphiques v_max, rho_max
├── gpu_stability_metrics.json       # Métriques numériques
└── session_summary.json             # Résumé verdict
```

---

## 📈 MÉTRIQUES À SURVEILLER

### Critères de succès:

| Métrique | Seuil | Kernel 10 (AVEC θ) | Kernel 11 (SANS θ) |
|----------|-------|-------------------|-------------------|
| **v_max final** | < 20 m/s | ❌ 172.11 m/s | ⏳ TBD |
| **rho_max final** | > 0.08 | ✅ 0.1147 | ⏳ TBD |
| **Temps atteint** | 15 s | ❌ 1.0 s (6.7%) | ⏳ TBD |
| **Instabilité** | Aucune | ❌ Oui (t=1.0s) | ⏳ TBD |

### Comparaison directe:

```python
# Kernel 10 (AVEC θ coupling):
t = 1.0s: v_max = 172.11 m/s → EXPLOSION
Conclusion: GPU + dt=0.0001 NE RÉSOUT PAS

# Kernel 11 (SANS θ coupling):
t = ???: v_max = ??? m/s → ???
Conclusion: ???
```

---

## 🔍 ANALYSE POST-EXPÉRIENCE

### Si SUCCESS (θ = cause racine):

**Questions à répondre**:
1. Pourquoi θ_k crée violation de causalité?
2. Le feedback loop θ → segment → θ existe-t-il?
3. Peut-on reformuler θ en préservant causalité?
4. Existe-t-il alternative au coupling phénoménologique?

**Analyse mathématique requise**:
```
Tracer graphe de dépendances causales:
θ_k(t) → flux_out(k,t) → segment_i(t+dt) → θ_j(t+dt) → ...

Vérifier: ∃ cycle θ_k → ... → θ_k ?
Si oui → violation CFL effective!
```

### Si FAILURE (autre cause):

**Pistes alternatives**:
1. **BC inflow formulation**:
   - Velocity prescription trop agressive?
   - Besoin de rampe d'entrée progressive?
   
2. **Numerical scheme**:
   - WENO5 plus stable que LF?
   - Entropy fix nécessaire?
   
3. **ODE solver**:
   - Euler explicit insuffisant?
   - RK2/RK3 requis?

4. **Rollback strategy** (Bremer 2021):
   - Checkpointing every 100 steps
   - Rollback + halve dt on instability
   - Proven for conservation laws!

---

## 📝 NOTES COMPLÉMENTAIRES

### Différence avec Kernel 10:

| Aspect | Kernel 10 | Kernel 11 (Exp A) |
|--------|-----------|------------------|
| **Coupling θ** | ✅ Activé | ❌ Désactivé |
| **Strang splitting** | ✅ GPU | ✅ GPU |
| **CFL checking** | ✅ Oui | ✅ Oui |
| **Traffic lights** | ✅ Oui | ✅ Oui |
| **BC inflow** | ✅ v_m=10.0 | ✅ v_m=10.0 |
| **Timestep** | ✅ dt=0.0001 | ✅ dt=0.0001 |

**Unique différence**: 1 ligne commentée dans NetworkGrid.step()!

### Timing estimation:

```
150,000 timesteps @ ~0.003s/timestep (from Kernel 10)
= 450s ≈ 7.5 minutes

+ Overhead (plotting, saving) ≈ 30s
Total: ~8 minutes expected
```

---

## ✅ CHECKLIST LANCEMENT

- [x] Branch créée: `experiment/no-behavioral-coupling`
- [x] Code modifié: `network_grid.py` ligne 590
- [x] Documentation ajoutée: Header + verdict
- [x] Commit & push effectué
- [x] Documentation expérience créée
- [ ] Kaggle notebook créé
- [ ] Kernel lancé
- [ ] Résultats téléchargés
- [ ] Verdict analysé
- [ ] Next steps décidés

---

## 🎯 DÉCISION POST-RÉSULTAT

### Si SUCCESS → Publication Path

1. **Court terme** (1 semaine):
   - Analyse mathématique causality violation
   - Documentation complete du mécanisme
   
2. **Moyen terme** (1 mois):
   - Design junction model causalité-preserving
   - Validation expérimentale
   
3. **Long terme** (3 mois):
   - Rédaction paper académique
   - Soumission conférence/journal

### Si FAILURE → Rollback Implementation

1. **Immédiat** (3 jours):
   - Implement CheckpointManager class
   - Add rollback logic
   
2. **Court terme** (1 semaine):
   - Test rollback avec θ coupling activé
   - Tune checkpoint frequency
   
3. **Moyen terme** (2 semaines):
   - Analyse BC inflow alternative
   - Test WENO5 / entropy-fix

---

**Prêt pour lancement sur Kaggle!** 🚀

Attendre résultats Kernel 11 pour décision finale.
