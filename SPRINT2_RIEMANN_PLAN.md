# Sprint 2 - Niveau 1: Fondations Mathématiques (Tests de Riemann)

## 🎯 **Objectif Sprint 2**
Valider la Revendication **R3**: *La stratégie numérique FVM + WENO garantit une résolution stable et précise.*

---

## 📋 **Vue d'ensemble**

Les **problèmes de Riemann** sont des tests standards en simulation d'écoulement hyperbolique. Ils consistent à résoudre l'évolution d'une discontinuité initiale (choc, détente) et à comparer la solution numérique à la solution analytique exacte.

**Pourquoi c'est crucial** :
- ✅ Prouve que le code résout **correctement** les équations ARZ
- ✅ Valide la **précision** du schéma WENO5 (ordre ~5 attendu)
- ✅ Démontre la capacité à capturer des **ondes de choc** sans oscillations
- ✅ Teste le **couplage multiclasse** (motos/voitures)

---

## 🧪 **Les 5 Cas de Test Proposés**

### **Test 1: Choc Simple (Motos seules)**
**Configuration initiale** :
```
Gauche (x < 500m):  ρ_motos = 0.08 veh/m,  v_motos = 40 km/h
Droite (x ≥ 500m):  ρ_motos = 0.02 veh/m,  v_motos = 60 km/h
```

**Phénomène attendu** :
- Formation d'une **onde de choc** se propageant vers la gauche
- Transition abrupte densité/vitesse à l'interface

**Solution analytique** :
```
Vitesse du choc: s = (q_R - q_L) / (ρ_R - ρ_L)
Position: x_choc(t) = 500 + s*t
```

**Ce qu'on valide** :
- Capture des discontinuités sans oscillations numériques
- Précision de la vitesse de propagation du choc

---

### **Test 2: Détente Simple (Motos seules)**
**Configuration initiale** :
```
Gauche (x < 500m):  ρ_motos = 0.02 veh/m,  v_motos = 60 km/h
Droite (x ≥ 500m):  ρ_motos = 0.08 veh/m,  v_motos = 40 km/h
```

**Phénomène attendu** :
- Formation d'une **onde de détente** (raréfaction)
- Transition smooth (non abrupte) avec éventail de caractéristiques

**Solution analytique** :
```
Onde de détente auto-similaire: ρ(x,t) = fonction de x/t
```

**Ce qu'on valide** :
- Résolution des zones de détente (plus difficile que chocs)
- Ordre de convergence élevé dans les régions lisses

---

### **Test 3: Choc Simple (Voitures seules)**
**Configuration initiale** :
```
Gauche (x < 500m):  ρ_voitures = 0.06 veh/m,  v_voitures = 35 km/h
Droite (x ≥ 500m):  ρ_voitures = 0.01 veh/m,  v_voitures = 50 km/h
```

**Phénomène attendu** :
- Similaire au Test 1, mais avec paramètres voitures (Vmax, ρ_max différents)

**Ce qu'on valide** :
- Consistance du solveur pour différentes classes de véhicules
- Respect des paramètres calibrés (Vmax_voitures < Vmax_motos)

---

### **Test 4: Détente Simple (Voitures seules)**
**Configuration initiale** :
```
Gauche (x < 500m):  ρ_voitures = 0.01 veh/m,  v_voitures = 50 km/h
Droite (x ≥ 500m):  ρ_voitures = 0.06 veh/m,  v_voitures = 35 km/h
```

**Phénomène attendu** :
- Détente smooth pour voitures

**Ce qu'on valide** :
- Robustesse du schéma pour classe "lente"

---

### **Test 5: Interaction Multi-Classes (LE PLUS IMPORTANT!)**
**Configuration initiale** :
```
Gauche (x < 500m):  
  ρ_motos = 0.05 veh/m,  v_motos = 50 km/h
  ρ_voitures = 0.03 veh/m,  v_voitures = 40 km/h

Droite (x ≥ 500m):  
  ρ_motos = 0.02 veh/m,  v_motos = 60 km/h
  ρ_voitures = 0.01 veh/m,  v_voitures = 50 km/h
```

**Phénomène attendu** :
- **Couplage motos-voitures** via pression d'anticipation α
- Motos plus rapides créent un "appel d'air" pour les voitures
- Solution complexe avec deux ondes (une par classe)

**Solution semi-analytique** :
```
Système couplé 4×4 (ρ_m, q_m, ρ_v, q_v)
Résolution par solveur de Riemann multicomposant
```

**Ce qu'on valide** :
- **COEUR DE LA THESE**: Le modèle ARZ étendu capture le couplage multiclasse
- Ordre de convergence maintenu malgré couplage

---

## 📊 **Métriques de Validation**

### **1. Erreur L2 (Norme Euclidienne)**
```
L2 = sqrt( sum((ρ_sim - ρ_exact)^2 * Δx) / L_domain )
```

**Critère d'acceptation** : L2 < 1e-3 pour chaque test

### **2. Ordre de Convergence**
Refinement study avec 3 maillages:
- Coarse: Δx = 5m (N = 200 cells)
- Medium: Δx = 2.5m (N = 400 cells)
- Fine: Δx = 1.25m (N = 800 cells)

```
Ordre = log(L2_coarse / L2_fine) / log(Δx_coarse / Δx_fine)
```

**Critère d'acceptation** : Ordre ≥ 4.5 (proche de l'ordre théorique 5 de WENO5)

### **3. Validation Visuelle**
Graphiques superposant:
- Solution simulée (ligne continue rouge)
- Solution analytique (points noirs)
- Zoom sur la discontinuité

---

## 🛠️ **Architecture des Fichiers Sprint 2**

```
validation_ch7_v2/
├── scripts/
│   ├── niveau1_mathematical_foundations/
│   │   ├── __init__.py
│   │   ├── riemann_solver_exact.py          # Solutions analytiques
│   │   ├── test_riemann_motos_shock.py      # Test 1
│   │   ├── test_riemann_motos_rarefaction.py # Test 2
│   │   ├── test_riemann_voitures_shock.py   # Test 3
│   │   ├── test_riemann_voitures_rarefaction.py # Test 4
│   │   ├── test_riemann_multiclass.py       # Test 5 (CRITIQUE)
│   │   ├── convergence_study.py             # Raffinement de maillage
│   │   └── generate_riemann_figures.py      # Génération figures LaTeX
│   │
│   └── config/
│       └── niveau1_mathematical_config.yaml  # Paramètres tests
│
├── data/
│   └── validation_results/
│       └── riemann_tests/
│           ├── test1_shock_motos.json
│           ├── test2_rarefaction_motos.json
│           ├── test3_shock_voitures.json
│           ├── test4_rarefaction_voitures.json
│           └── test5_multiclass.json
│
└── figures/
    └── niveau1_riemann/
        ├── riemann_choc_simple.pdf
        ├── riemann_interaction_multiclasse.pdf
        └── convergence_order_plot.pdf
```

---

## 📐 **Détails Techniques**

### **Solution Analytique - Cas Simple (1 classe)**

Pour le système ARZ monocomposant:
```
∂ρ/∂t + ∂q/∂x = 0
∂q/∂t + ∂(q²/ρ + P)/∂x = S (terme source relaxation)
```

**Sans terme source** (S=0), le problème de Riemann a une solution exacte:

1. **Choc (ρ_L > ρ_R)** :
   ```
   Vitesse choc: s = (q_R - q_L) / (ρ_R - ρ_L)
   
   Si x < x_0 + s*t:  ρ = ρ_L,  v = v_L
   Si x ≥ x_0 + s*t:  ρ = ρ_R,  v = v_R
   ```

2. **Détente (ρ_L < ρ_R)** :
   ```
   Onde de détente self-similar:
   ρ(x,t) = ρ(ξ) où ξ = (x - x_0) / t
   
   Résolution par caractéristiques
   ```

### **Solution Semi-Analytique - Cas Multiclasse**

Le système 4×4 couplé nécessite un solveur de Riemann multicomposant.

**Approche** :
1. Découplage par diagonalisation locale
2. Résolution de 4 problèmes scalaires
3. Recouplage via terme de pression α

**Implémentation** :
- Utiliser `scipy.integrate.solve_ivp` pour les caractéristiques
- Vérifier conservation de la masse totale

---

## 🎯 **Critères de Succès Sprint 2**

| Critère | Objectif | Seuil Acceptation |
|---------|----------|-------------------|
| Erreur L2 moyenne (5 tests) | < 1.5e-4 | < 5e-4 |
| Ordre de convergence moyen | ~4.75 | ≥ 4.5 |
| Test multiclasse L2 | < 2.5e-4 | < 1e-3 |
| Figures publication-ready | Oui | Oui |
| Documentation LaTeX | Complète | Complète |

---

## 📝 **Intégration LaTeX**

Sections à remplir dans `section7_validation_nouvelle_version.tex`:

1. **Tableau~\ref{tab:riemann_validation_results_revised}** :
   - Remplacer `[PLACEHOLDER]` par résultats réels
   - Ajouter colonne "Temps calcul (s)"

2. **Figures** :
   - Figure~\ref{fig:riemann_choc_simple_revised}
   - Figure~\ref{fig:riemann_interaction_multiclasse_revised}
   - Figure de convergence (nouvelle)

3. **Texte explicatif** :
   - Ajouter explication physique de chaque test
   - Justifier choix des conditions initiales
   - Discuter limitations (terme source négligé)

---

## ⏱️ **Estimation Temps**

| Tâche | Durée estimée |
|-------|---------------|
| Solutions analytiques (Tests 1-4) | 2h |
| Solution semi-analytique (Test 5) | 3h |
| Tests + convergence study | 2h |
| Génération figures | 1h |
| Documentation LaTeX | 1h |
| **TOTAL** | **9h** |

---

## 🚀 **Prochaines Étapes Immédiates**

1. ✅ **FAIT**: Corriger structure données (70 segments, 61 timestamps)
2. ✅ **FAIT**: Mettre à jour documentation LaTeX
3. ⏳ **SUIVANT**: Créer `riemann_solver_exact.py` avec solutions analytiques
4. ⏳ Implémenter les 5 tests
5. ⏳ Générer figures publication-ready

---

**Voulez-vous que je commence par créer `riemann_solver_exact.py` avec les solutions analytiques pour les Tests 1-4 ?**
