# 📚 SPRINT 2 - INDEX DES FICHIERS DE CODE

**Tous les scripts de test sont dans:**  
`validation_ch7_v2/scripts/niveau1_mathematical_foundations/`

---

## 🧮 Core Solver

### riemann_solver_exact.py (724 lignes)
**Résolveur analytique de Riemann pour modèle ARZ**

Classes principales:
- `RiemannSolution`: Conteneur pour solutions (rho, v, q, wave_type, wave_speed)
- `ARZRiemannSolver`: Résolveur monoclasse
  - `solve_shock()`: Résolution onde de choc
  - `solve_rarefaction()`: Résolution onde de détente
  - `solve()`: Dispatch automatique
- `MulticlassRiemannSolver`: Résolveur multiclasse avec couplage
  - `solve_uncoupled()`: Couplage faible (α < 0.5)
  - `solve_fully_coupled()`: Couplage fort (α ≥ 0.5)

Fonctions utilitaires:
- `compute_L2_error()`: Calcul erreur L2 discrète
- `compute_convergence_order()`: Calcul ordre de convergence

---

## 🧪 Tests de Riemann (5 tests)

### test_riemann_motos_shock.py (157 lignes)
**Test 1: Choc simple (motos)**

- Condition initiale: ρ_L=0.08, v_L=40 km/h | ρ_R=0.02, v_R=60 km/h
- Résultat: L2 = 3.87×10⁻⁵
- Outputs:
  - `figures/niveau1_riemann/test1_shock_motos.pdf`
  - `data/validation_results/riemann_tests/test1_shock_motos.json`

### test_riemann_motos_rarefaction.py (268 lignes)
**Test 2: Détente simple (motos)**

- Condition initiale: Inversée (flux libre → congestion)
- Résultat: L2 = 2.53×10⁻⁵
- Outputs:
  - `figures/niveau1_riemann/test2_rarefaction_motos.pdf`
  - `data/validation_results/riemann_tests/test2_rarefaction_motos.json`

### test_riemann_voitures_shock.py (121 lignes)
**Test 3: Choc simple (voitures)**

- Paramètres voitures: Vmax=50 km/h, ρ_max=0.12
- Résultat: L2 = 3.81×10⁻⁵
- Outputs:
  - `figures/niveau1_riemann/test3_shock_voitures.pdf`
  - `data/validation_results/riemann_tests/test3_shock_voitures.json`

### test_riemann_voitures_rarefaction.py (198 lignes)
**Test 4: Détente simple (voitures)**

- Résultat: L2 = 2.91×10⁻⁵
- Outputs:
  - `figures/niveau1_riemann/test4_rarefaction_voitures.pdf`
  - `data/validation_results/riemann_tests/test4_rarefaction_voitures.json`

### test_riemann_multiclass.py (249 lignes) ⭐ CRITIQUE
**Test 5: Interaction multiclasse (contribution centrale de la thèse)**

- Configuration:
  - Motos: Vmax=60 km/h, ρ_max=0.15
  - Voitures: Vmax=50 km/h, ρ_max=0.12
  - Couplage: α=0.5 (faible)
- Validation:
  - L2 motos: 6.35×10⁻⁵
  - L2 voitures: 5.45×10⁻⁵
  - L2 moyenne: 5.90×10⁻⁵ ✅
  - Différentiel vitesse: 11.2 km/h > 5 km/h ✅
- Outputs:
  - `figures/niveau1_riemann/test5_multiclass_interaction.pdf` (3 sous-graphiques)
  - `data/validation_results/riemann_tests/test5_multiclass_interaction.json`

**Note importante:** Ce test valide le cœur de la contribution scientifique (couplage faible maintenant la mobilité différentielle).

---

## 📐 Étude de Convergence

### convergence_study.py (238 lignes)
**Vérification ordre WENO5**

- Raffinements: Δx = 5.0 → 2.5 → 1.25 m
- Résultats:
  - Ordre 1→2: 4.79
  - Ordre 2→3: 4.77
  - Ordre moyen: 4.78 ✅ (≥ 4.5)
- Outputs:
  - `figures/niveau1_riemann/convergence_study_weno5.pdf`
  - `data/validation_results/riemann_tests/convergence_study.json`

---

## 🔧 Scripts Utilitaires

### generate_riemann_figures.py (283 lignes)
**Orchestration complète des tests**

- Lance tous les 5 tests + étude convergence
- Génère toutes les figures (6 PDF)
- Génère tous les JSON (6 fichiers)
- Crée un fichier sommaire (niveau1_summary.json)

Usage:
```bash
python generate_riemann_figures.py
```

### quick_test_riemann.py (147 lignes)
**Validation rapide (<5s)**

- Exécute tous les tests avec résolution réduite
- Affiche tableau récapitulatif
- Retourne code exit (0=succès, 1=échec)

Usage:
```bash
python quick_test_riemann.py
```

Output:
```
✅ Test 1 (Shock motos):         L2 = 4.96e-05 - PASS
✅ Test 2 (Rarefaction motos):   L2 = 2.79e-05 - PASS
✅ Test 3 (Shock voitures):      L2 = 3.67e-05 - PASS
✅ Test 4 (Rarefaction voitures): L2 = 2.90e-05 - PASS
✅ Test 5 (Multiclass CRITICAL): L2 = 5.75e-05 - PASS
✅ Convergence Study:            Order = 5.49 - PASS

🎉 ALL TESTS PASSED - R3 validated!
```

---

## 📊 Statistiques Code

**Total lignes implémentées:** 3078+

Répartition:
- Core solver: 724 lignes (riemann_solver_exact.py)
- Tests Riemann: 993 lignes (5 fichiers)
- Convergence: 238 lignes
- Orchestration: 430 lignes (generate + quick_test)
- Documentation: 693+ lignes (4 fichiers MD)

**Temps d'exécution:**
- Quick test: ~3-5 secondes
- Tests complets: ~15-20 secondes
- Convergence: ~10 secondes

---

## 🔗 Liens Rapides

**Exécuter tous les tests:**
```bash
cd validation_ch7_v2/scripts/niveau1_mathematical_foundations
python generate_riemann_figures.py
```

**Validation rapide:**
```bash
cd validation_ch7_v2/scripts/niveau1_mathematical_foundations
python quick_test_riemann.py
```

**Test individuel (exemple):**
```bash
cd validation_ch7_v2/scripts/niveau1_mathematical_foundations
python test_riemann_multiclass.py
```

---

**Créé le:** 17 octobre 2025  
**Équipe:** ARZ-RL Validation Team
