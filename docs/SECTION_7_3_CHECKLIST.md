# SECTION 7.3 - CHECKLIST DÉTAILLÉE
**Tests Analytiques - Validation R1 & R3**

---

## TEST 1: Riemann Problems  PRÊT
**Fichier:** test_riemann_problems() dans test_section_7_3_analytical.py
**État:**  Utilise vrai code SimulationRunner

### Workflow
- [ ] 1.1. Tester localement (1 cas Riemann)
- [ ] 1.2. Vérifier NPZ généré localement
- [ ] 1.3. Commit + Push GitHub
- [ ] 1.4. Lancer sur Kaggle GPU (5 cas)
- [ ] 1.5. Télécharger NPZ (5 fichiers)
- [ ] 1.6. Générer figures localement:
  - [ ] riemann_solutions_comparison.png (5 subplots)
  - [ ] shock_wave_speeds.png
- [ ] 1.7. Calculer métriques: vitesses chocs, conservation masse
- [ ] 1.8. Générer section_7_3_riemann.tex
- [ ] 1.9. Déplacer vers validated/section_7_3_analytical/

### Critères Validation
- Vitesse choc: |v_obs - v_theo| < 5%
- Conservation masse: < 1e-6
- Ordre convergence: > 4.0

### Commandes
`bash
# Local test
cd validation_ch7/scripts
python -c \"from test_section_7_3_analytical import AnalyticalValidationTests; t = AnalyticalValidationTests(); t.test_riemann_problems()\"

# Kaggle test
python ../../validation_cli.py --section section_7_3_analytical --commit-message \"Section 7.3: Riemann tests with real SimulationRunner\"
`

---

## TEST 2: Convergence WENO5  À VÉRIFIER
**Fichier:** test_convergence_analysis() dans test_section_7_3_analytical.py
**État:**  À vérifier si utilise vrai SimulationRunner

### Workflow
- [ ] 2.1. Vérifier validation_utils.run_convergence_analysis()
- [ ] 2.2. S'assurer utilise code/analysis/convergence.py
- [ ] 2.3. Tester localement (2 grilles: N=50, N=100)
- [ ] 2.4. Vérifier NPZ générés
- [ ] 2.5. Commit si corrections
- [ ] 2.6. Lancer sur Kaggle GPU (4 grilles: N=50,100,200,400)
- [ ] 2.7. Télécharger NPZ (4 fichiers)
- [ ] 2.8. Générer figures:
  - [ ] convergence_order_plot.png
  - [ ] error_vs_gridsize_loglog.png
- [ ] 2.9. Calculer ordre convergence observé
- [ ] 2.10. Générer section_7_3_convergence.tex
- [ ] 2.11. Déplacer vers validated/

### Critères Validation
- Ordre convergence: 4.0 - 5.0
- Erreur L2 décroît comme h^5

---

## TEST 3: Équilibre Free Flow  À RECRÉER
**Fichier:** test_equilibrium_profiles() - actuellement mock
**État:**  À recréer avec SimulationRunner

### À Implémenter
`python
def test_equilibrium_free_flow_REAL():
    # 1. Créer scenario YAML avec état uniforme
    # 2. Différentes densités: ρ = 0.1, 0.2, 0.3, 0.4
    # 3. Simuler longtemps (t >> 3*τ)
    # 4. Vérifier v_final  V_e(ρ_final)
    # 5. Comparer diagramme fondamental
    # 6. SAVE NPZ
`

### Workflow
- [ ] 3.1. Créer test_equilibrium_free_flow_REAL()
- [ ] 3.2. Créer scenarios YAML (4 densités)
- [ ] 3.3. Tester localement (1 densité)
- [ ] 3.4. Vérifier NPZ
- [ ] 3.5. Commit
- [ ] 3.6. Lancer sur Kaggle (4 densités)
- [ ] 3.7. Télécharger NPZ (4 fichiers)
- [ ] 3.8. Générer figures:
  - [ ] fundamental_diagram_moto_car.png
  - [ ] equilibrium_convergence_time.png
- [ ] 3.9. Calculer métriques: MAE(v, V_e), temps convergence
- [ ] 3.10. Générer section_7_3_equilibrium.tex
- [ ] 3.11. Déplacer vers validated/

### Critères Validation
- MAE(v_sim, V_e) < 0.5 m/s
- MAPE < 5%
- t_convergence  3-5*τ

---

## TEST 4: Relaxation Dynamics  NOUVEAU
**À Créer:** test_relaxation_dynamics()

### À Implémenter
`python
def test_relaxation_dynamics():
    # 1. État initial: w  w_eq (hors équilibre)
    # 2. Pas de BC perturbante
    # 3. Mesurer t_relax pour retour à w_eq
    # 4. Tester plusieurs τ: 1.5s, 3.0s
    # 5. SAVE NPZ avec évolution temporelle
`

### Workflow
- [ ] 4.1. Créer test_relaxation_dynamics()
- [ ] 4.2. Créer scenarios YAML (2 valeurs τ)
- [ ] 4.3. Tester localement
- [ ] 4.4. Commit
- [ ] 4.5. Lancer sur Kaggle
- [ ] 4.6. Télécharger NPZ (2 fichiers)
- [ ] 4.7. Générer figures:
  - [ ] w_trajectory_vs_time.png
  - [ ] relaxation_time_vs_tau.png
- [ ] 4.8. Calculer t_relax / τ ratio
- [ ] 4.9. Générer section_7_3_relaxation.tex
- [ ] 4.10. Déplacer vers validated/

### Critères Validation
- t_relax / τ  3-5
- Pas d'overshoot
- Convergence exponentielle visible

---

## TEST 5: Interactions Multi-Classes  NOUVEAU
**À Créer:** test_multiclass_interactions()

### À Implémenter
`python
def test_multiclass_interactions():
    # 1. État initial: 2 classes, perturbation classe 1
    # 2. Observer propagation vers classe 2
    # 3. Tester α = 0.3, 0.5, 0.7 (gap-filling)
    # 4. Mesurer couplage interweaving
    # 5. SAVE NPZ
`

### Workflow
- [ ] 5.1. Créer test_multiclass_interactions()
- [ ] 5.2. Créer scenarios YAML (3 valeurs α)
- [ ] 5.3. Tester localement
- [ ] 5.4. Commit
- [ ] 5.5. Lancer sur Kaggle
- [ ] 5.6. Télécharger NPZ (3 fichiers)
- [ ] 5.7. Générer figures:
  - [ ] perturbation_propagation_multiclass.png
  - [ ] gap_filling_effect_alpha.png
- [ ] 5.8. Calculer vitesses propagation différentielles
- [ ] 5.9. Générer section_7_3_multiclass.tex
- [ ] 5.10. Déplacer vers validated/

### Critères Validation
- v_moto > v_car (vérifié)
- Effet α visible
- Couplage bidirectionnel observé

---

## TEST 6: Infrastructure R(x)  NOUVEAU
**À Créer:** test_road_quality_influence()

### À Implémenter
`python
def test_road_quality_influence():
    # 1. Grille avec R(x) variable: R=1  R=0.6  R=1
    # 2. Observer V_e local réduit
    # 3. Mesurer distance récupération
    # 4. SAVE NPZ
`

### Workflow
- [ ] 6.1. Créer test_road_quality_influence()
- [ ] 6.2. Vérifier Grid1D supporte R(x) variable
- [ ] 6.3. Créer scenario YAML avec R(x) profile
- [ ] 6.4. Tester localement
- [ ] 6.5. Commit
- [ ] 6.6. Lancer sur Kaggle
- [ ] 6.7. Télécharger NPZ (1 fichier)
- [ ] 6.8. Générer figures:
  - [ ] velocity_vs_road_quality.png
  - [ ] recovery_distance_plot.png
- [ ] 6.9. Calculer corrélation V_e vs R
- [ ] 6.10. Générer section_7_3_infrastructure.tex
- [ ] 6.11. Déplacer vers validated/

### Critères Validation
- V_e réduit proportionnellement à R
- Récupération progressive visible
- Cohérence avec observations terrain

---

## SYNTHÈSE SECTION 7.3

### NPZ Attendus (Total: ~15 fichiers)
- riemann_*.npz (5)
- convergence_*.npz (4)
- equilibrium_*.npz (4)
- relaxation_*.npz (2)
- multiclass_*.npz (3)
- road_quality_*.npz (1)

### Figures Attendues (Total: ~12 figures)
- Riemann: 2 figures
- Convergence: 2 figures
- Équilibre: 2 figures
- Relaxation: 2 figures
- Multi-classes: 2 figures
- Infrastructure: 2 figures

### LaTeX Généré
- section_7_3_riemann.tex
- section_7_3_convergence.tex
- section_7_3_equilibrium.tex
- section_7_3_relaxation.tex
- section_7_3_multiclass.tex
- section_7_3_infrastructure.tex
- section_7_3_synthesis.tex (résumé complet)

### Temps Estimé
- Tests 1-2 (prêts): 4h
- Test 3 (recréer): 3h
- Tests 4-6 (créer): 5h
- **Total: 12h**

---

## ORDRE D'EXÉCUTION RECOMMANDÉ

1.  TEST 1 (Riemann) - PRÊT  START HERE
2.  TEST 2 (Convergence) - Vérifier  Facile
3.  TEST 3 (Équilibre) - Recréer  Important
4.  TEST 4 (Relaxation) - Créer  Nouveau
5.  TEST 5 (Multi-classes) - Créer  Nouveau
6.  TEST 6 (Infrastructure) - Créer  Nouveau

---

## NEXT ACTION: Commencer TEST 1 (Riemann)

`bash
python validation_cli.py --section section_7_3_analytical --commit-message \"Section 7.3.1: Riemann problems validation - R1 physical behavior\"
`
