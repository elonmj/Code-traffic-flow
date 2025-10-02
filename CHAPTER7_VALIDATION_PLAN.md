# CHAPITRE 7 - PLAN DE VALIDATION COMPLET
**Date:** 2 octobre 2025
**Objectif:** Validation systématique avec vrai code + Kaggle GPU + LaTeX intégré

---

## ANALYSE ÉTAT ACTUEL PAR SECTION

###  SECTION 7.3 - Validation ARZ Segment (R1, R3)
**Fichier:** test_section_7_3_analytical.py
**État:**  PARTIELLEMENT MOCK

#### Utilise Vrai Code:
-  code.io.data_manager.save_simulation_data (NPZ saving)
-  alidation_utils.run_real_simulation  SimulationRunner
-  alidation_utils.create_riemann_scenario_config
-  alidation_utils.run_convergence_analysis

#### Utilise Mock:
-  nalytical_riemann_solution() - solution analytique mockée
-  nalytical_equilibrium_profile() - profil équilibre mocké
-  	est_equilibrium_profiles() - entièrement mock

#### Actions Requises:
1. **GARDER** test_riemann_problems() tel quel (utilise vrai SimulationRunner)
2. **VÉRIFIER** run_convergence_analysis() utilise vraiment SimulationRunner
3. **RECRÉER** test_equilibrium_profiles() avec SimulationRunner
4. **AJOUTER** test_relaxation_dynamics() (nouveau)
5. **AJOUTER** test_multiclass_interactions() (nouveau)
6. **AJOUTER** test_road_quality_influence() (nouveau)

---

###  SECTION 7.4 - Calibration (R2)
**Fichier:** test_section_7_4_calibration.py
**État:**  UTILISE VRAI CODE

#### Utilise Vrai Code:
-  code.analysis.metrics
-  alidation_utils.RealARZValidationTest
-  Charge données Victoria Island (Code_RL/data/)
-  Génère données synthétiques si fichiers manquants

#### Actions Requises:
1. **VÉRIFIER** que calibration utilise code/calibration/core/
2. **TESTER** localement avec données synthétiques
3. **LANCER** sur Kaggle GPU
4. **VALIDER** métriques GEH, MAPE

---

###  SECTION 7.5 - Jumeau Numérique (R3, R4, R6)
**Fichier:** test_section_7_5_digital_twin.py
**État:**  UTILISE VRAI CODE

#### Utilise Vrai Code:
-  code.analysis.metrics
-  code.simulation.runner.SimulationRunner
-  code.core.parameters.ModelParameters

#### Actions Requises:
1. **VÉRIFIER** intégration avec code/calibration/data/digital_twin_calibrator.py
2. **TESTER** localement
3. **LANCER** sur Kaggle GPU
4. **VALIDER** métriques spatiotemporelles

---

###  SECTION 7.6 - RL Performance (R5)
**Fichier:** test_section_7_6_rl_performance.py
**État:**  UTILISE CODE ARZ, PAS Code_RL

#### Utilise Vrai Code ARZ:
-  code.simulation.runner.SimulationRunner
-  code.core.parameters.ModelParameters
-  code.analysis.metrics

#### Manque Intégration Code_RL:
-  N'importe pas Code_RL/src/rl/train_dqn.py
-  N'importe pas Code_RL/src/env/traffic_signal_env.py
-  Utilise baseline controller mock

#### Actions Requises:
1. **AJOUTER** imports Code_RL
2. **INTÉGRER** TrafficSignalEnv
3. **CHARGER** agent DQN entraîné (ou entraîner sur Kaggle)
4. **COMPARER** RL vs baseline avec vraies simulations
5. **LANCER** sur Kaggle GPU

---

###  SECTION 7.7 - Robustesse (R4, R6)
**Fichier:** test_section_7_7_robustness.py
**État:**  UTILISE VRAI CODE

#### Utilise Vrai Code:
-  code.simulation.runner.SimulationRunner
-  code.core.parameters.ModelParameters
-  code.analysis.metrics

#### Actions Requises:
1. **VÉRIFIER** tests GPU/CPU cohérence
2. **TESTER** localement (CPU)
3. **LANCER** sur Kaggle GPU
4. **VALIDER** robustesse conditions extrêmes

---

## PRIORITÉS D'EXÉCUTION

### PHASE 1: Section 7.3 - Tests Analytiques (URGENT)
**Temps:** 12 heures

1. **test_riemann_problems()** -  DÉJÀ BON
   - Action: TESTER localement  Kaggle
   
2. **test_convergence_analysis()** -  À VÉRIFIER
   - Action: Vérifier utilise SimulationRunner  Kaggle
   
3. **test_equilibrium_profiles()** -  À RECRÉER
   - Action: Créer avec SimulationRunner + scenarios YAML

4. **test_relaxation_dynamics()** -  À CRÉER (NOUVEAU)
   
5. **test_multiclass_interactions()** -  À CRÉER (NOUVEAU)
   
6. **test_road_quality_influence()** -  À CRÉER (NOUVEAU)

### PHASE 2: Section 7.6 - Intégration RL (IMPORTANT)
**Temps:** 8 heures

1. Ajouter imports Code_RL
2. Intégrer TrafficSignalEnv
3. Charger/entraîner agent DQN
4. Tests comparatifs RL vs baseline
5. Kaggle GPU

### PHASE 3: Sections 7.4, 7.5, 7.7 - Vérification (FACILE)
**Temps:** 6 heures

1. Vérifier chaque test utilise bien le vrai code
2. Tester localement
3. Lancer sur Kaggle GPU
4. Valider métriques

---

## STRUCTURE FINALE DES RÉSULTATS

`
validation_ch7/
 results/
    validated/                     Résultats VALIDÉS prêts pour thèse
       section_7_3_analytical/
          npz/
             riemann_shock_moto.npz
             riemann_rarefaction_car.npz
             convergence_N100.npz
             ...
          figures/
             riemann_solutions.png
             convergence_order.png
             ...
          metrics/
             riemann_metrics.csv
             convergence_orders.json
          latex/
              section_7_3_content.tex
       section_7_4_calibration/
       section_7_5_digital_twin/
       section_7_6_rl_performance/
       section_7_7_robustesse/
    work_in_progress/
`

---

## CHECKLIST GLOBALE

### Section 7.3
- [ ] test_riemann: tester local  Kaggle  valider NPZ
- [ ] test_convergence: vérifier code  Kaggle  valider ordre
- [ ] test_equilibrium: créer  tester  Kaggle
- [ ] test_relaxation: créer  tester  Kaggle
- [ ] test_multiclass: créer  tester  Kaggle
- [ ] test_road_quality: créer  tester  Kaggle
- [ ] Générer figures + LaTeX
- [ ] Déplacer vers validated/

### Section 7.4
- [ ] Vérifier utilise code/calibration/
- [ ] Tester local  Kaggle
- [ ] Valider GEH, MAPE
- [ ] Générer figures + LaTeX
- [ ] Déplacer vers validated/

### Section 7.5
- [ ] Vérifier digital twin calibrator
- [ ] Tester local  Kaggle
- [ ] Valider métriques spatiotemporelles
- [ ] Générer figures + LaTeX
- [ ] Déplacer vers validated/

### Section 7.6
- [ ] Ajouter imports Code_RL
- [ ] Intégrer TrafficSignalEnv
- [ ] Charger/entraîner agent DQN
- [ ] Tester local  Kaggle (entraînement)
- [ ] Comparer RL vs baseline
- [ ] Générer courbes apprentissage + LaTeX
- [ ] Déplacer vers validated/

### Section 7.7
- [ ] Vérifier tests GPU/CPU
- [ ] Tester local  Kaggle GPU
- [ ] Valider robustesse
- [ ] Générer figures + LaTeX
- [ ] Déplacer vers validated/

---

## TEMPS TOTAL ESTIMÉ: 26 heures

**Répartition:**
- Phase 1 (7.3): 12h
- Phase 2 (7.6): 8h
- Phase 3 (7.4, 7.5, 7.7): 6h

---

## NEXT ACTION: PHASE 1 - Section 7.3

**Commencer par test_riemann_problems() qui est DÉJÀ BON**
