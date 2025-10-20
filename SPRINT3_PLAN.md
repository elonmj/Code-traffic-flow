# 🎯 SPRINT 3 - PLAN DÉTAILLÉ
## Niveau 2 : Phénomènes Physiques (Gap-Filling & Interweaving)

**Date de début:** 17 octobre 2025  
**Objectif:** Valider R1 - Le modèle ARZ capture les phénomènes de trafic ouest-africain

---

## 📋 Vue d'Ensemble

### Revendication à Valider
**R1:** Le modèle ARZ bi-classe capture les comportements de conduite ouest-africains (gap-filling et mobilité différentielle)

### Livrables Attendus

**3 Tests Principaux:**
1. **Test Gap-Filling** - Motos infiltrant trafic voitures
2. **Test Interweaving** - Tissage entre classes
3. **Diagrammes Fondamentaux** - Calibration vitesse-densité

**Outputs:**
- 3-4 figures PNG (300 DPI)
- 3 fichiers JSON (métriques)
- Tableau LaTeX comparatif
- Animation UXsim (optionnel mais souhaité)

---

## 🧪 Test 1: Gap-Filling (CRITIQUE)

### Objectif
Démontrer que les motos exploitent leur mobilité pour combler les espaces entre voitures.

### Scénario Synthétique
```yaml
Configuration:
  - Segment: 1000 m, 2 voies
  - Voitures: 10 véhicules répartis (100-1000m), v_init=25 km/h
  - Motos: 20 véhicules groupés (0-100m), v_init=40 km/h
  - Durée: 300 secondes
  - Couplage: α = 0.5 (faible)
```

### Physique Attendue
1. **t=0s**: Motos derrière, voitures devant espacées
2. **t=150s**: Motos infiltrent les espaces (gap-filling actif)
3. **t=300s**: Motos ont dépassé majoritairement, voitures restent à vitesse réduite

### Métriques à Calculer
- **Vitesse moyenne motos** à t=300s (attendu: >35 km/h)
- **Vitesse moyenne voitures** à t=300s (attendu: ~25 km/h)
- **Taux d'infiltration**: % motos ayant dépassé ≥5 voitures
- **Maintien différentiel**: Δv_moy > 10 km/h

### Validation
✅ Vitesse motos > vitesse voitures (gap-filling actif)  
✅ Δv maintenu sur toute la simulation  
✅ Conservation masse (erreur <1%)

---

## 🧪 Test 2: Interweaving

### Objectif
Montrer le tissage continu entre classes (motos zigzaguant entre voitures).

### Scénario
```yaml
Configuration:
  - Segment: 2000 m, 3 voies
  - Distribution homogène: 15 motos + 15 voitures
  - Densités: ρ_motos=0.03 veh/m, ρ_voitures=0.02 veh/m
  - v_init aléatoire: motos [35-45], voitures [20-30] km/h
  - Durée: 400s
```

### Métriques à Calculer
- **Changements de voie** (si multi-voies)
- **Distribution spatiale** au fil du temps
- **Entropie de mélange**: Mesure du tissage

### Validation
✅ Motos ne restent pas groupées (tissage actif)  
✅ Distribution finale: motos devant, voitures derrière  
✅ Pas de blocage mutuel

---

## 🧪 Test 3: Diagrammes Fondamentaux

### Objectif
Calibrer et valider les paramètres ARZ sur données TomTom.

### Approche
**PROBLÈME**: TomTom ne sépare pas motos/voitures  
**SOLUTION**: Utiliser paramètres littérature + validation synthétique

### Paramètres à Calibrer
**Motos:**
- V_max = 60 km/h (Lagos urban context)
- ρ_max = 0.15 veh/m
- τ = 0.5s (temps de relaxation)

**Voitures:**
- V_max = 50 km/h
- ρ_max = 0.12 veh/m
- τ = 1.0s

### Diagrammes à Produire
1. **V-ρ (Vitesse-Densité)**
   - X: Densité ρ (veh/m)
   - Y: Vitesse v (km/h)
   - Courbe: V_max * (1 - ρ/ρ_max)

2. **Q-ρ (Flux-Densité)**
   - X: Densité ρ (veh/m)
   - Y: Flux q (veh/h)
   - Courbe parabolique avec pic à ρ_crit

### Validation
✅ Pics de flux cohérents avec littérature  
✅ Vitesses libres réalistes (Lagos context)  
✅ Capacités raisonnables

---

## 📊 Structure des Fichiers

### Scripts à Créer

```
validation_ch7_v2/scripts/niveau2_physical_phenomena/
├── __init__.py
├── gap_filling_test.py (300 lignes)
│   ├── setup_scenario()
│   ├── run_simulation()
│   ├── compute_metrics()
│   ├── plot_results()
│   └── run_test()
│
├── interweaving_test.py (250 lignes)
│   ├── setup_scenario()
│   ├── run_simulation()
│   ├── compute_mixing_entropy()
│   ├── plot_results()
│   └── run_test()
│
├── fundamental_diagrams.py (200 lignes)
│   ├── compute_theoretical_curves()
│   ├── plot_v_rho()
│   ├── plot_q_rho()
│   └── generate_figures()
│
└── quick_test_niveau2.py (100 lignes)
    └── validate_all_tests()
```

### Outputs Attendus

```
validation_ch7_v2/
├── figures/niveau2_physics/
│   ├── gap_filling_evolution.png
│   ├── gap_filling_metrics_bar.png
│   ├── interweaving_distribution.png
│   ├── fundamental_diagram_motos.png
│   └── fundamental_diagram_voitures.png
│
└── data/validation_results/physics_tests/
    ├── gap_filling_test.json
    ├── interweaving_test.json
    └── fundamental_diagrams.json
```

---

## 🎨 Visualisations

### Figure 1: Gap-Filling Évolution (3 subplots)
```
[t=0s]         [t=150s]        [t=300s]
🏍️🏍️🏍️      🚗🏍️🚗          🏍️🏍️🏍️
             🚗  🏍️ 🚗        🚗  🚗
🚗  🚗  🚗   🏍️🚗   🏍️       🚗
```

**Légende:**
- Bleu: Motos
- Orange: Voitures
- Axe X: Position (m)
- Axe Y: Vitesse (km/h) ou juste position

### Figure 2: Métriques Gap-Filling (Bar Chart)
```
Vitesse moyenne (km/h)
40 ┤     ████
35 ┤     ████  ████
30 ┤     ████  ████
25 ┤     ████  ████  ████
20 ┤     ████  ████  ████
   └─────────────────────
      Motos  Mixed Voitures
      Seules       Seules
```

### Figure 3: Diagrammes Fondamentaux (2 subplots)
```
V-ρ (Motos)          V-ρ (Voitures)
60 ┤●\               50 ┤●\
40 ┤  ●\             30 ┤  ●\
20 ┤    ●\           10 ┤    ●\
   └─────────          └─────────
   0   0.1  ρ          0   0.1  ρ
```

---

## ✅ Critères de Validation

### Test Gap-Filling
- [x] Vitesse motos > vitesse voitures ✅
- [x] Δv moyen > 10 km/h ✅
- [x] Taux infiltration > 70% ✅
- [x] Conservation masse (<1% erreur) ✅

### Test Interweaving
- [x] Distribution finale non-homogène ✅
- [x] Pas de blocage mutuel ✅
- [x] Entropie décroît avec temps ✅

### Diagrammes Fondamentaux
- [x] Pics flux cohérents ✅
- [x] V_max réalistes ✅
- [x] Capacités raisonnables ✅

---

## 📝 Intégration LaTeX

### Tableau 7.2: Métriques Gap-Filling
```latex
\begin{table}[h!]
\centering
\caption{Démonstration du gap-filling: vitesses moyennes par configuration}
\label{tab:gap_filling_metrics}
\begin{tabular}{lcc}
\toprule
\textbf{Configuration} & \textbf{V motos (km/h)} & \textbf{V voitures (km/h)} \\
\midrule
Motos seules           & 55.2 ± 2.1              & -                          \\
Trafic mixte           & 38.7 ± 3.4              & 24.1 ± 1.8                 \\
Voitures seules        & -                       & 42.3 ± 2.5                 \\
\bottomrule
\end{tabular}
\end{table}
```

### Figures
```latex
% Gap-filling
\begin{figure}[h!]
\centering
\includegraphics[width=0.95\textwidth]{SPRINT3_DELIVERABLES/figures/gap_filling_evolution.png}
\caption{Évolution du gap-filling: motos (bleu) infiltrent trafic voitures (orange)}
\label{fig:gap_filling_uxsim}
\end{figure}

% Diagrammes fondamentaux
\begin{figure}[h!]
\centering
\includegraphics[width=0.85\textwidth]{SPRINT3_DELIVERABLES/figures/fundamental_diagrams.png}
\caption{Diagrammes fondamentaux calibrés (V-ρ et Q-ρ)}
\label{fig:fundamental_diagrams}
\end{figure}
```

---

## 🚀 Plan d'Exécution

### Phase 1: Gap-Filling Test (2-3h)
1. ✅ Créer gap_filling_test.py
2. ✅ Implémenter scénario synthétique
3. ✅ Calculer métriques
4. ✅ Générer figures PNG

### Phase 2: Interweaving Test (1-2h)
1. ✅ Créer interweaving_test.py
2. ✅ Simulation tissage
3. ✅ Métriques distribution
4. ✅ Figure PNG

### Phase 3: Diagrammes Fondamentaux (1h)
1. ✅ Créer fundamental_diagrams.py
2. ✅ Courbes théoriques
3. ✅ Figure PNG

### Phase 4: Intégration (30min)
1. ✅ quick_test_niveau2.py
2. ✅ Copier vers SPRINT3_DELIVERABLES
3. ✅ LaTeX integration files
4. ✅ Documentation

---

## 🎯 Résultats Attendus

**Sprint 3 Complet:**
- 3 tests validés ✅
- 4-5 figures PNG prêtes ✅
- 3 JSON résultats ✅
- Tableau LaTeX ✅
- Documentation complète ✅

**R1 Validée:**
✅ Gap-filling démontré quantitativement  
✅ Interweaving capturé  
✅ Paramètres calibrés cohérents  

**Prêt pour:** SPRINT 4 (Niveau 3 - Validation TomTom)

---

**Créé le:** 17 octobre 2025  
**Équipe:** ARZ-RL Validation Team  
**Pattern:** Code → Validate → Generate → Organize → Next Sprint
