# Section 7.5 - Digital Twin Validation (Kaggle GPU)

## Vue d'ensemble

Ce script lance la validation du jumeau numérique (Digital Twin) sur Kaggle GPU pour tester les revendications R4 et R6.

### Revendications testées

- **R4**: Reproduction des comportements de trafic observés
- **R6**: Robustesse sous conditions dégradées

## Lancement rapide

```powershell
cd "d:\Projets\Alibi\Code project"
python validation_ch7\scripts\run_kaggle_validation_section_7_5.py
```

## Tests effectués

### 1. Behavioral Reproduction (R4)
Trois scénarios de trafic typiques :
- **Free flow** : Trafic fluide (10-20 veh/km, 72-100 km/h)
- **Congestion** : Congestion modérée (50-80 veh/km, 29-54 km/h)
- **Jam formation** : Formation de bouchon (80-100 veh/km, 7-29 km/h)

**Critères de validation** :
- Densité dans la plage attendue
- Vitesse cohérente avec le régime de trafic
- Conservation de masse < 1% erreur

### 2. Robustness Tests (R6)
Trois perturbations appliquées au scénario free_flow :
- **Density increase +50%** : Augmentation de la demande
- **Velocity decrease -30%** : Conditions météo défavorables
- **Road degradation (R=1)** : Qualité d'infrastructure dégradée

**Critères de validation** :
- Stabilité numérique (pas de NaN ou explosions)
- Temps de convergence < seuils (150-200s)
- RMSE final acceptable

### 3. Cross-Scenario Validation
Vérification de la cohérence du diagramme fondamental :
- Relation densité-vitesse monotone décroissante

## Outputs générés

### Figures (300 DPI PNG)
```
chapters/partie3/images/
├── fig_behavioral_patterns.png       # Patterns pour 3 scénarios
├── fig_robustness_perturbations.png  # Réponse aux perturbations
├── fig_fundamental_diagram.png       # Diagramme fondamental
└── fig_digital_twin_metrics.png      # Résumé métriques
```

### Données (CSV)
```
validation_output/results/{kernel_slug}/section_7_5_digital_twin/data/metrics/
├── behavioral_metrics.csv     # Métriques par scénario
├── robustness_metrics.csv     # Métriques par perturbation
└── summary_metrics.csv        # Résumé global
```

### LaTeX
```
chapters/partie3/section_7_5_digital_twin_content.tex
```

Contenu :
- Méthodologie détaillée
- Tableaux de résultats
- Figures avec légendes
- Discussion (forces, limitations)
- Conclusion

## Intégration dans la thèse

Dans `chapters/partie3/ch7_validation_entrainement.tex`, ajouter :

```latex
\input{chapters/partie3/section_7_5_digital_twin_content.tex}
```

## Temps d'exécution

- **Estimé** : 90-120 minutes sur Kaggle GPU T4
- **Timeout** : 4 heures (240 minutes)

## Architecture

Le script utilise `ValidationSection` (comme Section 7.4) pour :
- Structure organisée automatique
- Génération figures publication-ready
- Export CSV et LaTeX
- Session summary JSON

## Dépendances

- `validation_kaggle_manager.py` : Upload et monitoring Kaggle
- `test_section_7_5_digital_twin.py` : Script de test principal
- `validation_utils.py` : Utilitaires de validation
- `SimulationRunner` : Moteur de simulation ARZ réel

## Résultats attendus

### Success Criteria
- R4 : 100% scénarios validés (3/3)
- R6 : 100% perturbations validées (3/3)
- Cross-scenario : Monotonie vérifiée

### Métriques typiques
- Densités : 0.010-0.100 veh/m
- Vitesses : 2.0-28.0 m/s
- Conservation masse : < 0.1% erreur
- Convergence : 50-150s

## Troubleshooting

### Kernel timeout
Si timeout, augmenter dans le script :
```python
timeout=18000  # 5 heures au lieu de 4
```

### Simulation fails
Vérifier les configurations de scénarios dans :
```
validation_output/results/{kernel_slug}/section_7_5_digital_twin/data/scenarios/
```

### Missing figures
Vérifier que `setup_publication_style()` est appelé avant génération figures.

## Notes

- Utilise **vraies simulations** (pas de mock data)
- GPU recommandé pour performance
- Génération automatique de tous les artefacts
- Copie automatique des figures dans `chapters/partie3/images/`
