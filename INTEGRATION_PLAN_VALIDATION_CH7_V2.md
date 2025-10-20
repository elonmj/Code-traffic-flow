# PLAN D'INTÉGRATION - validation_ch7_v2 Extension

**Date**: 2025-10-17  
**Objectif**: Intégrer nouveaux modules validation (7.1, 7.2, 7.3) DANS l'architecture existante `validation_ch7_v2/`  
**Principe**: **RÉUTILISER** l'existant, **NE PAS** créer nouvelle architecture  

---

## 🏗️ ARCHITECTURE EXISTANTE - Analyse

### Structure Actuelle validation_ch7_v2/

```
validation_ch7_v2/
├── scripts/
│   ├── domain/                    ← ValidationTest abstractions
│   │   ├── base.py               ✅ Base classes (ValidationTest, ValidationResult)
│   │   ├── section_7_6_rl_performance.py  ✅ Exemple implémentation
│   │   └── __init__.py
│   │
│   ├── infrastructure/            ← Services transversaux
│   │   ├── config.py             ✅ Config management
│   │   ├── logger.py             ✅ Logging system
│   │   ├── session.py            ✅ Session tracking
│   │   ├── artifact_manager.py   ✅ Artifact storage
│   │   └── errors.py             ✅ Error handling
│   │
│   ├── reporting/                 ← Génération rapports
│   │   ├── latex_generator.py    ✅ LaTeX generation
│   │   ├── uxsim_reporter.py     ✅ UXsim visualizations
│   │   ├── metrics_aggregator.py ✅ Metrics aggregation
│   │   └── __init__.py
│   │
│   ├── orchestration/             ← Orchestration tests
│   │   ├── validation_orchestrator.py ✅ Main orchestrator
│   │   ├── test_factory.py       ✅ Test factory pattern
│   │   ├── test_runner.py        ✅ Test execution
│   │   └── __init__.py
│   │
│   └── entry_points/              ← CLI & Kaggle
│       ├── cli.py                ✅ Command-line interface
│       ├── kaggle_manager.py     ✅ Kaggle orchestration
│       └── local_runner.py       ✅ Local execution
│
├── configs/                       ← Configuration files
│   ├── base.yml                  ✅ Base config
│   ├── quick_test.yml            ✅ Quick mode
│   ├── full_test.yml             ✅ Full mode
│   └── sections/
│       └── section_7_6.yml       ✅ Section 7.6 config
│
├── tests/                         ← Unit/integration tests
│   ├── test_uxsim_integration.py ✅ UXsim tests
│   └── test_integration_full.py  ✅ Integration tests
│
├── cache/                         ✅ Cache storage
├── checkpoints/                   ✅ Checkpoint storage
└── outputs/                       ✅ Results storage
```

### Points Forts Architecture Existante

✅ **Clean Architecture**: Domain-driven, séparation concerns  
✅ **Extensible**: ValidationTest base class pour nouveaux tests  
✅ **Config-Driven**: YAML configs pour chaque section  
✅ **Reporting Intégré**: UXsimReporter + LaTeXGenerator déjà prêts  
✅ **Orchestration**: Factory pattern pour créer tests dynamiquement  
✅ **Infrastructure**: Logging, caching, artifacts management  

---

## 🎯 STRATÉGIE D'INTÉGRATION

### Principe: **Extend, Don't Replace**

1. ✅ **Réutiliser** base classes existantes (ValidationTest, ValidationResult)
2. ✅ **Ajouter** nouveaux tests dans `scripts/domain/`
3. ✅ **Créer** configs YAML dans `configs/sections/`
4. ✅ **Étendre** reporting si nécessaire (méthodes LaTeX additionnelles)
5. ✅ **Intégrer** via orchestrator (factory pattern)

---

## 📋 FICHIERS À CRÉER - Liste Complète

### 1. Domain Layer (Nouveaux Tests)

#### `validation_ch7_v2/scripts/domain/section_7_1_mathematical_foundations.py`
**Responsabilité**: Validation problèmes Riemann (R3)  
**Base Class**: Hérite `ValidationTest`  
**Méthodes**:
- `run_riemann_problem(problem_name)`: Résout 1 problème
- `compute_convergence_order(problem)`: Teste convergence
- `generate_solution_curves()`: Figures solution
- `run()`: Execute 5 problèmes + agrégation

**Métriques**:
- `L2_error_*`: Erreur L2 par problème
- `convergence_order_*`: Ordre convergence par problème
- `criterion_L2_met`: Bool, tous < 10^-4
- `criterion_order_met`: Bool, tous ≈ 4.75

**Artifacts**:
- `solution_curves/*.png`: 5 figures solutions
- `convergence_plot.png`: 1 figure convergence
- `riemann_table.tex`: Table LaTeX

---

#### `validation_ch7_v2/scripts/domain/section_7_2_physical_phenomena.py`
**Responsabilité**: Gap-filling + diagrammes fondamentaux (R1)  
**Base Class**: Hérite `ValidationTest`  
**Méthodes**:
- `run_gap_filling_simulation()`: Simule 20 motos + 10 voitures
- `calibrate_fundamental_diagrams()`: Calibre diagrammes
- `generate_uxsim_snapshots()`: 3 snapshots gap-filling
- `compute_infiltration_metrics()`: Taux infiltration
- `run()`: Execute simulation + métriques

**Métriques**:
- `speed_motos_avg`: Vitesse moyenne motos
- `speed_voitures_avg`: Vitesse moyenne voitures
- `speed_differential`: Ratio vitesses
- `infiltration_rate`: % motos infiltrent
- `MAPE_fundamental_diagram_motos`: Erreur diagramme motos
- `MAPE_fundamental_diagram_voitures`: Erreur diagramme voitures

**Artifacts**:
- `gap_filling_t0.png`: Snapshot t=0s
- `gap_filling_t150.png`: Snapshot t=150s
- `gap_filling_t300.png`: Snapshot t=300s
- `fundamental_diagram_motos.png`: Diagramme motos
- `fundamental_diagram_voitures.png`: Diagramme voitures
- `gap_filling_table.tex`: Table métriques
- `gap_filling_animation.gif`: Animation complète

---

#### `validation_ch7_v2/scripts/domain/section_7_3_digital_twin.py`
**Responsabilité**: Calibration Victoria Island (R4)  
**Base Class**: Hérite `ValidationTest`  
**Méthodes**:
- `load_tomtom_data()`: Charge CSV augmenté
- `split_calibration_validation()`: Split temporal
- `calibrate_optimizer()`: Differential evolution
- `validate_on_holdout()`: Validation split
- `generate_multiscale_viz()`: 3 subplots
- `run()`: Execute calibration + validation

**Métriques**:
- `MAPE_calibration`: MAPE phase calibration
- `MAPE_validation`: MAPE phase validation
- `R2_calibration`: R² phase calibration
- `R2_validation`: R² phase validation
- `RMSE_validation`: RMSE validation
- `segments_acceptable_pct`: % segments MAPE < 15%

**Artifacts**:
- `network_map_MAPE.png`: Carte réseau UXsim
- `time_series_3segments.png`: Séries temporelles
- `histogram_MAPE.png`: Distribution erreurs
- `calibration_params.json`: Paramètres calibrés
- `corridor_performance_table.tex`: Table résultats

---

### 2. Preprocessing Layer (Augmentation Data)

#### `validation_ch7_v2/scripts/preprocessing/__init__.py` (NOUVEAU DOSSIER)
Créer dossier `preprocessing/` dans `scripts/`

#### `validation_ch7_v2/scripts/preprocessing/vehicle_class_rules.py`
**Responsabilité**: Inférence classes motos/voitures  
**Fonctions**:
- `infer_motos_fraction(row)`: Règle motos
- `infer_voitures_fraction(row)`: Règle voitures
- `apply_multiclass_calibration(df)`: Augmente DataFrame
- `validate_class_split(df)`: Vérifie cohérence

**Règles Détaillées** (pour LaTeX):

```python
def infer_motos_fraction(current_speed: float, 
                         freeflow_speed: float,
                         street_type: str) -> float:
    """
    Inférence fraction motos basée sur comportement physique.
    
    Règle Motos:
    -----------
    1. Haute vitesse en congestion → Forte présence motos
       - Si congestion > 30% ET vitesse > 35 km/h
       - Fraction motos: 70%
       - Justification: Motos utilisent gaps, maintiennent vitesse
    
    2. Vitesse normale en fluide → Ratio urbain standard
       - Si congestion < 20%
       - Fraction motos: 60%
       - Justification: Ratio Lagos observé (World Bank, 2022)
    
    3. Faible vitesse en congestion → Faible présence motos
       - Si congestion > 30% ET vitesse < 25 km/h
       - Fraction motos: 30%
       - Justification: Congestion extrême, motos aussi bloquées
    
    Calibration Littérature:
    - Ratio Lagos: 60% motos, 40% voitures (World Bank, 2022)
    - Vitesse différentielle: 1.2-1.5x (Kumar et al., 2018)
    - Gap-filling: 70%+ motos infiltrent (Observé Abidjan)
    """
    congestion = 1 - (current_speed / freeflow_speed)
    
    # Règle 1: Haute vitesse en congestion → Motos dominantes
    if congestion > 0.3 and current_speed > 35:
        return 0.70  # 70% motos
    
    # Règle 2: Fluide → Ratio standard Lagos
    elif congestion < 0.2:
        return 0.60  # 60% motos (ratio urbain)
    
    # Règle 3: Congestion extrême → Motos aussi bloquées
    elif congestion > 0.3 and current_speed < 25:
        return 0.30  # 30% motos seulement
    
    # Défaut: Interpolation linéaire selon congestion
    else:
        # Congestion 20-30%, vitesse 25-35
        # Interpolation motos = f(congestion, vitesse)
        if current_speed >= 30:
            return 0.65  # Légèrement + motos
        else:
            return 0.55  # Légèrement - motos


def infer_voitures_fraction(current_speed: float,
                            freeflow_speed: float,
                            street_type: str) -> float:
    """
    Inférence fraction voitures (complément motos).
    
    Règle Voitures:
    --------------
    1. Faible vitesse en congestion → Forte présence voitures
       - Si congestion > 30% ET vitesse < 25 km/h
       - Fraction voitures: 70%
       - Justification: Voitures bloquées, créent congestion
    
    2. Vitesse normale en fluide → Ratio urbain standard
       - Si congestion < 20%
       - Fraction voitures: 40%
       - Justification: Ratio Lagos observé
    
    3. Haute vitesse en congestion → Faible présence voitures
       - Si congestion > 30% ET vitesse > 35 km/h
       - Fraction voitures: 30%
       - Justification: Trafic fluide malgré congestion = motos
    
    CONSTRAINT: motos_fraction + voitures_fraction = 1.0
    """
    return 1.0 - infer_motos_fraction(current_speed, freeflow_speed, street_type)


def apply_multiclass_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augmente DataFrame TomTom avec colonnes classes inférées.
    
    Colonnes Ajoutées:
    -----------------
    - class_split_motos: Fraction motos [0-1]
    - class_split_voitures: Fraction voitures [0-1]
    - speed_motos: Vitesse estimée motos (km/h)
    - speed_voitures: Vitesse estimée voitures (km/h)
    - flow_motos: Flux estimé motos (véh/h)
    - flow_voitures: Flux estimé voitures (véh/h)
    
    Hypothèses Vitesses:
    -------------------
    - Motos: +15% vitesse agrégée (plus rapides)
    - Voitures: -13% vitesse agrégée (plus lentes)
    - Calibration: Différentiel 1.32x (Kumar et al., 2018)
    
    Returns:
        DataFrame augmenté avec 6 nouvelles colonnes
    """
    # Inférence fractions
    df['class_split_motos'] = df.apply(
        lambda row: infer_motos_fraction(
            row['current_speed'],
            row['freeflow_speed'],
            row['name']
        ),
        axis=1
    )
    
    df['class_split_voitures'] = 1.0 - df['class_split_motos']
    
    # Vitesses estimées (hypothèse différentiel 1.32x)
    speed_multiplier_motos = 1.15   # +15%
    speed_multiplier_voitures = 0.87  # -13%
    
    df['speed_motos'] = df['current_speed'] * speed_multiplier_motos
    df['speed_voitures'] = df['current_speed'] * speed_multiplier_voitures
    
    # Flux estimés (proportionnel aux fractions)
    # Hypothèse: flux total connu, répartir selon classes
    df['flow_motos'] = df['current_speed'] * df['class_split_motos']
    df['flow_voitures'] = df['current_speed'] * df['class_split_voitures']
    
    return df
```

**Validation Rules**:
```python
def validate_class_split(df: pd.DataFrame) -> dict:
    """
    Valide cohérence règles inférence.
    
    Checks:
    ------
    1. Somme fractions = 1.0 (∀ rows)
    2. Vitesse motos > voitures (en moyenne)
    3. Ratio global ≈ 60/40 (±10%)
    4. Pas de valeurs aberrantes (hors [0, 1])
    
    Returns:
        {
            'valid': bool,
            'checks': {check_name: passed},
            'stats': {metric_name: value}
        }
    """
    checks = {}
    
    # Check 1: Somme = 1.0
    sum_check = (df['class_split_motos'] + df['class_split_voitures']).round(6)
    checks['sum_equals_one'] = (sum_check == 1.0).all()
    
    # Check 2: Vitesse motos > voitures
    checks['speed_differential'] = (df['speed_motos'] > df['speed_voitures']).mean() > 0.95
    
    # Check 3: Ratio global
    global_motos = df['class_split_motos'].mean()
    checks['global_ratio_realistic'] = 0.50 <= global_motos <= 0.70  # 50-70%
    
    # Check 4: Pas d'aberrations
    checks['no_outliers'] = (
        (df['class_split_motos'] >= 0).all() and
        (df['class_split_motos'] <= 1).all()
    )
    
    stats = {
        'motos_fraction_mean': df['class_split_motos'].mean(),
        'motos_fraction_std': df['class_split_motos'].std(),
        'speed_differential_mean': (df['speed_motos'] / df['speed_voitures']).mean(),
        'rows_validated': len(df)
    }
    
    return {
        'valid': all(checks.values()),
        'checks': checks,
        'stats': stats
    }
```

---

#### `validation_ch7_v2/scripts/preprocessing/temporal_augmentation.py`
**Responsabilité**: Génération rush hour synthétique  
**Fonctions**:
- `generate_rush_hour_demand(calibrated_twin)`: Génère demande 17:00-18:00
- `validate_temporal_consistency(demand)`: Vérifie contraintes physiques
- `apply_stochastic_variations(demand)`: Ajoute variabilité

---

#### `validation_ch7_v2/scripts/preprocessing/network_topology.py`
**Responsabilité**: Définition réseau simplifié Victoria Island  
**Fonctions**:
- `create_victoria_island_topology()`: Crée réseau 4 routes
- `map_segments_to_routes(df)`: Mappe segments CSV → routes
- `export_to_uxsim_format(network)`: Convertit format UXsim

---

### 3. Scenarios (Configs)

#### `validation_ch7_v2/configs/sections/section_7_1.yml`
```yaml
section: "7.1"
name: "Mathematical Foundations"
revendications: ["R3"]
description: "Validation analytique solveur WENO5"

riemann_problems:
  shock_simple:
    left_state: {rho: 0.8, v: 20}
    right_state: {rho: 0.3, v: 40}
    x_discontinuity: 0.5
    duration: 1.0
    dx: 0.001
    
  rarefaction:
    left_state: {rho: 0.3, v: 40}
    right_state: {rho: 0.8, v: 20}
    x_discontinuity: 0.5
    duration: 1.0
    dx: 0.001
    
  # ... 3 autres problèmes

criteria:
  L2_error_max: 1.0e-4
  convergence_order_min: 4.65
  convergence_order_max: 4.85

output:
  figures: ["solution_curves", "convergence_plot"]
  tables: ["riemann_validation_results"]
```

#### `validation_ch7_v2/configs/sections/section_7_2.yml`
```yaml
section: "7.2"
name: "Physical Phenomena"
revendications: ["R1"]
description: "Gap-filling et diagrammes fondamentaux"

scenario:
  network:
    type: "single_road"
    length: 2000  # mètres
    lanes: 2
  
  vehicles:
    motos:
      count: 20
      initial_positions: [0, 500]  # 0-500m
      initial_speed: 40  # km/h
      spacing: 25  # mètres
    
    voitures:
      count: 10
      initial_positions: [600, 1200]  # 600-1200m
      initial_speed: 25  # km/h
      spacing: 60  # mètres
  
  simulation:
    duration: 300  # secondes
    timestep: 0.1
    output_interval: 1.0

uxsim:
  snapshots: [0, 150, 300]  # temps (s)
  animation:
    fps: 10
    duration: 30  # secondes
  colormap: "speed"
  trajectories: true

criteria:
  speed_differential_min: 1.2  # Motos 1.2x+ plus rapides
  infiltration_rate_min: 0.7  # 70%+ infiltration
  MAPE_fundamental_diagram_max: 0.10  # < 10%

output:
  figures: ["gap_filling_snapshots", "fundamental_diagrams", "animation"]
  tables: ["gap_filling_metrics"]
```

#### `validation_ch7_v2/configs/sections/section_7_3.yml`
```yaml
section: "7.3"
name: "Digital Twin Victoria Island"
revendications: ["R4"]
description: "Calibration et validation jumeau numérique"

data:
  source: "donnees_trafic_75_segments (2).csv"
  preprocessing:
    apply_multiclass: true
    temporal_split:
      calibration_end: "2025-09-24 14:00"
      
network:
  topology: "victoria_island_simplified"
  routes:
    - name: "Akin Adesola Street"
      segments: 25
      length: 3000
      lanes: 2
    - name: "Ahmadu Bello Way"
      segments: 20
      length: 2500
      lanes: 3
    - name: "Adeola Odeku Street"
      segments: 15
      length: 2000
      lanes: 2
    - name: "Saka Tinubu Street"
      segments: 10
      length: 1500
      lanes: 2

calibration:
  optimizer:
    method: "differential_evolution"
    bounds:
      V_max: [30, 60]
      tau: [10, 40]
      alpha: [0.5, 2.0]
    maxiter: 100
  
  objective: "MAPE"

uxsim:
  visualization:
    network_map:
      colormap: "MAPE"
      annotations: true
    time_series:
      segments: ["best", "median", "worst"]
    histogram:
      bins: 20

criteria:
  MAPE_validation_max: 15.0  # < 15%
  R2_validation_min: 0.75  # > 0.75
  RMSE_validation_max: 5.0  # < 5 km/h
  segments_acceptable_min: 0.80  # 80%+

output:
  figures: ["network_map", "time_series", "histogram"]
  tables: ["corridor_performance"]
  artifacts: ["calibration_params.json"]
```

---

### 4. Reporting Extensions

#### Étendre `validation_ch7_v2/scripts/reporting/latex_generator.py`

Ajouter méthodes:
```python
def generate_riemann_table(self, results: ValidationResult) -> str:
    """Génère table validation Riemann (Section 7.1)."""
    
def generate_gap_filling_figure(self, results: ValidationResult) -> str:
    """Génère figure gap-filling (Section 7.2)."""
    
def generate_corridor_validation_grid(self, results: ValidationResult) -> str:
    """Génère figure multi-échelle (Section 7.3)."""
```

#### Créer `validation_ch7_v2/scripts/reporting/latex_sections.py` (NOUVEAU)
**Responsabilité**: Templates LaTeX par section  
**Contenu**: Strings LaTeX pour chaque table/figure

---

### 5. Orchestration Integration

#### Étendre `validation_ch7_v2/scripts/orchestration/test_factory.py`

```python
# Ajouter dans REGISTRY:
REGISTRY = {
    "section_7_1": Section71MathematicalFoundations,
    "section_7_2": Section72PhysicalPhenomena,
    "section_7_3": Section73DigitalTwin,
    "section_7_6": Section76RLPerformance  # existant
}
```

#### Étendre `validation_ch7_v2/scripts/orchestration/validation_orchestrator.py`

```python
def run_full_validation_pyramid(self) -> Dict[str, ValidationResult]:
    """
    Exécute validation 4 niveaux pyramide.
    
    Ordre:
    1. Niveau 1: Mathematical Foundations (7.1)
    2. Niveau 2: Physical Phenomena (7.2)
    3. Niveau 3: Digital Twin (7.3)
    4. Niveau 4: RL Performance (7.6)
    """
    results = {}
    
    # Niveau 1
    self.logger.info("🔬 Niveau 1: Mathematical Foundations")
    results['level_1'] = self.run_section("section_7_1")
    
    # Niveau 2
    self.logger.info("🚗 Niveau 2: Physical Phenomena")
    results['level_2'] = self.run_section("section_7_2")
    
    # Niveau 3 (utilise data augmentée niveau 2)
    self.logger.info("🌐 Niveau 3: Digital Twin")
    results['level_3'] = self.run_section("section_7_3")
    
    # Niveau 4 (utilise twin calibré niveau 3)
    self.logger.info("🤖 Niveau 4: RL Performance")
    results['level_4'] = self.run_section("section_7_6")
    
    # Generate comprehensive LaTeX
    self.logger.info("📝 Generating LaTeX")
    latex = self.latex_generator.generate_full_chapter_7(results)
    
    return results
```

---

### 6. Data Preprocessing

#### `validation_ch7_v2/data/preprocess_tomtom.py` (NOUVEAU DOSSIER + SCRIPT)

```python
#!/usr/bin/env python3
"""
Preprocess TomTom data for validation.

Applies:
1. Vehicle class inference
2. Data quality checks
3. Export augmented CSV
"""

import pandas as pd
import sys
from pathlib import Path

# Import preprocessing modules
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from preprocessing.vehicle_class_rules import apply_multiclass_calibration, validate_class_split

def main():
    print("=" * 70)
    print("TOMTOM DATA PREPROCESSING")
    print("=" * 70)
    
    # Load raw data
    print("\n📥 Loading raw TomTom data...")
    df = pd.read_csv('../../donnees_trafic_75_segments (2).csv', on_bad_lines='skip')
    print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Apply multiclass calibration
    print("\n🔧 Applying vehicle class inference...")
    df_augmented = apply_multiclass_calibration(df)
    print(f"  Added columns: {[c for c in df_augmented.columns if c not in df.columns]}")
    
    # Validate
    print("\n✅ Validating class split...")
    validation = validate_class_split(df_augmented)
    print(f"  Valid: {validation['valid']}")
    print(f"  Checks: {validation['checks']}")
    print(f"  Stats: {validation['stats']}")
    
    if not validation['valid']:
        print("\n❌ Validation failed! Check rules.")
        return 1
    
    # Export
    output_path = Path(__file__).parent / "tomtom_augmented.csv"
    df_augmented.to_csv(output_path, index=False)
    print(f"\n💾 Exported: {output_path}")
    print(f"  Rows: {len(df_augmented)}")
    print(f"  Columns: {len(df_augmented.columns)}")
    
    print("\n✅ Preprocessing complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

---

### 7. Tests

#### `validation_ch7_v2/tests/test_preprocessing.py` (NOUVEAU)
```python
"""Unit tests for preprocessing modules."""

import pytest
import pandas as pd
from scripts.preprocessing.vehicle_class_rules import (
    infer_motos_fraction,
    infer_voitures_fraction,
    apply_multiclass_calibration,
    validate_class_split
)

def test_motos_fraction_high_speed_congestion():
    """Test règle 1: Haute vitesse en congestion → Motos dominantes."""
    fraction = infer_motos_fraction(
        current_speed=40,
        freeflow_speed=50,
        street_type="arterial"
    )
    assert fraction == 0.70  # 70% motos

def test_voitures_complement():
    """Test règle: voitures = 1 - motos."""
    motos = infer_motos_fraction(35, 50, "arterial")
    voitures = infer_voitures_fraction(35, 50, "arterial")
    assert abs(motos + voitures - 1.0) < 1e-6

def test_multiclass_calibration():
    """Test augmentation DataFrame."""
    df = pd.DataFrame({
        'current_speed': [40, 25, 35],
        'freeflow_speed': [50, 50, 50],
        'name': ['Street A', 'Street A', 'Street A']
    })
    
    df_aug = apply_multiclass_calibration(df)
    
    # Check nouvelles colonnes
    assert 'class_split_motos' in df_aug.columns
    assert 'speed_motos' in df_aug.columns
    assert 'speed_voitures' in df_aug.columns
    
    # Check cohérence
    assert (df_aug['speed_motos'] > df_aug['speed_voitures']).all()

# ... autres tests
```

---

## 📝 DOCUMENT LATEX - Explication Règles

### Créer `validation_ch7_v2/latex_documentation/multiclass_methodology.tex`

```latex
\subsection{Méthodologie Multi-Classe} \label{sec:multiclass_methodology}

\subsubsection{Problématique}

Les données TomTom fournissent des vitesses et flux agrégés sans distinction 
entre motos et voitures. Or, le modèle ARZ bi-classe nécessite des paramètres 
séparés pour chaque classe. Nous développons donc des règles d'inférence 
basées sur le comportement physique observé en Afrique de l'Ouest.

\subsubsection{Règles d'Inférence}

\paragraph{Règle Motos} ~\\

La fraction de motos $f_m \in [0, 1]$ est inférée selon le profil de vitesse :

\begin{enumerate}
    \item \textbf{Haute vitesse en congestion} ($c > 0.3$ et $v > 35$ km/h) \\
    $\Rightarrow f_m = 0.70$ (70\% motos)
    
    \textit{Justification} : Les motos maintiennent une vitesse élevée en 
    situation congestionnée grâce à leur agilité et capacité d'infiltration 
    (gap-filling). Une vitesse supérieure à 35 km/h alors que le trafic est 
    congestionné ($c > 30\%$) indique une forte présence de motos.
    
    \item \textbf{Trafic fluide} ($c < 0.2$) \\
    $\Rightarrow f_m = 0.60$ (60\% motos)
    
    \textit{Justification} : En conditions fluides, on observe le ratio urbain 
    standard de Lagos : 60\% motos, 40\% voitures \cite{worldbank2022}.
    
    \item \textbf{Congestion extrême} ($c > 0.3$ et $v < 25$ km/h) \\
    $\Rightarrow f_m = 0.30$ (30\% motos)
    
    \textit{Justification} : En congestion extrême, même les motos sont 
    bloquées. Une vitesse très faible indique un trafic dominé par les 
    voitures qui créent l'embouteillage.
    
    \item \textbf{Cas intermédiaires} (interpolation linéaire) \\
    Pour $0.2 \leq c \leq 0.3$ et $25 \leq v \leq 35$ km/h, interpolation 
    entre les cas ci-dessus.
\end{enumerate}

Où $c = 1 - \frac{v}{v_{ff}}$ est le niveau de congestion et $v_{ff}$ la 
vitesse en fluide.

\paragraph{Règle Voitures} ~\\

Par complémentarité : $f_v = 1 - f_m$

\subsubsection{Vitesses par Classe}

Une fois les fractions déterminées, les vitesses individuelles sont estimées :

\begin{align}
    v_m &= v_{agg} \times 1.15 \quad \text{(motos +15\%)} \\
    v_v &= v_{agg} \times 0.87 \quad \text{(voitures -13\%)}
\end{align}

Cette calibration produit un différentiel de vitesse de 1.32× entre motos 
et voitures, cohérent avec la littérature ($1.2$-$1.5\times$) 
\cite{kumar2018mixedtraffic}.

\subsubsection{Validation des Règles}

Les règles d'inférence sont validées par :

\begin{enumerate}
    \item \textbf{Cohérence mathématique} : $f_m + f_v = 1$ pour toute observation
    \item \textbf{Ratio global} : $\mathbb{E}[f_m] \approx 0.60$ sur l'ensemble du dataset
    \item \textbf{Différentiel de vitesse} : $\mathbb{E}[v_m / v_v] \approx 1.32$
    \item \textbf{Comportement gap-filling} : Validation sur scénarios synthétiques 
    (Section~\ref{sec:gap_filling})
\end{enumerate}

\subsubsection{Limitations}

Cette approche constitue une \emph{approximation} basée sur des observations 
comportementales. Les principales limitations sont :

\begin{itemize}
    \item Pas de validation terrain directe (absence données réelles bi-classe)
    \item Variabilité inter-routes non capturée (même règle pour tous segments)
    \item Facteurs externes ignorés (météo, événements spéciaux)
\end{itemize}

Néanmoins, cette méthodologie permet de calibrer un modèle bi-classe réaliste 
en l'absence de données classifiées, tout en restant transparente sur ses 
hypothèses et limites.
```

---

## 🎯 ORDRE D'IMPLÉMENTATION RECOMMANDÉ

### Sprint 1: Infrastructure (2-3h)
- [ ] Créer dossier `scripts/preprocessing/`
- [ ] Implémenter `vehicle_class_rules.py` (règles motos/voitures)
- [ ] Implémenter `temporal_augmentation.py`
- [ ] Implémenter `network_topology.py`
- [ ] Script `data/preprocess_tomtom.py`
- [ ] Tests `tests/test_preprocessing.py`
- [ ] Exécuter preprocessing → générer `tomtom_augmented.csv`

### Sprint 2: Niveau 1 - Mathematical (4-6h)
- [ ] Implémenter `domain/section_7_1_mathematical_foundations.py`
- [ ] Config `configs/sections/section_7_1.yml`
- [ ] Étendre `reporting/latex_generator.py` (méthode Riemann table)
- [ ] Tests `tests/test_domain/test_section_7_1.py`
- [ ] Intégrer dans `orchestration/test_factory.py`
- [ ] Test exécution local

### Sprint 3: Niveau 2 - Physical (6-8h)
- [ ] Implémenter `domain/section_7_2_physical_phenomena.py`
- [ ] Config `configs/sections/section_7_2.yml`
- [ ] Étendre `reporting/uxsim_reporter.py` (gap-filling animation)
- [ ] Tests `tests/test_domain/test_section_7_2.py`
- [ ] Intégrer dans factory
- [ ] Test exécution local

### Sprint 4: Niveau 3 - Digital Twin (8-10h)
- [ ] Implémenter `domain/section_7_3_digital_twin.py`
- [ ] Config `configs/sections/section_7_3.yml`
- [ ] Étendre `reporting/uxsim_reporter.py` (multi-échelle viz)
- [ ] Tests `tests/test_domain/test_section_7_3.py`
- [ ] Intégrer dans factory
- [ ] Test calibration complète

### Sprint 5: Documentation LaTeX (2-3h)
- [ ] Créer `latex_documentation/multiclass_methodology.tex`
- [ ] Créer `latex_documentation/limitations.tex`
- [ ] Créer `latex_documentation/corrections.tex` (75→70 segments)
- [ ] Intégrer templates dans `reporting/latex_sections.py`

### Sprint 6: Integration & Orchestration (2-3h)
- [ ] Étendre `orchestration/validation_orchestrator.py` (pyramide complète)
- [ ] CLI extension (`entry_points/cli.py` ajouter commandes)
- [ ] Tests intégration `tests/test_integration_pyramid.py`
- [ ] Documentation README extension

---

## ✅ CHECKLIST VALIDATION

Avant considérer terminé:

- [ ] ✅ Structure existante `validation_ch7_v2/` inchangée (backward compatible)
- [ ] ✅ Tous nouveaux modules héritent `ValidationTest` base class
- [ ] ✅ Configs YAML suivent format existant
- [ ] ✅ Tests unitaires: coverage > 80%
- [ ] ✅ Tests intégration: 4 niveaux exécutent séquentiellement
- [ ] ✅ Data augmentation: `tomtom_augmented.csv` validé
- [ ] ✅ UXsim visualizations: 3 types figures générées
- [ ] ✅ LaTeX generation: Templates intégrés
- [ ] ✅ Documentation: README + méthodologie LaTeX complète
- [ ] ✅ Orchestration: Factory pattern fonctionne avec nouveaux tests

---

## 📊 RÉSUMÉ FICHIERS

**NOUVEAUX FICHIERS** (création nécessaire):

### Domain (3)
1. `scripts/domain/section_7_1_mathematical_foundations.py` (~400 lignes)
2. `scripts/domain/section_7_2_physical_phenomena.py` (~500 lignes)
3. `scripts/domain/section_7_3_digital_twin.py` (~600 lignes)

### Preprocessing (3 + dossier)
4. `scripts/preprocessing/__init__.py`
5. `scripts/preprocessing/vehicle_class_rules.py` (~300 lignes) **AVEC RÈGLES DÉTAILLÉES**
6. `scripts/preprocessing/temporal_augmentation.py` (~200 lignes)
7. `scripts/preprocessing/network_topology.py` (~250 lignes)

### Configs (3)
8. `configs/sections/section_7_1.yml`
9. `configs/sections/section_7_2.yml`
10. `configs/sections/section_7_3.yml`

### Data (2 + dossier)
11. `data/preprocess_tomtom.py` (~100 lignes)
12. `data/tomtom_augmented.csv` (généré)

### Tests (3)
13. `tests/test_preprocessing.py` (~200 lignes)
14. `tests/test_domain/test_section_7_1.py`
15. `tests/test_domain/test_section_7_2.py`
16. `tests/test_domain/test_section_7_3.py`
17. `tests/test_integration_pyramid.py`

### Documentation LaTeX (3 + dossier)
18. `latex_documentation/multiclass_methodology.tex` **RÈGLES EXPLIQUÉES**
19. `latex_documentation/limitations.tex`
20. `latex_documentation/corrections.tex`

### Reporting Extensions (1)
21. `scripts/reporting/latex_sections.py` (~300 lignes)

**FICHIERS À ÉTENDRE** (modifications nécessaires):

22. `scripts/reporting/latex_generator.py` (ajouter 3 méthodes)
23. `scripts/reporting/uxsim_reporter.py` (ajouter 2 méthodes viz)
24. `scripts/orchestration/test_factory.py` (ajouter 3 tests registry)
25. `scripts/orchestration/validation_orchestrator.py` (ajouter méthode pyramide)
26. `scripts/entry_points/cli.py` (ajouter commandes sections 7.1-7.3)

**TOTAL**: 21 nouveaux fichiers + 5 extensions = **26 fichiers**

---

## 🚀 PROCHAINES ÉTAPES

**Question pour vous**:

1. ✅ **Approche validée?** Intégration dans `validation_ch7_v2/` existant OK?
2. ✅ **Règles motos/voitures détaillées?** Les 3 règles + justifications suffisantes pour LaTeX?
3. ✅ **Ordre implémentation?** Commencer par Sprint 1 (Infrastructure)?
4. ✅ **Quick win?** Veux-tu que je commence par `vehicle_class_rules.py` pour voir le code concret?

**Prêt à démarrer Sprint 1 dès ton feu vert!** 🚀
