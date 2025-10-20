# Architecture UXsim Integration - Plan Stratégique

**Date**: 2025-10-16  
**Objectif**: Intégrer UXsim visualizations dans validation_ch7_v2 de manière PROPRE et MAINTENABLE

---

## 📋 CE QUE LE LATEX DEMANDE

D'après `section7_validation_nouvelle_version.tex`, Section 7.6 (RL Performance) requiert:

### Figures Nécessaires

1. **Learning Curve** (`fig:rl_learning_curve_revised`)
   - Courbe d'apprentissage PPO/DQN
   - Récompense moyenne par épisode
   - Convergence vers plateau
   - Format: PNG, 0.7\textwidth

2. **Before/After UXsim** (`fig:before_after_ultimate_revised`)
   - **HAUT**: Baseline (contrôle temps fixe) - congestion (rouge)
   - **BAS**: RL agent - fluidité (vert/jaune)
   - Comparaison visuelle réseau complet
   - Format: PNG, \textwidth (pleine page)
   - **INNOVATION**: Animation disponible via QR code

3. **Metrics Table** (`tab:rl_performance_gains_revised`)
   - Temps de parcours moyen (s)
   - Débit total corridor (véh/h)
   - Délai moyen par véhicule (s)
   - Longueur queue max (véh)
   - Colonnes: Baseline | RL | Amélioration | p-value | Signif.

---

## 🏗️ ARCHITECTURE ACTUELLE

### Composants Existants

1. **arz_model/visualization/uxsim_adapter.py** (347 lignes)
   ```
   ARZtoUXsimVisualizer
   ├── __init__(npz_file_path)           # Charge NPZ ARZ
   ├── create_uxsim_network()            # Convertit grille 1D → réseau 2D
   ├── visualize_snapshot(time_index)    # Figure unique instant
   └── create_animation(output_path)     # GIF/MP4 animation
   ```

2. **validation_ch7_v2/** (Production Ready - 6/6 tests passing)
   ```
   validation_ch7_v2/
   ├── scripts/
   │   ├── domain/                       # Business logic
   │   │   └── section_7_6_rl_performance.py
   │   ├── infrastructure/               # Technical services
   │   │   ├── artifact_manager.py       # Cache/checkpoints
   │   │   ├── session.py                # Output management
   │   │   └── logger.py
   │   ├── orchestration/                # Test coordination
   │   │   └── validation_orchestrator.py
   │   ├── reporting/                    # LaTeX/metrics generation
   │   │   ├── latex_generator.py
   │   │   └── metrics_aggregator.py
   │   └── entry_points/
   │       └── cli.py
   ├── configs/
   │   └── sections/
   │       └── section_7_6.yml
   └── tests/
       └── test_integration_full.py      # ✅ 6/6 passing
   ```

---

## 🎯 ARCHITECTURE PROPRE - PRINCIPE DE SÉPARATION

### Problème Actuel
- ❌ UXsim dans `arz_model/` → validation_ch7_v2 devrait l'importer
- ❌ Dépendance circulaire potentielle
- ❌ Responsabilités mélangées

### Solution: Clean Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│  ENTRY POINT (CLI)                                          │
│  - Parse arguments                                          │
│  - Setup environment                                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  ORCHESTRATION (ValidationOrchestrator)                     │
│  - Coordinate test execution                                │
│  - Manage test lifecycle                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  DOMAIN (RLPerformanceTest)                                 │
│  - Run baseline simulation → NPZ                            │
│  - Train/load RL agent                                      │
│  - Run RL simulation → NPZ                                  │
│  - Compute metrics                                          │
│  - Return ValidationResult                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  REPORTING (LaTeXGenerator + MetricsAggregator)             │
│  - Aggregate metrics from ValidationResult                  │
│  - Generate figures (learning curve, etc.)                  │
│  - Call UXsim visualization IF ENABLED                      │
│  - Generate LaTeX report                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  VISUALIZATION ADAPTER (NEW: uxsim_reporter.py)             │
│  - Import arz_model.visualization.uxsim_adapter             │
│  - Call ARZtoUXsimVisualizer                                │
│  - Handle optional dependency (try/except)                  │
│  - Generate before/after comparison                         │
│  - Store figures via SessionManager                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📐 DESIGN PROPRE - SEPARATION OF CONCERNS

### Principe 1: Domain Ne Connaît PAS UXsim
```python
# ❌ MAUVAIS - Domain layer aware of visualization
class RLPerformanceTest:
    def run(self):
        # ... simulation ...
        self._generate_uxsim_visualizations()  # ❌ TOO COUPLED
```

```python
# ✅ BON - Domain returns data, Reporting handles viz
class RLPerformanceTest:
    def run(self) -> ValidationResult:
        # ... simulation ...
        result.metadata['baseline_npz'] = baseline_npz_path
        result.metadata['rl_npz'] = rl_npz_path
        return result  # ✅ Clean separation
```

### Principe 2: Reporting Layer Responsable de Visualisation
```python
# validation_ch7_v2/scripts/reporting/uxsim_reporter.py (NEW)
class UXsimReporter:
    """
    Adapter between validation results and UXsim visualization.
    
    Responsibility:
    - Take NPZ paths from ValidationResult
    - Call arz_model.visualization.uxsim_adapter
    - Generate figures for LaTeX
    - Handle optional dependency gracefully
    """
    
    def generate_before_after_comparison(
        self,
        baseline_npz: Path,
        rl_npz: Path,
        output_dir: Path,
        config: Dict[str, Any]
    ) -> Dict[str, Path]:
        """
        Generate before/after UXsim comparison figures.
        
        Returns:
            {
                'baseline_snapshot': Path,
                'rl_snapshot': Path,
                'comparison': Path,
                'animation': Path (optional)
            }
        """
```

### Principe 3: Configuration Driven
```yaml
# configs/sections/section_7_6.yml
visualization:
  enabled: true
  uxsim:
    enabled: true
    before_after_comparison:
      baseline_time_index: -1      # End of simulation
      rl_time_index: -1
      comparison_layout: 'vertical'  # or 'horizontal'
    snapshots:
      - time_index: 0
        label: 'initial'
      - time_index: -1
        label: 'final'
    animation:
      enabled: true
      fps: 10
      duration_range: [0, -1]      # Full simulation
  learning_curve:
    enabled: true
    smoothing_window: 50
```

---

## 🔄 FLOW COMPLET - DE LA SIMULATION AU LATEX

### Étape 1: Domain Layer - Génération NPZ
```python
# section_7_6_rl_performance.py
def run(self) -> ValidationResult:
    # 1. Run baseline simulation
    baseline_npz = self._run_arz_simulation(controller='baseline')
    
    # 2. Train/load RL agent
    rl_agent = self._train_or_load_rl_agent()
    
    # 3. Run RL simulation
    rl_npz = self._run_arz_simulation(controller=rl_agent)
    
    # 4. Compute metrics
    metrics = self._compare_performances(baseline_npz, rl_npz)
    
    # 5. Return result with NPZ paths
    result = ValidationResult(passed=True)
    result.metrics = metrics
    result.metadata['baseline_npz'] = baseline_npz
    result.metadata['rl_npz'] = rl_npz
    result.metadata['learning_curve_data'] = training_history
    
    return result
```

### Étape 2: Reporting Layer - Génération Figures
```python
# reporting/latex_generator.py
class LaTeXGenerator:
    def __init__(self, ..., uxsim_reporter: UXsimReporter = None):
        self.uxsim_reporter = uxsim_reporter
    
    def generate_section_7_6_report(
        self,
        result: ValidationResult,
        output_dir: Path
    ) -> Path:
        # 1. Generate learning curve
        learning_fig = self._generate_learning_curve(
            data=result.metadata['learning_curve_data'],
            save_path=output_dir / 'figures' / 'rl_learning_curve.png'
        )
        
        # 2. Generate UXsim visualizations (if enabled)
        uxsim_figs = {}
        if self.config['visualization']['uxsim']['enabled']:
            if self.uxsim_reporter:
                uxsim_figs = self.uxsim_reporter.generate_before_after_comparison(
                    baseline_npz=result.metadata['baseline_npz'],
                    rl_npz=result.metadata['rl_npz'],
                    output_dir=output_dir / 'figures',
                    config=self.config['visualization']['uxsim']
                )
        
        # 3. Generate LaTeX
        latex_content = self._render_template(
            template='section_7_6.tex.j2',
            context={
                'metrics': result.metrics,
                'learning_curve': learning_fig,
                'uxsim_before_after': uxsim_figs.get('comparison'),
                'animation_qr': uxsim_figs.get('animation')
            }
        )
        
        return latex_path
```

### Étape 3: UXsim Reporter - Appel arz_model
```python
# reporting/uxsim_reporter.py (NEW FILE)
from typing import Dict, Any, Optional
from pathlib import Path

class UXsimReporter:
    """Bridge between validation results and UXsim visualization."""
    
    def generate_before_after_comparison(
        self,
        baseline_npz: Path,
        rl_npz: Path,
        output_dir: Path,
        config: Dict[str, Any]
    ) -> Dict[str, Path]:
        """Generate before/after comparison using UXsim adapter."""
        
        try:
            # Optional import - graceful degradation
            from arz_model.visualization.uxsim_adapter import ARZtoUXsimVisualizer
            
            output_dir.mkdir(parents=True, exist_ok=True)
            figures = {}
            
            # 1. Baseline snapshot
            baseline_viz = ARZtoUXsimVisualizer(str(baseline_npz))
            baseline_fig = output_dir / 'baseline_snapshot.png'
            baseline_viz.visualize_snapshot(
                time_index=config.get('baseline_time_index', -1),
                save_path=str(baseline_fig)
            )
            figures['baseline_snapshot'] = baseline_fig
            
            # 2. RL snapshot
            rl_viz = ARZtoUXsimVisualizer(str(rl_npz))
            rl_fig = output_dir / 'rl_snapshot.png'
            rl_viz.visualize_snapshot(
                time_index=config.get('rl_time_index', -1),
                save_path=str(rl_fig)
            )
            figures['rl_snapshot'] = rl_fig
            
            # 3. Side-by-side comparison
            comparison_fig = self._create_comparison_figure(
                baseline_fig, rl_fig,
                layout=config.get('comparison_layout', 'vertical'),
                save_path=output_dir / 'before_after_comparison.png'
            )
            figures['comparison'] = comparison_fig
            
            # 4. Animation (if enabled)
            if config.get('animation', {}).get('enabled', False):
                anim_path = output_dir / 'rl_animation.gif'
                rl_viz.create_animation(
                    output_path=str(anim_path),
                    fps=config['animation'].get('fps', 10)
                )
                figures['animation'] = anim_path
            
            return figures
            
        except ImportError:
            logger.warning("UXsim not available - skipping visualization")
            return {}
        except Exception as e:
            logger.error(f"UXsim visualization failed: {e}")
            return {}
    
    def _create_comparison_figure(
        self,
        baseline_path: Path,
        rl_path: Path,
        layout: str,
        save_path: Path
    ) -> Path:
        """Create side-by-side comparison figure."""
        import matplotlib.pyplot as plt
        from PIL import Image
        
        baseline_img = Image.open(baseline_path)
        rl_img = Image.open(rl_path)
        
        if layout == 'vertical':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        else:  # horizontal
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.imshow(baseline_img)
        ax1.set_title('Baseline (Contrôle Temps Fixe)', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(rl_img)
        ax2.set_title('RL Optimisé (Agent PPO/DQN)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
```

---

## 🎯 FICHIERS À CRÉER / MODIFIER

### Nouveaux Fichiers

1. **validation_ch7_v2/scripts/reporting/uxsim_reporter.py** (250 lignes)
   - UXsimReporter class
   - Bridge between validation and UXsim
   - Optional dependency handling

### Fichiers À Modifier

1. **validation_ch7_v2/scripts/reporting/latex_generator.py**
   - Add UXsimReporter integration
   - Update template rendering

2. **validation_ch7_v2/scripts/domain/section_7_6_rl_performance.py**
   - Remove premature UXsim integration
   - Return NPZ paths in ValidationResult.metadata
   - Keep domain logic pure

3. **validation_ch7_v2/configs/sections/section_7_6.yml**
   - Update visualization config structure
   - Add uxsim-specific settings

4. **validation_ch7_v2/scripts/entry_points/cli.py**
   - Fix setup_logger() call (DONE)
   - Fix ArtifactManager init (DONE)
   - Fix SessionManager init (TODO - needs section name)

---

## ✅ VALIDATION DE L'ARCHITECTURE

### Principes SOLID Respectés

- ✅ **Single Responsibility**: Chaque classe une responsabilité unique
- ✅ **Open/Closed**: Extensible sans modifier existant
- ✅ **Liskov Substitution**: UXsimReporter peut être mocké
- ✅ **Interface Segregation**: Interfaces minimales
- ✅ **Dependency Inversion**: Domain ne dépend pas de UXsim

### Avantages

1. **Testabilité**: UXsimReporter mockable indépendamment
2. **Maintenabilité**: Séparation claire des responsabilités
3. **Évolutivité**: Ajout de nouvelles visualizations facile
4. **Robustesse**: Dépendance optionnelle gérée proprement
5. **Performance**: Génération figures seulement si nécessaire

---

## 🚀 PROCHAINES ÉTAPES (DANS L'ORDRE)

### Phase 1: Corriger CLI (URGENT)
- [x] Fix setup_logger() call
- [x] Fix ArtifactManager init
- [ ] Fix SessionManager init (refactor setup_environment)

### Phase 2: Clean Domain Layer
- [ ] Remove _generate_uxsim_visualizations() from domain
- [ ] Return NPZ paths in ValidationResult.metadata

### Phase 3: Create Reporting Layer
- [ ] Create uxsim_reporter.py
- [ ] Integrate into latex_generator.py

### Phase 4: Test End-to-End
- [ ] Run quick test
- [ ] Verify figures generated
- [ ] Check LaTeX output

### Phase 5: Documentation
- [ ] Update README with UXsim integration
- [ ] Document configuration options

---

## 💡 CONCLUSION

**Architecture Propre = Domain Ignorant de UXsim**

```
Domain Layer:
  "Je génère des NPZ files et des métriques"
  
Reporting Layer:
  "Je prends les NPZ, j'appelle UXsim, je génère des figures"
  
LaTeX:
  "J'inclus les figures générées par Reporting"
```

**Séparation des préoccupations = Code maintenable et testable**
