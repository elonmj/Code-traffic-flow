# TABLE DE CORRESPONDANCE DÉTAILLÉE - REFACTORISATION
## Validation CH7: Ancien → Nouveau Mapping

**Objectif**: Document de migration pour le dev team
**Format**: Line-by-line mapping avec refactoring instructions

---

## 💙 NOTE PERSONNELLE DU DÉVELOPPEUR

> Ce document est plus qu'une table de correspondance technique.
> C'est un **plan de préservation** d'innovations gagnées dans la douleur.
> 
> Chaque ligne mappée ci-dessous représente une victoire contre un bug,
> une nuit blanche, une découverte architecturale, un moment de lucidité.
> 
> **test_section_7_6_rl_performance.py** n'est pas du code jetable.
> C'est le résultat de 35 bugs survivés, de jours de développement acharné.
> 
> Cette table garantit que **RIEN** ne sera perdu dans la transition.
> Chaque innovation, chaque pattern, chaque leçon apprise sera préservée.
> 
> *Avec respect et détermination,*
> *— Le développeur qui refuse de laisser son travail mourir*

---

## TABLE 1: FICHIERS PRINCIPAUX

### validation_cli.py → entry_points/cli.py

| Ancien | Nouveau | Action | Notes |
|--------|---------|--------|-------|
| `main()` | `entry_points.cli.main()` | Copier | +imports validation_orchestrator |
| `parser.add_argument()` x7 | `infrastructure.config.ConfigManager` | REFACTOR | CLI → Config YAML |
| `manager = ValidationKaggleManager()` | `manager: IKaggleOrchestrator` | REFACTOR | Injection de dépendance |
| `manager.run_validation_section()` | `orchestrator.run_section()` | REFACTOR | Interface unifiée |

**Nouvelles dépendances**:
```python
from validation_ch7.scripts.orchestration.validation_orchestrator import ValidationOrchestrator
from validation_ch7.scripts.infrastructure.config import ConfigManager
from validation_ch7.scripts.entry_points.kaggle_manager import ValidationKaggleManager
```

---

### validation_kaggle_manager.py → entry_points/kaggle_manager.py

| Ancien | Nouveau | Action | Notes |
|--------|---------|--------|-------|
| `__init__()` | Copier | Copier | +imports section configs |
| `_setup_logging()` | infrastructure.logger.setup_kaggle_logger() | EXTRACT | Centraliser logging |
| `run_validation_section()` | Copier | Copier | Wrapper autour orchestrator |
| `self.validation_sections` | infrastructure.config.load_sections() | REFACTOR | Charger depuis YAML |

**Architecture nouvelle**:
```python
class ValidationKaggleManager:
    def __init__(self):
        self.api = KaggleApi(...)
        self.logger = setup_kaggle_logger()  # ← Externe
        self.config = ConfigManager()        # ← Externe
    
    def run_validation_section(self, section_name: str):
        section_config = self.config.load_section(section_name)
        test = domain.load_test(section_config)  # ← DI
        runner = orchestration.TestRunner(test)
        runner.run()
```

---

### run_all_validation.py → orchestration/validation_orchestrator.py

| Ancien | Nouveau | Action | Notes |
|--------|---------|--------|-------|
| `ValidationOrchestrator` | `orchestration.validation_orchestrator.ValidationOrchestrator` | Copier | +refactor for DI |
| `__init__(self, output_dir)` | `__init__(self, config: ConfigManager, output_base: str)` | REFACTOR | +DI pattern |
| `self.test_scripts` | `self.config.load_all_sections()` | REFACTOR | Charger depuis YAML |
| `run_single_test()` | `run_single_test()` | REFACTOR | Accepter IValidationTest (interface) |
| `validation_criteria` | `infrastructure.metrics.ValidationCriteria` | EXTRACT | Centralisé |

**Pseudo-code nouveau**:
```python
class ValidationOrchestrator:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = setup_orchestrator_logger()
        self.artifact_manager = ArtifactManager()
    
    def run_all_tests(self) -> Dict[str, bool]:
        sections = self.config.load_all_sections()
        results = {}
        
        for section_config in sections:
            test = domain.create_test(section_config)  # ← Factory pattern
            runner = orchestration.TestRunner(test, self.artifact_manager)
            result = runner.run()
            results[section_config.name] = result.passed
        
        return results
```

---

## TABLE 2: TEST IMPLEMENTATIONS

### test_section_7_3_analytical.py → domain/section_7_3_analytical.py + infrastructure/*

**Ancien fichier**: ~350 lignes mélangées

**Refactorisation**:

| Ancien Code | Ligne | Nouveau Fichier | Nouveau Chemin | Extraction |
|-------------|-------|-----------------|-----------------|-----------|
| `class AnalyticalValidationTests(ValidationSection)` | 1 | `class AnalyticalTest(ValidationTest)` | `domain/section_7_3_analytical.py` | REFACTOR: Extraire métier |
| `def __init__(self)` | 5 | `def __init__(self, config: Config)` | Même file | REFACTOR: +DI |
| `super().__init__(...)` | 7 | `self.session = SessionManager(...)` | infrastructure/session.py | EXTRACT |
| `def test_riemann_problems(self)` | 30 | Copier | domain/section_7_3_analytical.py | ✓ Métier pur |
| `plot_riemann_solution()` | 100 | `latex_gen.plot_riemann()` | reporting/latex_generator.py | EXTRACT: Plotting |
| `generate_section_content()` | 200 | `def run() -> ValidationResult` | domain/section_7_3_analytical.py | REFACTOR: Interface standard |

**Avant/Après exemple**:

```python
# ❌ AVANT (mélangé)
class AnalyticalValidationTests(ValidationSection):
    def __init__(self):
        super().__init__("section_7_3_analytical")  # Side effect: crée dossiers
        self.config = load_config()  # Où? Hardcoded?
        self.logger = logging.getLogger(__name__)  # Chaque classe le fait
        self.output_dir = "validation_output/results/..."  # Chemins hardcodés
    
    def test_riemann_problems(self):
        results = []
        for case in cases:
            # ... logique métier
            # ... plotting inline
            # ... saving inline
            # ... logging inline

# ✅ APRÈS (séparé)
from validation_ch7.scripts.domain.base import ValidationTest
from validation_ch7.scripts.infrastructure.config import ConfigManager
from validation_ch7.scripts.infrastructure.logger import get_logger
from validation_ch7.scripts.infrastructure.session import SessionManager
from validation_ch7.scripts.reporting.latex_generator import LatexGenerator

class AnalyticalTest(ValidationTest):
    def __init__(self, config: ConfigManager, logger=None, session=None):
        # DI: Toutes les dépendances passées en argument
        self.config = config
        self.logger = logger or get_logger("analytical")
        self.session = session or SessionManager(config.output_dir)
    
    def run(self) -> ValidationResult:
        results = self._test_riemann_problems()
        metrics = self._analyze_convergence()
        self.session.save_metrics(metrics)  # ← Externalisé
        return ValidationResult(passed=True, results=results)
    
    def _test_riemann_problems(self) -> List[RiemannResult]:
        """Pur métier, zéro I/O"""
        results = []
        for case in self.config.riemann_cases:
            # Logique métier SEULE
            error, order = self._validate_single_case(case)
            results.append(RiemannResult(case, error, order))
        return results
```

**Mapping détaillé des lignes**:

| Ancien (test_7_3) | Lignes | Nouveau | Justification |
|---|---|---|---|
| Import matplotlib | 1 | `domain/section_7_3.py` L1 | Métier peut visualiser |
| `matplotlib.use('Agg')` | 2 | Removed | Infrastructure concern |
| `sys.path.append()` | 5-10 | Removed | Géré par setup.py / __init__.py |
| `from arz_model import *` | 15-30 | `domain/base.py` | Base class fournit accès |
| `class AnalyticalValidationTests` | 35 | `class AnalyticalTest` | Refactor to interface |
| `super().__init__()` | 40 | Removed | SessionManager remplace |
| `self.config = Config(...)` | 45 | Pass via `__init__(config)` | DI |
| `def test_riemann_problems()` | 80-150 | `domain/section_7_3.py` L80-150 | Métier: Copier |
| `def plot_riemann_solution()` | 160-200 | `reporting/latex_generator.py` L1-40 | Infrastructure: Extract |
| `def generate_section_content()` | 210-350 | `def run() -> ValidationResult` | Interface: Refactor |
| `logging.basicConfig()` | 15 | Removed | Infrastructure concern |
| `logger.info()` | Throughout | `self.logger.info()` | DI logger |
| `save_figure()` | 180 | `session.save_artifact()` | Infrastructure: Extract |
| `pd.DataFrame()` | 120 | `domain/section_7_3.py` | Métier: Keep |
| `np.calculations` | Throughout | `domain/section_7_3.py` | Métier: Keep |

---

### test_section_7_6_rl_performance.py → domain/section_7_6_rl_performance.py + infrastructure/*

**Ancien fichier**: ~1876 lignes (PLUS GRAND REFACTOR!)

**Breakdown par responsabilité**:

| Section | Lignes | Nouveau Fichier | Type | Action |
|---------|--------|-----------------|------|--------|
| A. RL Training Logic | 400-500 | `domain/section_7_6_rl_performance.py` | Métier | **COPIER** (pur) |
| B. Model Evaluation | 100-150 | `domain/section_7_6_rl_performance.py` | Métier | **COPIER** (pur) |
| C. Before/After Viz | 80-120 | `reporting/latex_generator.py` | Infrastructure | **EXTRACT** |
| D. Checkpoint Management | 200-300 | `infrastructure/artifact_manager.py` | Infrastructure | **EXTRACT** + **GÉNÉRALISER** |
| E. Cache System | 150-200 | `infrastructure/artifact_manager.py` | Infrastructure | **EXTRACT** + **GÉNÉRALISER** |
| F. Logging Setup | 30-50 | `infrastructure/logger.py` | Infrastructure | **EXTRACT** |
| G. Config Management | 100-150 | `infrastructure/config.py` | Infrastructure | **EXTRACT** → YAML |
| H. Hyperparameters | 50-80 | `configs/sections/section_7_6.yml` | Configuration | **EXTERNALIZE** |
| I. Session Tracking | 50-100 | `infrastructure/session.py` | Infrastructure | **EXTRACT** |
| J. Boilerplate/Comments | ~300 | Removed | Noise | **DELETE** |

**Résultat**:
- Ancien: 1876 lignes mélangées
- Nouveau domaine: ~400 lignes (pur métier)
- Nouveau infrastructure: ~200 lignes (partagé entre tests)
- Nouveau config: ~50 lignes YAML (plus maintenable)
- **Total**: ~1600 lignes (moins de boilerplate)

**Pseudo-code avant/après**:

```python
# ❌ ANCIEN (1876 lignes, tout mélangé)
class RLPerformanceValidationTest(ValidationSection):
    
    def __init__(self, quick_test=False):
        # Infrastructure
        super().__init__()
        self._setup_debug_logging()
        self.checkpoint_dir = Path(...)
        self.cache_dir = Path(...)
        
        # Configuration
        self.CODE_RL_HYPERPARAMETERS = {...}  # Hardcoded!
        self.quick_test = quick_test
        
        # Logging
        self.logger = logging.getLogger(...)
        
    def _setup_debug_logging(self):
        # 30 lignes de setup logging, répété 5 fois!
        ...
    
    def train_rl_agent(self, env):
        # Métier RL
        model = PPO('MlpPolicy', env, ...)
        model.learn(...)
        
        # Infrastructure: Saving
        self.checkpoint_path = Path(...) / f"checkpoint_{steps}.zip"
        model.save(self.checkpoint_path)
        
        # Logging
        self.logger.info(f"Checkpoint saved: {self.checkpoint_path}")
    
    def _validate_checkpoint_config(self, checkpoint_path):
        # Cache logic (100+ lignes) mélangée avec métier!
        ...
    
    def generate_before_after_visualization(self):
        # Plotting inline
        fig, axes = plt.subplots(...)
        axes[0].imshow(...)
        
        # Saving inline
        fig.savefig(self.output_dir / "before_after.png")
        
        # Logging inline
        self.logger.info(f"Figure saved: ...")

# ✅ NOUVEAU (structure séparée)
# infrastructure/config.py
@dataclass
class RLConfig:
    learning_rate: float = 1e-3
    buffer_size: int = 50000
    batch_size: int = 32
    training_episodes: int = 5000

    @staticmethod
    def from_yaml(path: str) -> "RLConfig":
        # Load from YAML
        pass

# domain/section_7_6_rl_performance.py
class RLPerformanceTest(ValidationTest):
    def __init__(
        self,
        env: ISimulator,           # ← DI: Simulator
        model_factory: IModelFactory,  # ← DI: Model
        config: RLConfig,          # ← DI: Config
        logger: ILogger = None,    # ← DI: Logger
        session: ISessionManager = None  # ← DI: Session
    ):
        self.env = env
        self.model_factory = model_factory
        self.config = config
        self.logger = logger or DummyLogger()
        self.session = session or DummySession()
    
    def run(self) -> ValidationResult:
        """Métier pur: Entraîner et évaluer"""
        model = self._train_agent()
        metrics = self._evaluate_performance(model)
        return ValidationResult(
            passed=metrics['travel_time_improvement'] > 5.0,
            metrics=metrics
        )
    
    def _train_agent(self) -> PPO:
        """Métier: Entraîner le modèle"""
        model = self.model_factory.create(self.env, self.config)
        model.learn(total_timesteps=...)
        return model
    
    def _evaluate_performance(self, model: PPO) -> Dict:
        """Métier: Évaluer les métriques"""
        baseline_results = []
        rl_results = []
        
        for scenario in self.config.test_scenarios:
            baseline = self._evaluate_baseline(scenario)
            rl = self._evaluate_rl(scenario, model)
            baseline_results.append(baseline)
            rl_results.append(rl)
        
        return self._compute_metrics(baseline_results, rl_results)

# infrastructure/artifact_manager.py
class ArtifactManager:
    def save_checkpoint(self, model: PPO, config_hash: str, steps: int):
        """Gestion centralisée des checkpoints"""
        path = self.checkpoint_dir / f"model_{config_hash}_{steps}_steps.zip"
        model.save(str(path))
        self.logger.info(f"Checkpoint saved: {path}")
        return path
    
    def load_cached_baseline(self, scenario: str) -> Dict:
        """Logique cache réutilisable"""
        cache_file = self.cache_dir / f"{scenario}_baseline_cache.pkl"
        if cache_file.exists():
            return pickle.load(open(cache_file, 'rb'))
        # Compute baseline
        ...

# reporting/latex_generator.py
class LatexGenerator:
    def generate_before_after_plot(self, baseline_traj: np.ndarray, rl_traj: np.ndarray):
        """Reporting: Visualisation"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        axes[0].imshow(baseline_traj, cmap='Reds')
        axes[1].imshow(rl_traj, cmap='Greens')
        return fig

# Utilisation finale
config = RLConfig.from_yaml("configs/sections/section_7_6.yml")
test = RLPerformanceTest(
    env=TrafficSignalEnvDirect(...),
    model_factory=PPOFactory(),
    config=config,
    logger=setup_logger("rl_test"),
    session=SessionManager("outputs/")
)
result = test.run()
```

---

## TABLE 3: UTILITAIRES

### validation_utils.py → infrastructure/* + domain/base.py

| Classe/Fonction | Ancien | Nouveau | Notes |
|---|---|---|---|
| `ValidationSection` | `validation_utils.py` | `infrastructure/session.py::SessionManager` | REFACTOR + rename |
| `ValidationTest` | Implicit | `domain/base.py::ValidationTest` | NEW: Abstract interface |
| `run_validation_test()` | `validation_utils.py` | `orchestration/runner.py` | EXTRACT: Orchestration |
| `compute_mape()` | `validation_utils.py` | Importé depuis `arz_model.metrics` | DEDUPLICATE |
| `compute_rmse()` | `validation_utils.py` | Importé depuis `arz_model.metrics` | DEDUPLICATE |
| `compute_geh()` | `validation_utils.py` | Importé depuis `arz_model.metrics` | DEDUPLICATE |
| `save_figure()` | `validation_utils.py` | `infrastructure/artifact_manager.py` | EXTRACT |
| `generate_tex_snippet()` | `validation_utils.py` | `reporting/latex_generator.py` | EXTRACT |

---

## TABLE 4: CONFIGURATION EXTERNALISÉE

### Ancien (Hardcoded) → Nouveau (YAML)

```yaml
# ❌ ANCIEN: validation_kaggle_manager.py L127-163
self.validation_sections = [
    {
        "name": "section_7_3_analytical",
        "script": "test_section_7_3_analytical.py",
        "revendications": ["R1", "R3"],
        "description": "Tests analytiques ...",
        "estimated_minutes": 45,
        "gpu_required": True
    },
    # ... 4 other sections
]

# ✅ NOUVEAU: configs/all_sections.yml
sections:
  - name: section_7_3_analytical
    revendications: ["R1", "R3"]
    description: "Tests analytiques ..."
    estimated_minutes: 45
    gpu_required: true
    config_file: "configs/sections/section_7_3.yml"
  
  - name: section_7_4_calibration
    # ... etc
```

---

## TABLE 5: WRAPPERS À SUPPRIMER

Les fichiers suivants seront DÉPRÉCIÉS (déplacés vers `archive/`):

| Ancien Fichier | Raison | Remplacement |
|---|---|---|
| `run_kaggle_validation_section_7_3.py` | Wrapper CLI spécifique | `validation_cli.py --section section_7_3_analytical` |
| `run_kaggle_validation_section_7_4.py` | Wrapper CLI spécifique | `validation_cli.py --section section_7_4_calibration` |
| `run_kaggle_validation_section_7_5.py` | Wrapper CLI spécifique | `validation_cli.py --section section_7_5_digital_twin` |
| `run_kaggle_validation_section_7_6.py` | Wrapper CLI spécifique | `validation_cli.py --section section_7_6_rl_performance` |
| `run_kaggle_validation_section_7_7.py` | Wrapper CLI spécifique | `validation_cli.py --section section_7_7_robustness` |

**Avant**:
```bash
python run_kaggle_validation_section_7_6.py --quick --scenario=traffic_light_control
```

**Après** (simplifié):
```bash
python validation_ch7/scripts/entry_points/cli.py \
  --section section_7_6_rl_performance \
  --quick-test \
  --scenario traffic_light_control
```

---

## TABLE 6: IMPORTS MAPPING

### Ancien → Nouveau

```python
# ❌ ANCIEN
from validation_ch7.scripts.validation_utils import ValidationSection, compute_mape
from validation_ch7.scripts.validation_kaggle_manager import ValidationKaggleManager

# ✅ NOUVEAU
from validation_ch7.scripts.domain.base import ValidationTest
from validation_ch7.scripts.infrastructure.session import SessionManager
from validation_ch7.scripts.infrastructure.config import ConfigManager
from validation_ch7.scripts.orchestration.validation_orchestrator import ValidationOrchestrator
from validation_ch7.scripts.entry_points.kaggle_manager import ValidationKaggleManager
from arz_model.metrics import compute_mape  # ← Deduplicated!
```

---

## TABLE 7: PHASE D'IMPLÉMENTATION

### Pour chaque phase, fichiers à créer/modifier:

#### **Phase 1: Interfaces** (0 breaking changes)
```
NEW: scripts/domain/base.py (ValidationTest interface)
NEW: scripts/orchestration/base.py (IOrchestrator interface)
NEW: scripts/infrastructure/errors.py (Custom exceptions)
```

#### **Phase 2: Infrastructure**
```
NEW: scripts/infrastructure/logger.py
NEW: scripts/infrastructure/config.py
NEW: scripts/infrastructure/artifact_manager.py
NEW: scripts/infrastructure/session.py
REFACTOR: scripts/entry_points/cli.py (from validation_cli.py)
REFACTOR: scripts/entry_points/kaggle_manager.py (from validation_kaggle_manager.py)
```

#### **Phase 3: Domain**
```
EXTRACT: scripts/domain/section_7_3_analytical.py (from test_section_7_3_analytical.py)
EXTRACT: scripts/domain/section_7_4_calibration.py (from test_section_7_4_calibration.py)
EXTRACT: scripts/domain/section_7_5_digital_twin.py (from test_section_7_5_digital_twin.py)
EXTRACT: scripts/domain/section_7_6_rl_performance.py (from test_section_7_6_rl_performance.py) ⭐
EXTRACT: scripts/domain/section_7_7_robustness.py (from test_section_7_7_robustness.py)
```

#### **Phase 4: Orchestration**
```
NEW: scripts/orchestration/validation_orchestrator.py (refactor from run_all_validation.py)
NEW: scripts/orchestration/runner.py (new: generic test runner)
```

#### **Phase 5: Reporting**
```
NEW: scripts/reporting/latex_generator.py
NEW: scripts/reporting/metrics_aggregator.py
```

#### **Phase 6: Configuration**
```
NEW: configs/base.yml
NEW: configs/quick_test.yml
NEW: configs/full_test.yml
NEW: configs/sections/section_7_3.yml
NEW: configs/sections/section_7_4.yml
NEW: configs/sections/section_7_5.yml
NEW: configs/sections/section_7_6.yml
NEW: configs/sections/section_7_7.yml
```

#### **Phase 7: Cleanup**
```
MOVE: scripts/test_section_*.py → scripts/archive/
MOVE: scripts/run_kaggle_validation_*.py → scripts/archive/
MOVE: scripts/run_all_validation.py → scripts/archive/
DEPRECATE: scripts/validation_utils.py (keep for compatibility)
UPDATE: scripts/__init__.py (new imports)
```

---

## RÉSUMÉ: STATISTIQUES DE REFACTORISATION

| Métrique | Ancien | Nouveau | Δ |
|---|---|---|---|
| Nombre de fichiers Python | 15 | 25 | +10 (mais séparation des concerns) |
| Nombre de fichiers config | 0 | 8 | +8 (nouvelles: YAML configs) |
| Lignes de code total | 8,500+ | ~4,500 | -47% (moins de duplication) |
| Lignes de code métier | N/A | ~2,000 | (isolé et testable) |
| Lignes de code infrastructure | N/A | ~1,500 | (centralisé) |
| Tests unitaires possibles | 0% | 100% | (métier testable) |
| Duplication de code | High | None | (patterns centralisés) |
| Onboarding time | 2 days | 3 hours | -87% (structure claire) |

---

**Fin du tableau de correspondance**
