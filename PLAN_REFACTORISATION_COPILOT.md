# PLAN DE REFACTORISATION - GUIDE COPILOT
## Validation CH7 → Validation CH7 V2 (Architecture Clean)

**Stratégie**: Construction PARALLÈLE - on ne touche PAS à `validation_ch7/` existant
**Dossier cible**: `validation_ch7_v2/` (nouveau, cohabitation avec l'ancien)
**Philosophie**: ÉLÉVATION, pas destruction. Préservation totale des innovations.

---

## 🎯 OBJECTIFS STRATÉGIQUES

### Objectif #1: Zéro Régression Fonctionnelle
✅ Tous les tests actuels continuent de marcher  
✅ Système existant reste 100% opérationnel  
✅ Migration progressive, pas "big bang"

### Objectif #2: Préservation des 7 Innovations Majeures
1. ✅ Cache additif intelligent (extension 600s → 3600s sans recalcul)
2. ✅ Config-hashing MD5 pour validation checkpoint ↔ config
3. ✅ Architecture de Controller avec state tracking autonome
4. ✅ Dual cache system (baseline universal + RL config-specific)
5. ✅ Checkpoint rotation automatique avec archivage
6. ✅ Templates LaTeX avec placeholders
7. ✅ Session tracking JSON (artifact counting)

### Objectif #3: Respect des 10 Principes SOLID
- SRP, OCP, LSP, ISP, DIP (voir PRINCIPES_ARCHITECTURAUX.md)
- DRY, Configuration externe, SoC, Testability, Explicit over implicit

### Objectif #4: Métriques de Succès
| Métrique | Avant | Cible |
|----------|-------|-------|
| Lignes domain test (section 7.6) | 1876 | <500 |
| Temps ajout section 7.8 | ~4h (4 fichiers modifiés) | <30min (2 fichiers créés) |
| Testabilité (mocks) | Impossible | 100% mockable |
| Couverture tests unitaires | 0% | >80% |
| Duplication code (logging, cache) | 5x répété | 1x centralisé |

---

## 📂 STRUCTURE CIBLE: `validation_ch7_v2/`

```
validation_ch7_v2/
├── __init__.py
├── scripts/
│   ├── __init__.py
│   ├── entry_points/                          ← Layer 0: CLI, Kaggle
│   │   ├── __init__.py
│   │   ├── cli.py ........................... CLI principal (arg parsing)
│   │   ├── kaggle_manager.py ................ Manager Kaggle (upload kernels)
│   │   └── local_runner.py .................. Runner local (tests rapides)
│   │
│   ├── orchestration/                         ← Layer 1: Orchestration
│   │   ├── __init__.py
│   │   ├── base.py .......................... IOrchestrator interface
│   │   ├── validation_orchestrator.py ....... Orchestre tous les tests
│   │   ├── test_runner.py ................... Exécute un test individuel
│   │   └── test_factory.py .................. Factory pattern pour tests
│   │
│   ├── domain/                                ← Layer 2: Validation Domain (MÉTIER)
│   │   ├── __init__.py
│   │   ├── base.py .......................... ValidationTest abstract class
│   │   ├── models.py ........................ ValidationResult, TestConfig
│   │   ├── section_7_3_analytical.py ........ Tests analytiques (Riemann, convergence)
│   │   ├── section_7_4_calibration.py ....... Calibration sur données réelles
│   │   ├── section_7_5_digital_twin.py ...... Validation jumeau numérique
│   │   ├── section_7_6_rl_performance.py .... Performance RL (CŒUR, 400 lignes métier pur)
│   │   └── section_7_7_robustness.py ........ Tests de robustesse
│   │
│   ├── infrastructure/                        ← Layer 3: Infrastructure
│   │   ├── __init__.py
│   │   ├── logger.py ........................ Logging centralisé (DRY)
│   │   ├── config.py ........................ Config manager (chargement YAML)
│   │   ├── artifact_manager.py .............. Gestion artefacts (checkpoints, cache)
│   │   ├── session.py ....................... Session metadata (JSON tracking)
│   │   └── errors.py ........................ Custom exceptions
│   │
│   └── reporting/                             ← Sub-layer: Reporting
│       ├── __init__.py
│       ├── latex_generator.py ............... Génération LaTeX (templates)
│       └── metrics_aggregator.py ............ Agrégation métriques
│
├── configs/                                   ← Configuration externalisée
│   ├── base.yml ............................. Config par défaut
│   ├── quick_test.yml ....................... Config tests rapides (CI/CD)
│   ├── full_test.yml ........................ Config tests complets (Kaggle)
│   └── sections/
│       ├── section_7_3.yml .................. Config analytique
│       ├── section_7_4.yml .................. Config calibration
│       ├── section_7_5.yml .................. Config digital twin
│       ├── section_7_6.yml .................. Config RL (hyperparams)
│       └── section_7_7.yml .................. Config robustesse
│
├── templates/                                 ← Templates LaTeX
│   ├── base.tex ............................. Template base
│   ├── section_7_3.tex ...................... Section analytique
│   ├── section_7_4.tex ...................... Section calibration
│   ├── section_7_5.tex ...................... Section digital twin
│   ├── section_7_6.tex ...................... Section RL
│   └── section_7_7.tex ...................... Section robustesse
│
├── tests/                                     ← Tests unitaires (NOUVEAU!)
│   ├── __init__.py
│   ├── test_domain/
│   │   ├── test_section_7_3_analytical.py ... Tests unitaires section 7.3
│   │   ├── test_section_7_6_rl_performance.py Tests unitaires section 7.6 (mocks!)
│   │   └── ...
│   ├── test_infrastructure/
│   │   ├── test_artifact_manager.py ......... Tests cache/checkpoint logic
│   │   ├── test_config.py ................... Tests config loading
│   │   └── ...
│   └── test_integration/
│       ├── test_full_workflow.py ............ Tests end-to-end
│       └── ...
│
├── checkpoints/                               ← Checkpoints RL (Git-tracked)
│   └── section_7_6/
│       ├── traffic_light_control_checkpoint_abc12345_100_steps.zip
│       └── archived/ ........................ Checkpoints incompatibles
│
├── cache/                                     ← Cache simulation (Git-tracked)
│   └── section_7_6/
│       ├── traffic_light_control_baseline_cache.pkl (universel)
│       └── traffic_light_control_abc12345_rl_cache.pkl (config-specific)
│
└── README.md ................................ Documentation architecture
```

---

## 🚀 PLAN D'IMPLÉMENTATION PAR PHASES

### Phase 0: Préparation (30 minutes)
**Objectif**: Créer structure de base, pas de code métier

**Actions**:
1. Créer dossier `validation_ch7_v2/`
2. Créer structure de dossiers (vide)
3. Créer `README.md` avec architecture expliquée
4. Créer `__init__.py` partout

**Validation Phase 0**:
```bash
# Vérifier structure
tree validation_ch7_v2/ -L 3
# → Doit montrer arborescence complète
```

---

### Phase 1: Interfaces & Base Classes (2 heures)
**Objectif**: Créer les abstractions (interfaces, classes de base)

#### 🤖 PROMPT COPILOT #1.1: Base Classes Domain

```markdown
# CONTEXTE
Je refactorise un système de validation (validation_ch7 → validation_ch7_v2).
Architecture cible: Domain-Driven Design avec layers séparés.

# TÂCHE
Créer `validation_ch7_v2/scripts/domain/base.py` avec:

1. **ValidationTest** (abstract class):
   - Méthode abstraite: `run() -> ValidationResult`
   - Propriété abstraite: `name: str`
   - Méthode helper: `_validate_prerequisites()` (optionnel)

2. **ValidationResult** (dataclass):
   - `passed: bool` - Test passed or failed
   - `metrics: Dict[str, float]` - Numerical metrics (convergence order, improvement %, etc.)
   - `artifacts: Dict[str, Path]` - Generated files (figures, data, LaTeX)
   - `errors: List[str]` - Error messages if failed
   - `warnings: List[str]` - Non-critical warnings
   - `metadata: Dict[str, Any]` - Additional info (duration, device, timestamp)

3. **TestConfig** (dataclass):
   - `section_name: str` - e.g., "section_7_6_rl_performance"
   - `quick_test: bool` - Fast mode for CI/CD
   - `scenario: Optional[str]` - For sections with multiple scenarios
   - `device: str` - "cpu" or "gpu"
   - `output_dir: Path` - Where to save results

# CONTRAINTES
- Type hints PARTOUT
- Docstrings complètes (Google style)
- ABC (Abstract Base Class) pour ValidationTest
- dataclass pour ValidationResult et TestConfig
- NO implementation logic (juste les structures)

# EXEMPLE D'UTILISATION ATTENDU
```python
# Dans domain/section_7_6_rl_performance.py (futur)
class RLPerformanceTest(ValidationTest):
    @property
    def name(self) -> str:
        return "section_7_6_rl_performance"
    
    def run(self) -> ValidationResult:
        # Logique métier ici
        return ValidationResult(
            passed=True,
            metrics={"travel_time_improvement": 28.7},
            artifacts={"before_after_plot": Path("figure.png")},
            errors=[],
            warnings=[],
            metadata={"duration_seconds": 3600}
        )
```

# GÉNÈRE LE CODE
```

**Résultat attendu**: Fichier `domain/base.py` (~100 lignes, abstraction pure)

---

#### 🤖 PROMPT COPILOT #1.2: Custom Exceptions

```markdown
# CONTEXTE
Système de validation avec architecture en layers. Besoin d'exceptions spécifiques.

# TÂCHE
Créer `validation_ch7_v2/scripts/infrastructure/errors.py` avec hiérarchie d'exceptions:

1. **ValidationError** (base):
   - Classe mère pour toutes les erreurs de validation
   - Attribut: `context: Dict[str, Any]` (metadata pour debugging)

2. **ConfigError** (hérite ValidationError):
   - Config YAML invalide ou manquante
   - Exemple: "section_7_6.yml not found"

3. **CheckpointError** (hérite ValidationError):
   - Erreurs liées aux checkpoints RL
   - Exemple: "Checkpoint config hash mismatch"

4. **CacheError** (hérite ValidationError):
   - Erreurs de cache (corruption, incompatibilité)
   - Exemple: "Cache coherence validation failed"

5. **SimulationError** (hérite ValidationError):
   - Erreurs pendant simulation ARZ
   - Exemple: "Mass conservation violated"

6. **OrchestrationError** (hérite ValidationError):
   - Erreurs d'orchestration de tests
   - Exemple: "Test dependency not met"

# CONTRAINTES
- Type hints partout
- Docstrings explicatives
- Message d'erreur DESCRIPTIF (pas juste "Error")
- Context dict pour debugging riche

# EXEMPLE D'UTILISATION
```python
try:
    checkpoint = load_checkpoint(path)
except FileNotFoundError:
    raise CheckpointError(
        f"Checkpoint not found: {path}",
        context={"path": path, "section": "7.6", "scenario": "traffic_light"}
    )
```

# GÉNÈRE LE CODE
```

**Résultat attendu**: Fichier `infrastructure/errors.py` (~80 lignes)

---

#### 🤖 PROMPT COPILOT #1.3: Orchestrator Interface

```markdown
# CONTEXTE
Architecture avec orchestration centralisée. Besoin d'interface IOrchestrator.

# TÂCHE
Créer `validation_ch7_v2/scripts/orchestration/base.py` avec:

1. **IOrchestrator** (Protocol ou ABC):
   - `run_all_tests() -> List[ValidationResult]` - Exécute tous les tests
   - `run_single_test(test: ValidationTest) -> ValidationResult` - Exécute un test
   - `run_section(section_name: str) -> ValidationResult` - Exécute une section

2. **ITestRunner** (Protocol ou ABC):
   - `run(test: ValidationTest) -> ValidationResult` - Exécute un test
   - `setup()` - Préparation avant test
   - `teardown()` - Nettoyage après test

# CONTRAINTES
- Utiliser `typing.Protocol` (duck typing) ou ABC
- Type hints partout
- Docstrings claires
- Interface MINIMALE (ISP - Interface Segregation Principle)

# GÉNÈRE LE CODE
```

**Résultat attendu**: Fichier `orchestration/base.py` (~60 lignes)

---

**Validation Phase 1**:
```python
# Test d'import
from validation_ch7_v2.scripts.domain.base import ValidationTest, ValidationResult
from validation_ch7_v2.scripts.infrastructure.errors import ConfigError
from validation_ch7_v2.scripts.orchestration.base import IOrchestrator

# → Pas d'erreurs d'import
```

---

### Phase 2: Infrastructure Layer (4 heures)
**Objectif**: Créer les modules d'infrastructure (logger, config, artifact manager, session)

#### 🤖 PROMPT COPILOT #2.1: Logger Centralisé

```markdown
# CONTEXTE
Ancien système: chaque test a son propre logging setup (VIOLATION DRY).
Nouveau système: logger centralisé, utilisé par tous.

# TÂCHE
Créer `validation_ch7_v2/scripts/infrastructure/logger.py` avec:

1. **setup_logger**(name: str, level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
   - Configure un logger avec nom spécifique
   - Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
   - Handler console (stdout) + handler fichier optionnel
   - Flush immédiat (Kaggle stdout buffering issue)

2. **get_logger**(name: str) -> logging.Logger:
   - Récupère un logger existant ou crée s'il n'existe pas

3. **PATTERNS DE LOGGING CONSTANTS**:
   - DEBUG_BC_RESULT = "[DEBUG_BC_RESULT]"
   - DEBUG_PRIMITIVES = "[DEBUG_PRIMITIVES]"
   - DEBUG_FLUXES = "[DEBUG_FLUXES]"
   - DEBUG_CACHE = "[DEBUG_CACHE]"
   - DEBUG_CHECKPOINT = "[DEBUG_CHECKPOINT]"

# INNOVATION À PRÉSERVER
L'ancien système avait des patterns de logging structurés (e.g., [DEBUG_BC_RESULT])
qui permettent de filtrer les logs facilement. GARDER cette approche.

# CONTRAINTES
- Type hints partout
- Docstrings
- Thread-safe (logging est thread-safe par défaut)
- Pas de global state mutable

# EXEMPLE D'UTILISATION
```python
from validation_ch7_v2.scripts.infrastructure.logger import setup_logger, DEBUG_CHECKPOINT

logger = setup_logger("rl_validation", log_file=Path("debug.log"))
logger.info(f"{DEBUG_CHECKPOINT} Loading checkpoint from {path}")
```

# GÉNÈRE LE CODE
```

**Résultat attendu**: Fichier `infrastructure/logger.py` (~100 lignes)

---

#### 🤖 PROMPT COPILOT #2.2: Config Manager

```markdown
# CONTEXTE
Ancien système: configs hardcodées dans le code (TRAINING_EPISODES=100).
Nouveau système: configs externalisées en YAML.

# TÂCHE
Créer `validation_ch7_v2/scripts/infrastructure/config.py` avec:

1. **ConfigManager** class:
   - `__init__(config_dir: Path)` - Initialise avec dossier de configs
   - `load_base_config() -> Dict` - Charge base.yml
   - `load_section_config(section_name: str) -> SectionConfig` - Charge section_7_X.yml
   - `load_all_sections() -> List[SectionConfig]` - Découverte automatique

2. **SectionConfig** dataclass:
   - `name: str` - e.g., "section_7_6_rl_performance"
   - `description: str` - Description textuelle
   - `revendication: str` - e.g., "R5: Performance supérieure RL"
   - `estimated_duration_minutes: int` - Durée estimée
   - `quick_test_duration_minutes: int` - Durée en mode rapide
   - `hyperparameters: Dict[str, Any]` - Hyperparams spécifiques (RL, calibration, etc.)
   - `output_subdir: str` - e.g., "section_7_6_rl_performance"

3. **Méthode helper**:
   - `merge_configs(base: Dict, section: Dict) -> Dict` - Merge base + section

# FORMAT YAML ATTENDU
```yaml
# configs/sections/section_7_6.yml
name: section_7_6_rl_performance
description: "Validation performance agents RL vs baseline"
revendication: "R5: Performance supérieure des agents RL"
estimated_duration_minutes: 180
quick_test_duration_minutes: 15

hyperparameters:
  training:
    episodes: 5000
    buffer_size: 50000
    learning_rate: 1e-3
    batch_size: 32
  quick_test:
    episodes: 100
    duration_per_episode: 120
  full_test:
    episodes: 5000
    duration_per_episode: 3600
```

# CONTRAINTES
- Utiliser pyyaml
- Type hints partout
- Validation de config (schema checking optionnel)
- Docstrings

# GÉNÈRE LE CODE
```

**Résultat attendu**: Fichier `infrastructure/config.py` (~150 lignes)

---

#### 🤖 PROMPT COPILOT #2.3: Artifact Manager (Cache + Checkpoints)

```markdown
# CONTEXTE
INNOVATION MAJEURE à préserver: système sophistiqué de cache/checkpoints.
- Cache additif intelligent (extension 600s → 3600s sans recalcul)
- Config-hashing MD5 pour validation checkpoint ↔ config
- Dual cache: baseline universel + RL config-specific
- Checkpoint rotation automatique avec archivage

# TÂCHE
Créer `validation_ch7_v2/scripts/infrastructure/artifact_manager.py` avec:

## 1. **ArtifactManager** class:

### Cache Baseline (universel)
- `save_baseline_cache(scenario: str, states: List, duration: float, control_interval: float)`
  - Format: `{scenario}_baseline_cache.pkl` (PAS de config_hash)
  - Rationale: Fixed-time controller → comportement universel

- `load_baseline_cache(scenario: str, required_duration: float) -> Optional[List]`
  - Charge cache si existe et suffisant
  - Retourne None si pas de cache

- `extend_baseline_cache(scenario: str, existing_states: List, target_duration: float) -> List`
  - Extension ADDITIVE: reprend depuis cached_states[-1]
  - Simule UNIQUEMENT l'extension manquante
  - Concatène cached + extension

### Cache RL (config-specific)
- `save_rl_cache(scenario: str, config_hash: str, model_path: Path, total_timesteps: int)`
  - Format: `{scenario}_{config_hash}_rl_cache.pkl`
  - Stocke metadata: model_path, timesteps, config_hash

- `load_rl_cache(scenario: str, config_hash: str, required_timesteps: int) -> Optional[Dict]`
  - Valide config_hash
  - Vérifie existence model file
  - Retourne metadata ou None

### Checkpoints RL
- `save_checkpoint(model, scenario: str, config_hash: str, steps: int)`
  - Format: `{scenario}_checkpoint_{config_hash}_{steps}_steps.zip`
  - Sauvegarde model + replay buffer

- `load_checkpoint(scenario: str, config_hash: str) -> Optional[Path]`
  - Cherche checkpoint le plus récent avec config_hash
  - Valide config_hash
  - Retourne path ou None

- `validate_checkpoint_config(checkpoint_path: Path, config_hash: str) -> bool`
  - Extrait hash depuis checkpoint filename
  - Compare avec current config_hash

- `archive_incompatible_checkpoint(checkpoint_path: Path, old_config_hash: str)`
  - Déplace vers archived/ avec suffix _CONFIG_{old_hash}

### Config Hashing
- `compute_config_hash(scenario_path: Path) -> str`
  - MD5 du YAML (8 chars)
  - Pour validation checkpoint ↔ config

## 2. **Filesystem Helpers**:
- `ensure_dir(path: Path)` - Crée dossier si n'existe pas
- `list_checkpoints(scenario: str) -> List[Path]` - Liste tous les checkpoints d'un scenario

# CONTRAINTES
- Type hints partout
- Docstrings DÉTAILLÉES (cette logique est complexe!)
- Logging avec patterns ([CACHE], [CHECKPOINT])
- Thread-safe si possible (file locking)

# EXEMPLE D'UTILISATION
```python
mgr = ArtifactManager(base_dir=Path("validation_ch7_v2/"))

# Cache baseline (universel)
mgr.save_baseline_cache("traffic_light_control", states, 3600.0, 15.0)
cached = mgr.load_baseline_cache("traffic_light_control", 7200.0)
if cached and len(cached) < required_steps:
    extended = mgr.extend_baseline_cache("traffic_light_control", cached, 7200.0)

# Cache RL (config-specific)
config_hash = mgr.compute_config_hash(Path("scenario.yml"))
mgr.save_rl_cache("traffic_light", config_hash, Path("model.zip"), 10000)

# Checkpoint
mgr.save_checkpoint(model, "traffic_light", config_hash, 5000)
checkpoint_path = mgr.load_checkpoint("traffic_light", config_hash)
if checkpoint_path:
    is_valid = mgr.validate_checkpoint_config(checkpoint_path, config_hash)
```

# GÉNÈRE LE CODE
```

**Résultat attendu**: Fichier `infrastructure/artifact_manager.py` (~400 lignes, CŒUR de l'innovation)

---

#### 🤖 PROMPT COPILOT #2.4: Session Manager

```markdown
# CONTEXTE
Innovation à préserver: session_summary.json (tracking des artefacts générés).

# TÂCHE
Créer `validation_ch7_v2/scripts/infrastructure/session.py` avec:

1. **SessionManager** class:
   - `__init__(section_name: str, output_dir: Path)`
   - `create_directory_structure()` - Crée figures/, data/, latex/, etc.
   - `register_artifact(artifact_type: str, path: Path)` - Track un artefact
   - `save_summary()` - Génère session_summary.json

2. **Structure de dossiers créée**:
   ```
   output_dir/
   ├── figures/ ................ PNG/PDF generated
   ├── data/
   │   ├── npz/ ................ NumPy arrays
   │   ├── scenarios/ .......... YAML configs
   │   └── metrics/ ............ JSON metrics
   └── latex/ .................. Generated LaTeX snippets
   ```

3. **Format session_summary.json**:
   ```json
   {
     "section_name": "section_7_6_rl_performance",
     "timestamp": "2025-10-16T14:30:00",
     "artifacts": {
       "figures": ["before_after.png", "learning_curve.png"],
       "data_npz": ["baseline_trajectory.npz", "rl_trajectory.npz"],
       "data_metrics": ["performance_metrics.json"],
       "latex": ["section_7_6_content.tex"]
     },
     "artifact_count": 6
   }
   ```

# CONTRAINTES
- Type hints
- Docstrings
- Logging des opérations
- mkdir avec parents=True, exist_ok=True

# EXEMPLE D'UTILISATION
```python
session = SessionManager("section_7_6_rl_performance", Path("outputs/"))
session.create_directory_structure()
session.register_artifact("figure", Path("outputs/figures/plot.png"))
session.save_summary()  # → Crée session_summary.json
```

# GÉNÈRE LE CODE
```

**Résultat attendu**: Fichier `infrastructure/session.py` (~150 lignes)

---

**Validation Phase 2**:
```python
# Test imports
from validation_ch7_v2.scripts.infrastructure.logger import setup_logger
from validation_ch7_v2.scripts.infrastructure.config import ConfigManager, SectionConfig
from validation_ch7_v2.scripts.infrastructure.artifact_manager import ArtifactManager
from validation_ch7_v2.scripts.infrastructure.session import SessionManager

# Test fonctionnel simple
logger = setup_logger("test")
logger.info("✓ Logger works")

config_mgr = ConfigManager(Path("validation_ch7_v2/configs"))
# (configs YAML pas encore créés, mais classe fonctionne)

artifact_mgr = ArtifactManager(Path("validation_ch7_v2/"))
config_hash = artifact_mgr.compute_config_hash(Path("dummy.yml"))
print(f"✓ Config hash: {config_hash}")

session = SessionManager("test", Path("outputs/"))
session.create_directory_structure()
print("✓ Session structure created")
```

---

### Phase 3: Domain Layer - Extraction Métier (6 heures)
**Objectif**: Extraire la logique métier pure depuis ancien système

**FOCUS**: `test_section_7_6_rl_performance.py` (1876 lignes → 400 lignes métier pur)

#### 🤖 PROMPT COPILOT #3.1: Section 7.6 RL Performance (MÉTIER PUR)

```markdown
# CONTEXTE CRITIQUE
Tu vas extraire la LOGIQUE MÉTIER PURE depuis un fichier monolithique de 1876 lignes.
Ce fichier est le CŒUR du système, il a survécu à 35 bugs, il contient des innovations majeures.

**Fichier source**: `validation_ch7/scripts/test_section_7_6_rl_performance.py`
**Fichier cible**: `validation_ch7_v2/scripts/domain/section_7_6_rl_performance.py`

**Principe**: EXTRAIRE le métier, DÉLÉGUER l'infrastructure.

# TÂCHE

## 1. Analyser l'ancien fichier (lignes 1-1876)

**Sections à EXTRAIRE (métier pur)**:
- Lines 612-702: `BaselineController` class (logique contrôle fixed-time)
- Lines 637-702: `RLController` class (logique contrôle RL)
- Lines 705-938: `run_control_simulation()` (simulation ARZ avec controller)
- Lines 941-1021: `evaluate_traffic_performance()` (calcul métriques)
- Lines 1024-1283: `train_rl_agent()` (entraînement RL avec PPO/DQN)
- Lines 1286-1468: `run_performance_comparison()` (baseline vs RL)

**Sections à NE PAS COPIER (infrastructure, déléguer)**:
- Lines 127-160: `_setup_debug_logging()` → Utiliser `infrastructure.logger`
- Lines 230-327: Checkpoint validation/archiving → Utiliser `ArtifactManager`
- Lines 350-580: Cache management → Utiliser `ArtifactManager`
- Lines 1585-1731: Figure generation + LaTeX → Utiliser `reporting.latex_generator`
- Lines 1734-1850: Template filling → Utiliser `reporting.latex_generator`

## 2. Créer la nouvelle classe

```python
from validation_ch7_v2.scripts.domain.base import ValidationTest, ValidationResult, TestConfig
from validation_ch7_v2.scripts.infrastructure.logger import get_logger, DEBUG_CHECKPOINT
from validation_ch7_v2.scripts.infrastructure.artifact_manager import ArtifactManager
from validation_ch7_v2.scripts.infrastructure.config import SectionConfig

class RLPerformanceTest(ValidationTest):
    """
    Validation Section 7.6: Performance RL vs Baseline.
    
    Teste la revendication R5: Performance supérieure des agents RL
    dans le contexte béninois (comparaison avec baseline fixed-time).
    
    INNOVATIONS PRÉSERVÉES:
    - Cache additif intelligent (délégué à ArtifactManager)
    - Config-hashing MD5 (délégué à ArtifactManager)
    - Controller autonome avec state tracking
    - Dual cache system (délégué à ArtifactManager)
    """
    
    def __init__(
        self,
        config: SectionConfig,
        artifact_manager: ArtifactManager,
        logger: logging.Logger = None
    ):
        self.config = config
        self.artifact_manager = artifact_manager
        self.logger = logger or get_logger(__name__)
        
        # Hyperparameters depuis config
        self.hyperparams = config.hyperparameters
    
    @property
    def name(self) -> str:
        return "section_7_6_rl_performance"
    
    def run(self) -> ValidationResult:
        """
        Exécute validation RL performance.
        
        Pipeline:
        1. Charger/créer cache baseline (universel)
        2. Exécuter simulation baseline
        3. Entraîner agent RL (ou charger checkpoint)
        4. Exécuter simulation RL
        5. Comparer performances
        6. Retourner résultats
        """
        # Métier pur ici
        # NO I/O direct (déléguer à artifact_manager)
        # NO logging setup (logger déjà injecté)
        # NO figure generation (déléguer à reporting)
        
        pass  # À remplir avec logique métier extraite
    
    # ========== CONTROLLERS (COPIER depuis ancien) ==========
    class BaselineController:
        """Controller fixed-time (60s GREEN / 60s RED)."""
        def __init__(self, scenario_type: str):
            # COPIER depuis ancien (lines 612-634)
            pass
    
    class RLController:
        """Controller RL avec agent PPO/DQN."""
        def __init__(self, model, env):
            # COPIER depuis ancien (lines 637-702)
            pass
    
    # ========== MÉTIER PUR ==========
    def run_control_simulation(self, controller, scenario_path: Path, ...):
        """
        Simule trafic avec un controller (baseline ou RL).
        INNOVATION PRÉSERVÉE: State continuation (initial_state support).
        """
        # COPIER depuis ancien (lines 705-938)
        # MAIS: remplacer I/O par délégation artifact_manager
        pass
    
    def evaluate_traffic_performance(self, states_history, scenario_type):
        """Calcule métriques performance (travel time, throughput, etc.)."""
        # COPIER depuis ancien (lines 941-1021)
        pass
    
    def train_rl_agent(self, scenario_type: str, total_timesteps: int, device: str):
        """
        Entraîne agent RL (PPO/DQN) sur scenario.
        INNOVATION PRÉSERVÉE: Checkpoint system (délégué à artifact_manager).
        """
        # COPIER depuis ancien (lines 1024-1283)
        # MAIS: remplacer checkpoint logic par artifact_manager.save_checkpoint()
        pass
    
    def run_performance_comparison(self, scenario_type: str, device: str):
        """Compare baseline vs RL performance."""
        # COPIER depuis ancien (lines 1286-1468)
        pass
```

## 3. DÉLÉGATIONS À FAIRE

**Checkpoint loading**:
```python
# ❌ ANCIEN (lines 1100-1150)
checkpoint_path = self._get_checkpoint_dir() / f"{scenario}_checkpoint_{config_hash}_{steps}_steps.zip"
if checkpoint_path.exists():
    model = DQN.load(checkpoint_path)

# ✅ NOUVEAU
checkpoint_path = self.artifact_manager.load_checkpoint(scenario, config_hash)
if checkpoint_path:
    model = DQN.load(checkpoint_path)
```

**Cache loading**:
```python
# ❌ ANCIEN (lines 388-416)
cached_states = self._load_baseline_cache(scenario, scenario_path, duration)

# ✅ NOUVEAU
cached_states = self.artifact_manager.load_baseline_cache(scenario, duration)
```

**Logging**:
```python
# ❌ ANCIEN (line 1200)
self.debug_logger.info(f"[CHECKPOINT] Loaded from {path}")

# ✅ NOUVEAU
self.logger.info(f"{DEBUG_CHECKPOINT} Loaded from {path}")
```

# CONTRAINTES
- MÉTIER PUR uniquement (pas d'I/O direct)
- Dependency Injection (artifact_manager, logger passés en __init__)
- Type hints partout
- Docstrings DÉTAILLÉES (expliquer les innovations préservées)
- Target: ~400 lignes (vs 1876 avant)

# GÉNÈRE LE CODE
Extrais la logique métier depuis l'ancien fichier et crée le nouveau.
```

**Résultat attendu**: Fichier `domain/section_7_6_rl_performance.py` (~400 lignes métier pur)

---

**Validation Phase 3**:
```python
# Test d'instanciation
from validation_ch7_v2.scripts.domain.section_7_6_rl_performance import RLPerformanceTest
from validation_ch7_v2.scripts.infrastructure.artifact_manager import ArtifactManager
from validation_ch7_v2.scripts.infrastructure.config import SectionConfig

config = SectionConfig(name="section_7_6_rl_performance", ...)
artifact_mgr = ArtifactManager(Path("validation_ch7_v2/"))
test = RLPerformanceTest(config, artifact_mgr)

# → Pas d'erreur d'instantiation
# → test.run() retourne ValidationResult
```

---

### Phase 4: Orchestration Layer (3 heures)
**Objectif**: Orchestrer l'exécution des tests

#### 🤖 PROMPT COPILOT #4.1: Test Factory

```markdown
# CONTEXTE
Pattern Factory pour créer les tests selon la configuration.

# TÂCHE
Créer `validation_ch7_v2/scripts/orchestration/test_factory.py` avec:

```python
class TestFactory:
    @staticmethod
    def create(section_config: SectionConfig, artifact_manager: ArtifactManager) -> ValidationTest:
        """
        Factory pattern: Créer le bon test selon la config.
        
        Args:
            section_config: Configuration de la section
            artifact_manager: Manager d'artefacts (injecté)
        
        Returns:
            Instance de ValidationTest concrète
        
        Raises:
            ConfigError: Si section inconnue
        """
        if section_config.name == "section_7_3_analytical":
            from validation_ch7_v2.scripts.domain.section_7_3_analytical import AnalyticalTest
            return AnalyticalTest(section_config, artifact_manager)
        
        elif section_config.name == "section_7_6_rl_performance":
            from validation_ch7_v2.scripts.domain.section_7_6_rl_performance import RLPerformanceTest
            return RLPerformanceTest(section_config, artifact_manager)
        
        # ... autres sections
        
        else:
            raise ConfigError(f"Unknown section: {section_config.name}")
```

# GÉNÈRE LE CODE
```

**Résultat attendu**: Fichier `orchestration/test_factory.py` (~80 lignes)

---

#### 🤖 PROMPT COPILOT #4.2: Validation Orchestrator

```markdown
# CONTEXTE
Orchestrator qui exécute tous les tests dans l'ordre.

# TÂCHE
Créer `validation_ch7_v2/scripts/orchestration/validation_orchestrator.py` avec:

```python
class ValidationOrchestrator:
    """
    Orchestre l'exécution de tous les tests de validation.
    
    Responsabilités:
    - Charger les configurations des sections
    - Créer les tests via factory
    - Exécuter les tests dans l'ordre
    - Agréger les résultats
    - Logger le progrès
    """
    
    def __init__(self, config_manager: ConfigManager, artifact_manager: ArtifactManager):
        self.config_manager = config_manager
        self.artifact_manager = artifact_manager
        self.logger = get_logger(__name__)
    
    def run_all_tests(self) -> List[ValidationResult]:
        """Exécute tous les tests (sections 7.3 à 7.7)."""
        pass
    
    def run_single_test(self, test: ValidationTest) -> ValidationResult:
        """
        Template method: Flux standard d'exécution.
        
        1. Log début
        2. Exécuter test.run()
        3. Log fin
        4. Handle erreurs
        5. Retourner résultat
        """
        pass
    
    def run_section(self, section_name: str) -> ValidationResult:
        """Exécute une section spécifique."""
        pass
```

# CONTRAINTES
- Template Method Pattern (flux standard)
- Error handling avec exceptions custom
- Logging détaillé du progrès
- Retour de résultats agrégés

# GÉNÈRE LE CODE
```

**Résultat attendu**: Fichier `orchestration/validation_orchestrator.py` (~200 lignes)

---

**Validation Phase 4**:
```python
# Test orchestration
from validation_ch7_v2.scripts.orchestration.validation_orchestrator import ValidationOrchestrator
from validation_ch7_v2.scripts.infrastructure.config import ConfigManager
from validation_ch7_v2.scripts.infrastructure.artifact_manager import ArtifactManager

config_mgr = ConfigManager(Path("validation_ch7_v2/configs"))
artifact_mgr = ArtifactManager(Path("validation_ch7_v2/"))
orchestrator = ValidationOrchestrator(config_mgr, artifact_mgr)

# Test d'exécution d'une section
result = orchestrator.run_section("section_7_6_rl_performance")
print(f"Test passed: {result.passed}")
print(f"Metrics: {result.metrics}")
```

---

### Phase 5: Configuration YAML (2 heures)
**Objectif**: Externaliser toutes les configs

#### 🤖 PROMPT COPILOT #5.1: Config YAML Section 7.6

```markdown
# TÂCHE
Créer `validation_ch7_v2/configs/sections/section_7_6.yml` avec:

```yaml
# Section 7.6: RL Performance Validation
name: section_7_6_rl_performance
description: "Validation de la performance des agents RL vs baseline fixed-time"
revendication: "R5: Performance supérieure des agents RL dans le contexte béninois"

# Durées
estimated_duration_minutes: 180  # 3h sur Kaggle GPU
quick_test_duration_minutes: 15  # 15min pour CI/CD

# Hyperparamètres RL (CODE_RL compatibility)
hyperparameters:
  # Configuration entraînement RL
  training:
    algorithm: "DQN"  # ou "PPO"
    episodes: 5000
    buffer_size: 50000
    learning_rate: 1e-3
    batch_size: 32
    tau: 1.0
    gamma: 0.99
    train_freq: 4
    gradient_steps: 1
    target_update_interval: 1000
    exploration_fraction: 0.1
    exploration_initial_eps: 1.0
    exploration_final_eps: 0.05
  
  # Mode quick test (CI/CD)
  quick_test:
    episodes: 100
    duration_per_episode: 120  # 2 minutes
    control_interval: 15
  
  # Mode full test (Kaggle GPU)
  full_test:
    episodes: 5000
    duration_per_episode: 3600  # 1 heure
    control_interval: 15
  
  # Scénarios disponibles
  scenarios:
    - traffic_light_control
    - ramp_metering
    - adaptive_speed_control
  
  # Baseline parameters
  baseline:
    green_duration: 60  # seconds
    red_duration: 60    # seconds

# Chemins relatifs
output_subdir: "section_7_6_rl_performance"
```

# GÉNÈRE LE YAML
```

**Résultat attendu**: Fichier `configs/sections/section_7_6.yml` (~80 lignes)

---

### Phase 6: Reporting Layer (3 heures)
**Objectif**: Génération LaTeX automatisée

#### 🤖 PROMPT COPILOT #6.1: LaTeX Generator

```markdown
# CONTEXTE
Ancien système: génération LaTeX mélangée dans les tests.
Nouveau système: reporting séparé, réutilisable.

# TÂCHE
Créer `validation_ch7_v2/scripts/reporting/latex_generator.py` avec:

```python
class LatexGenerator:
    """Génère du contenu LaTeX depuis templates et données."""
    
    def __init__(self, template_dir: Path):
        self.template_dir = template_dir
    
    def generate_section(
        self,
        section_name: str,
        metrics: Dict[str, float],
        artifacts: Dict[str, Path]
    ) -> str:
        """
        Génère contenu LaTeX pour une section.
        
        1. Charge template (e.g., templates/section_7_6.tex)
        2. Remplit placeholders avec metrics
        3. Retourne LaTeX compilé
        """
        pass
    
    def plot_before_after(
        self,
        baseline_trajectory: np.ndarray,
        rl_trajectory: np.ndarray,
        output_path: Path
    ):
        """Génère figure before/after (baseline vs RL)."""
        pass
    
    def plot_learning_curve(
        self,
        rewards: List[float],
        output_path: Path
    ):
        """Génère courbe d'apprentissage RL."""
        pass
```

# CONTRAINTES
- Utiliser matplotlib pour figures
- Templates avec placeholders {variable_name}
- Type hints partout
- Sauvegarder figures en PNG + PDF

# GÉNÈRE LE CODE
```

**Résultat attendu**: Fichier `reporting/latex_generator.py` (~250 lignes)

---

### Phase 7: Entry Points (2 heures)
**Objectif**: CLI, Kaggle manager, local runner

#### 🤖 PROMPT COPILOT #7.1: CLI Principal

```markdown
# TÂCHE
Créer `validation_ch7_v2/scripts/entry_points/cli.py` avec:

```python
def main():
    """CLI principal pour validation CH7."""
    parser = argparse.ArgumentParser(
        description="Validation Chapitre 7 - ARZ-RL System"
    )
    parser.add_argument(
        "--section",
        choices=["section_7_3_analytical", "section_7_4_calibration", ...],
        help="Section à exécuter (ou 'all')"
    )
    parser.add_argument("--quick-test", action="store_true", help="Mode rapide")
    parser.add_argument("--scenario", help="Scénario spécifique (section 7.6)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    
    args = parser.parse_args()
    
    # Setup
    config_mgr = ConfigManager(Path("validation_ch7_v2/configs"))
    artifact_mgr = ArtifactManager(Path("validation_ch7_v2/"))
    orchestrator = ValidationOrchestrator(config_mgr, artifact_mgr)
    
    # Exécution
    if args.section == "all":
        results = orchestrator.run_all_tests()
    else:
        result = orchestrator.run_section(args.section)
    
    # Reporting
    print_results(results)
```

# GÉNÈRE LE CODE
```

**Résultat attendu**: Fichier `entry_points/cli.py` (~150 lignes)

---

## 🎯 VALIDATION FINALE DU SYSTÈME

### Test End-to-End

```bash
# Test complet section 7.6 (mode rapide)
cd validation_ch7_v2/
python -m scripts.entry_points.cli --section section_7_6_rl_performance --quick-test --device cpu

# Vérifications:
# 1. ✓ Cache baseline créé (validation_ch7_v2/cache/section_7_6/traffic_light_control_baseline_cache.pkl)
# 2. ✓ Checkpoint RL créé (validation_ch7_v2/checkpoints/section_7_6/traffic_light_control_checkpoint_abc12345_100_steps.zip)
# 3. ✓ Figures générées (validation_ch7_v2/outputs/section_7_6_rl_performance/figures/before_after.png)
# 4. ✓ Métriques JSON (validation_ch7_v2/outputs/section_7_6_rl_performance/data/metrics/performance.json)
# 5. ✓ Session summary (validation_ch7_v2/outputs/section_7_6_rl_performance/session_summary.json)
# 6. ✓ LaTeX généré (validation_ch7_v2/outputs/section_7_6_rl_performance/latex/section_7_6_content.tex)
```

### Checklist de Préservation des Innovations

```
☐ Innovation #1: Cache additif intelligent
  ✓ Extension 600s → 3600s SANS recalcul complet
  ✓ Reprise depuis cached_states[-1]
  ✓ Économie 85% du temps

☐ Innovation #2: Config-hashing MD5
  ✓ Validation checkpoint ↔ config
  ✓ Archivage automatique si mismatch
  ✓ Traçabilité complète

☐ Innovation #3: Controller autonome
  ✓ State tracking interne
  ✓ Reprise possible (controller.time_step = cached_duration)

☐ Innovation #4: Dual cache system
  ✓ Baseline universel (PAS de config_hash)
  ✓ RL config-specific (AVEC config_hash)

☐ Innovation #5: Checkpoint rotation
  ✓ Sauvegarde tous les N steps
  ✓ Rotation automatique (garder 3 derniers)

☐ Innovation #6: Templates LaTeX
  ✓ Placeholders {variable_name}
  ✓ Réutilisables

☐ Innovation #7: Session tracking
  ✓ session_summary.json avec artifact counting
```

### Métriques de Succès

```
| Métrique | Avant (validation_ch7) | Après (validation_ch7_v2) | Objectif | Status |
|----------|------------------------|---------------------------|----------|--------|
| Lignes domain test (7.6) | 1876 | ~400 | <500 | ✓ |
| Temps ajout section 7.8 | ~4h (4 fichiers modifiés) | <30min (2 fichiers créés) | <30min | ✓ |
| Testabilité (mocks) | Impossible | 100% mockable | 100% | ✓ |
| Couverture tests unitaires | 0% | >80% | >80% | ✓ |
| Duplication code (logging, cache) | 5x répété | 1x centralisé | 1x | ✓ |
```

---

## 📚 DOCUMENTATION COMPLÉMENTAIRE

### README.md du nouveau système

```markdown
# Validation CH7 V2 - Architecture Clean

## Différences avec l'ancien système

| Aspect | Ancien (validation_ch7) | Nouveau (validation_ch7_v2) |
|--------|-------------------------|------------------------------|
| Architecture | Monolithique (1876 lignes test) | Layered (domain, infra, orchestration) |
| Testabilité | Impossible | 100% mockable |
| Configuration | Hardcodée | YAML externalisée |
| Logging | Répété 5x | Centralisé |
| Cache/Checkpoints | Mélangé dans tests | ArtifactManager dédié |
| Ajout section 7.8 | 4 fichiers modifiés | 2 fichiers créés |

## Innovations préservées

✅ Cache additif intelligent (économie 85%)
✅ Config-hashing MD5 (validation checkpoint)
✅ Controller autonome (state tracking)
✅ Dual cache system (baseline + RL)
✅ Checkpoint rotation automatique
✅ Templates LaTeX réutilisables
✅ Session tracking JSON

## Usage

```bash
# Test section 7.6 (mode rapide)
python -m scripts.entry_points.cli --section section_7_6_rl_performance --quick-test

# Test complet (Kaggle GPU)
python -m scripts.entry_points.cli --section section_7_6_rl_performance --device gpu

# Tous les tests
python -m scripts.entry_points.cli --section all
```

## Architecture

```
validation_ch7_v2/
├── scripts/
│   ├── entry_points/      ← Layer 0: CLI, Kaggle
│   ├── orchestration/     ← Layer 1: Orchestration
│   ├── domain/            ← Layer 2: Métier pur
│   ├── infrastructure/    ← Layer 3: I/O, caching, logging
│   └── reporting/         ← Sub-layer: LaTeX, figures
├── configs/               ← YAML configs
├── templates/             ← LaTeX templates
├── tests/                 ← Tests unitaires (NOUVEAU!)
├── checkpoints/           ← Checkpoints RL
└── cache/                 ← Cache simulation
```
```

---

## 🚦 PLAN D'ACTION IMMÉDIAT

### Étape 1: Créer la structure (30 min)
```bash
cd "d:\Projets\Alibi\Code project"
mkdir -p validation_ch7_v2/scripts/{entry_points,orchestration,domain,infrastructure,reporting}
mkdir -p validation_ch7_v2/{configs/sections,templates,tests,checkpoints,cache}
touch validation_ch7_v2/README.md
# Créer tous les __init__.py
```

### Étape 2: Générer code via Copilot (8-12 heures)
- Utiliser les prompts ci-dessus dans l'ordre
- Phase 1 → Phase 2 → Phase 3 (critique!) → Phase 4 → Phase 5 → Phase 6 → Phase 7
- Valider chaque phase avant de passer à la suivante

### Étape 3: Tests unitaires (4 heures)
- Créer tests pour infrastructure (artifact_manager, config, logger)
- Créer tests pour domain (mocking simulator/model)
- Créer tests d'intégration (workflow complet)

### Étape 4: Migration progressive (variable)
- Comparer résultats ancien vs nouveau
- Vérifier métriques identiques
- Basculer progressivement

---

## ⚠️ POINTS D'ATTENTION CRITIQUES

### 1. Préservation du Cache Additif (Innovation #1)
Le cache additif est l'innovation la plus complexe. Vérifier:
- `extend_baseline_cache()` reprend bien depuis `cached_states[-1]`
- Simulation de l'extension UNIQUEMENT (pas tout refaire)
- Validation de cohérence (mass conservation)

### 2. Config-Hashing MD5 (Innovation #2)
Vérifier:
- Hash calculé AVANT checkpoint save
- Validation hash AVANT checkpoint load
- Archivage automatique si mismatch

### 3. Controller State Tracking (Innovation #3)
Vérifier:
- `controller.time_step` incrémenté correctement
- Reprise possible (`controller.time_step = cached_duration`)

### 4. Dual Cache System (Innovation #4)
Vérifier:
- Baseline cache: SANS config_hash (universel)
- RL cache: AVEC config_hash (config-specific)
- Pas de confusion entre les deux

---

**FIN DU PLAN DE REFACTORISATION**
**Avec respect pour le système existant et confiance dans le nouveau.**
