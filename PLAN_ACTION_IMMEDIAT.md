# 🎯 PLAN D'ACTION IMMÉDIAT - REFACTORISATION COMPLÈTE

**Date**: 2025-10-26  
**Objectif**: Éliminer YAML + Refactoriser runner.py  
**Durée estimée**: 1 semaine (5-7 jours)

---

## 📋 RÉSUMÉ EXÉCUTIF

**Problèmes identifiés** :
1. ❌ **YAML** = bugs runtime, pas de validation, pas d'IDE support
2. ❌ **runner.py** = 999 lignes (3.3x limite), 9+ responsabilités, God Object

**Solution** :
1. ✅ **Pydantic configs** = validation immédiate, type-safe, IDE autocomplete
2. ✅ **4 extractions** = ICBuilder, BCController, StateManager, TimeStepper

**Résultat attendu** :
- Configuration 100% Python (0 YAML)
- runner.py : 999 → ~664 lignes (-34%)
- Code maintenable, testable, type-safe

---

## 🚀 ORDRE D'EXÉCUTION RECOMMANDÉ

### **PHASE 1: PYDANTIC CONFIGS FIRST** (Jours 1-2)

**Pourquoi d'abord ?**
- Indépendant du reste du code
- Validation facile isolément
- Fournit interfaces claires pour refactoring runner

### **PHASE 2: RUNNER REFACTORING** (Jours 3-5)

**Pourquoi après ?**
- Une fois configs type-safe, extraction plus sûre
- Interfaces bien définies par Pydantic
- Backward compatibility préservée

---

## 📅 JOUR 1 : PYDANTIC CONFIGS (Partie 1)

**Objectif** : Créer 6 modules config + ConfigBuilder

### Étape 1.1 : Structure (5 min)

```powershell
# Créer structure
New-Item -ItemType Directory -Path "arz_model\config" -Force
New-Item -ItemType File -Path "arz_model\config\__init__.py"
New-Item -ItemType File -Path "arz_model\config\grid_config.py"
New-Item -ItemType File -Path "arz_model\config\ic_config.py"
New-Item -ItemType File -Path "arz_model\config\bc_config.py"
New-Item -ItemType File -Path "arz_model\config\physics_config.py"
New-Item -ItemType File -Path "arz_model\config\simulation_config.py"
New-Item -ItemType File -Path "arz_model\config\builders.py"
```

### Étape 1.2 : Installer Pydantic (1 min)

```powershell
pip install pydantic
```

### Étape 1.3 : Créer GridConfig (30 min)

**Fichier** : `arz_model/config/grid_config.py`

**Contenu** : Copier depuis `ARCHITECTURE_FINALE_SANS_YAML.md` section "MODULE 1"

**Test rapide** :
```powershell
python -c "from arz_model.config.grid_config import GridConfig; g = GridConfig(N=200, xmin=0.0, xmax=20.0); print(g)"
```

### Étape 1.4 : Créer IC Config (45 min)

**Fichier** : `arz_model/config/ic_config.py`

**Contenu** : Copier depuis `ARCHITECTURE_FINALE_SANS_YAML.md` section "MODULE 2"

**Test rapide** :
```powershell
python -c "from arz_model.config.ic_config import UniformEquilibriumIC; ic = UniformEquilibriumIC(rho_m=0.1, rho_c=0.05, R_val=10); print(ic)"
```

### Étape 1.5 : Créer BC Config (45 min)

**Fichier** : `arz_model/config/bc_config.py`

**Contenu** : Copier depuis `ARCHITECTURE_FINALE_SANS_YAML.md` section "MODULE 3"

**Test rapide** :
```powershell
python -c "from arz_model.config.bc_config import BoundaryConditionsConfig, InflowBC, OutflowBC, BCState; bc = BoundaryConditionsConfig(left=InflowBC(state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)), right=OutflowBC(state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0))); print(bc)"
```

**Total Jour 1** : ~2h

---

## 📅 JOUR 2 : PYDANTIC CONFIGS (Partie 2)

### Étape 2.1 : Créer PhysicsConfig (20 min)

**Fichier** : `arz_model/config/physics_config.py`

**Contenu** : Copier depuis `ARCHITECTURE_FINALE_SANS_YAML.md` section "MODULE 4"

### Étape 2.2 : Créer SimulationConfig (ROOT) (40 min)

**Fichier** : `arz_model/config/simulation_config.py`

**Contenu** : Copier depuis `ARCHITECTURE_FINALE_SANS_YAML.md` section "MODULE 5"

### Étape 2.3 : Créer ConfigBuilder (30 min)

**Fichier** : `arz_model/config/builders.py`

**Contenu** : Copier depuis `ARCHITECTURE_FINALE_SANS_YAML.md` section "MODULE 6"

### Étape 2.4 : Tests unitaires configs (1h)

**Fichier** : `tests/test_pydantic_configs.py`

```python
"""Test Pydantic configurations"""

from arz_model.config.builders import ConfigBuilder
from arz_model.config.grid_config import GridConfig
import pytest


def test_grid_config_valid():
    """Test valid grid configuration"""
    grid = GridConfig(N=200, xmin=0.0, xmax=20.0)
    assert grid.N == 200
    assert grid.dx == 0.1
    print("✅ GridConfig valid test OK")


def test_grid_config_invalid():
    """Test invalid grid configuration (xmax < xmin)"""
    with pytest.raises(ValueError):
        grid = GridConfig(N=200, xmin=20.0, xmax=0.0)
    print("✅ GridConfig validation test OK")


def test_section76_builder():
    """Test Section 7.6 config builder"""
    config = ConfigBuilder.section_7_6(N=200, t_final=1000.0, device='gpu')
    assert config.grid.N == 200
    assert config.t_final == 1000.0
    assert config.device == 'gpu'
    print("✅ ConfigBuilder section_7_6 test OK")


def test_simple_test_builder():
    """Test simple test config builder"""
    config = ConfigBuilder.simple_test()
    assert config.grid.N == 100
    assert config.t_final == 10.0
    print("✅ ConfigBuilder simple_test test OK")


if __name__ == '__main__':
    test_grid_config_valid()
    test_grid_config_invalid()
    test_section76_builder()
    test_simple_test_builder()
    print("\n✅ ALL CONFIG TESTS PASSED!")
```

**Lancer tests** :
```powershell
python tests/test_pydantic_configs.py
```

**✅ CHECKPOINT JOUR 2** : Tous les configs Pydantic créés et testés

**Total Jour 2** : ~2h30

---

## 📅 JOUR 3 : ADAPTER RUNNER.PY (Partie 1)

### Étape 3.1 : Backup runner.py original (1 min)

```powershell
Copy-Item "arz_model\simulation\runner.py" "arz_model\simulation\runner_OLD_BACKUP.py"
```

### Étape 3.2 : Modifier signature __init__ (30 min)

**Dans** `arz_model/simulation/runner.py` :

**AVANT** :
```python
def __init__(self, scenario_config_path=None, base_config_path=None, ...):
    # Load YAML
    self.params = ModelParameters()
    self.params.load_from_yaml(...)
```

**APRÈS** :
```python
from arz_model.config.simulation_config import SimulationConfig

def __init__(self, config: SimulationConfig):
    """
    Initialize simulation runner
    
    Args:
        config: SimulationConfig object (validated by Pydantic)
    """
    self.config = config
    self.quiet = config.quiet
```

### Étape 3.3 : Remplacer self.params par self.config (1h)

**Rechercher/Remplacer dans runner.py** :
- `self.params.N` → `self.config.grid.N`
- `self.params.xmin` → `self.config.grid.xmin`
- `self.params.xmax` → `self.config.grid.xmax`
- `self.params.device` → `self.config.device`
- `self.params.t_final` → `self.config.t_final`
- etc.

**Total Jour 3** : ~1h30

---

## 📅 JOUR 4 : EXTRACTIONS RUNNER.PY (Partie 1)

### Étape 4.1 : Créer structure extraction (5 min)

```powershell
New-Item -ItemType Directory -Path "arz_model\simulation\initialization" -Force
New-Item -ItemType Directory -Path "arz_model\simulation\boundaries" -Force
New-Item -ItemType Directory -Path "arz_model\simulation\state" -Force
New-Item -ItemType Directory -Path "arz_model\simulation\execution" -Force
```

### Étape 4.2 : Extract ICBuilder (1h)

**Créer** `arz_model/simulation/initialization/ic_builder.py`

**Extraire de runner.py** : Logique création IC (lignes ~150-250)

**Signature** :
```python
class ICBuilder:
    @staticmethod
    def build(ic_config: InitialConditionsConfig, 
              grid: Grid, 
              physics: PhysicsConfig) -> np.ndarray:
        """Build initial conditions from config"""
        # Extract IC creation logic here
        ...
```

### Étape 4.3 : Extract BCController (1h)

**Créer** `arz_model/simulation/boundaries/bc_controller.py`

**Extraire de runner.py** : Logique BC (lignes ~250-350)

**Signature** :
```python
class BCController:
    def __init__(self, bc_config: BoundaryConditionsConfig, physics: PhysicsConfig):
        """Initialize BC controller"""
        self.bc_config = bc_config
        self.physics = physics
        # Extract BC logic here
    
    def apply_bc(self, U, t):
        """Apply boundary conditions at time t"""
        ...
```

**Total Jour 4** : ~2h

---

## 📅 JOUR 5 : EXTRACTIONS RUNNER.PY (Partie 2)

### Étape 5.1 : Extract StateManager (1h)

**Créer** `arz_model/simulation/state/state_manager.py`

**Extraire de runner.py** : Logique gestion état (lignes ~400-500)

**Signature** :
```python
class StateManager:
    def __init__(self, U0: np.ndarray, device: str = 'cpu'):
        """Initialize state manager"""
        self.U = U0
        self.device = device
        self.t = 0.0
        self.step_count = 0
        self.times = []
        self.states = []
    
    def update_state(self, U_new):
        """Update current state"""
        ...
    
    def store_output(self, U_phys):
        """Store output snapshot"""
        ...
```

### Étape 5.2 : Extract TimeStepper (1h)

**Créer** `arz_model/simulation/execution/time_stepper.py`

**Extraire de runner.py** : Logique intégration temps (lignes ~500-700)

**Signature** :
```python
class TimeStepper:
    def __init__(self, physics: PhysicsConfig, grid: Grid, 
                 bc_controller: BCController, device: str = 'cpu'):
        """Initialize time stepper"""
        ...
    
    def step(self, U, t):
        """Perform one time step"""
        # Returns: U_new, dt
        ...
```

### Étape 5.3 : Simplifier runner.py (1h)

**runner.py devient ORCHESTRATION uniquement** :

```python
class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.grid = GridBuilder.build(config.grid)
        U0 = ICBuilder.build(config.initial_conditions, self.grid, config.physics)
        self.bc_controller = BCController(config.boundary_conditions, config.physics)
        self.state = StateManager(U0, device=config.device)
        self.time_stepper = TimeStepper(config.physics, self.grid, 
                                       self.bc_controller, device=config.device)
    
    def run(self):
        """Main simulation loop (ORCHESTRATION ONLY)"""
        while self.state.t < self.config.t_final:
            U = self.state.get_current_state()
            U_new, dt = self.time_stepper.step(U, self.state.t)
            self.state.update_state(U_new)
            self.state.advance_time(dt)
            # Store output if needed...
        return results
```

**✅ CHECKPOINT JOUR 5** : runner.py refactorisé (999 → ~664 lignes)

**Total Jour 5** : ~3h

---

## 📅 JOUR 6 : TESTS & VALIDATION

### Étape 6.1 : Test Bug 31 avec nouveaux configs (1h)

**Créer** `test_bug31_pydantic.py` :

```python
"""Test Bug 31 fix with Pydantic configs"""

from arz_model.config.builders import ConfigBuilder
from arz_model.simulation.runner import SimulationRunner

def test_bug31_ic_bc_separation():
    """Test that IC and BC are properly separated"""
    
    # Create config
    config = ConfigBuilder.section_7_6(N=200, t_final=10.0, device='cpu')
    
    # Create runner
    runner = SimulationRunner(config)
    
    # Check IC independent of BC
    assert runner.config.initial_conditions is not None
    assert runner.config.boundary_conditions is not None
    
    # Run short simulation
    results = runner.run()
    
    # Check results
    assert len(results['times']) > 0
    assert results['states'].shape[0] > 0
    
    print("✅ Bug 31 test PASSED with Pydantic configs!")

if __name__ == '__main__':
    test_bug31_ic_bc_separation()
```

**Lancer test** :
```powershell
python test_bug31_pydantic.py
```

### Étape 6.2 : Test intégration complète (1h)

**Test court (1000 steps)** :

```python
from arz_model.config.builders import ConfigBuilder
from arz_model.simulation.runner import SimulationRunner

config = ConfigBuilder.section_7_6(N=200, t_final=100.0, device='gpu')
runner = SimulationRunner(config)
results = runner.run()

print(f"✅ Integration test completed!")
print(f"   Steps: {len(results['times'])}")
print(f"   Final time: {results['times'][-1]:.2f}s")
```

### Étape 6.3 : Tests unitaires extractions (1h)

**Test chaque classe extraite isolément** :

```python
# Test ICBuilder
from arz_model.simulation.initialization.ic_builder import ICBuilder
from arz_model.config.builders import ConfigBuilder

config = ConfigBuilder.simple_test()
grid = GridBuilder.build(config.grid)
U0 = ICBuilder.build(config.initial_conditions, grid, config.physics)
assert U0.shape == (4, grid.total_cells)
print("✅ ICBuilder test OK")

# Test BCController
# Test StateManager
# Test TimeStepper
```

**Total Jour 6** : ~3h

---

## 📅 JOUR 7 : SECTION 7.6 TRAINING

### Étape 7.1 : Script training final (30 min)

**Créer** `train_section76_pydantic.py` :

```python
"""Section 7.6 Training with Pydantic configs (NO YAML!)"""

from arz_model.config.builders import ConfigBuilder
from arz_model.simulation.runner import SimulationRunner
import time

def main():
    print("=" * 80)
    print("SECTION 7.6 TRAINING - PYDANTIC VERSION (NO YAML!)")
    print("=" * 80)
    
    # Build config (NO YAML!)
    config = ConfigBuilder.section_7_6(
        N=200,
        t_final=1000.0,
        device='gpu'
    )
    
    print(f"\n✅ Configuration created:")
    print(f"   Grid: N={config.grid.N}, domain=[{config.grid.xmin}, {config.grid.xmax}] km")
    print(f"   IC: {config.initial_conditions.type}")
    print(f"   BC: left={config.boundary_conditions.left.type}, "
          f"right={config.boundary_conditions.right.type}")
    print(f"   Device: {config.device}")
    print(f"   t_final: {config.t_final}s")
    
    # Create runner
    print("\n🚀 Creating simulation runner...")
    runner = SimulationRunner(config)
    
    # Run simulation
    print("\n🚀 Starting Section 7.6 training...")
    start = time.time()
    results = runner.run()
    elapsed = time.time() - start
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED!")
    print("=" * 80)
    print(f"   Elapsed time: {elapsed/3600:.2f} hours")
    print(f"   Total steps: {len(results['times'])}")
    print(f"   Final time: {results['times'][-1]:.2f}s")
    
    # Save results
    import numpy as np
    np.save('results_section76_pydantic.npy', results)
    print(f"\n💾 Results saved to: results_section76_pydantic.npy")

if __name__ == '__main__':
    main()
```

### Étape 7.2 : Lancer training (8-10h GPU)

```powershell
python train_section76_pydantic.py
```

**✅ SUCCESS CRITERIA** :
- No crashes
- No NaNs
- Training completes to t_final
- Results saved successfully

**Total Jour 7** : 8-10h (GPU training)

---

## 📊 CHECKLIST FINAL

### ✅ Configs Pydantic
- [ ] GridConfig créé et testé
- [ ] ICConfig créé et testé
- [ ] BCConfig créé et testé
- [ ] PhysicsConfig créé et testé
- [ ] SimulationConfig (ROOT) créé et testé
- [ ] ConfigBuilder créé et testé
- [ ] Tests unitaires configs PASSENT

### ✅ Runner Refactoring
- [ ] ICBuilder extrait
- [ ] BCController extrait
- [ ] StateManager extrait
- [ ] TimeStepper extrait
- [ ] runner.py simplifié (999 → ~664 lignes)
- [ ] Backward compatibility préservée

### ✅ Tests & Validation
- [ ] Bug 31 test PASSE
- [ ] Test intégration court PASSE
- [ ] Tests unitaires extractions PASSENT
- [ ] Section 7.6 training COMPLÉTÉ sans crash

---

## 🎯 COMMANDES RAPIDES (COPIER-COLLER)

### Jour 1 : Setup Pydantic

```powershell
# Structure
New-Item -ItemType Directory -Path "arz_model\config" -Force
New-Item -ItemType File -Path "arz_model\config\__init__.py"
New-Item -ItemType File -Path "arz_model\config\grid_config.py"
New-Item -ItemType File -Path "arz_model\config\ic_config.py"
New-Item -ItemType File -Path "arz_model\config\bc_config.py"
New-Item -ItemType File -Path "arz_model\config\physics_config.py"
New-Item -ItemType File -Path "arz_model\config\simulation_config.py"
New-Item -ItemType File -Path "arz_model\config\builders.py"

# Install
pip install pydantic

# Test
python -c "from pydantic import BaseModel; print('✅ Pydantic installed')"
```

### Jour 4-5 : Setup Extractions

```powershell
# Structure
New-Item -ItemType Directory -Path "arz_model\simulation\initialization" -Force
New-Item -ItemType Directory -Path "arz_model\simulation\boundaries" -Force
New-Item -ItemType Directory -Path "arz_model\simulation\state" -Force
New-Item -ItemType Directory -Path "arz_model\simulation\execution" -Force

New-Item -ItemType File -Path "arz_model\simulation\initialization\__init__.py"
New-Item -ItemType File -Path "arz_model\simulation\initialization\ic_builder.py"
New-Item -ItemType File -Path "arz_model\simulation\boundaries\__init__.py"
New-Item -ItemType File -Path "arz_model\simulation\boundaries\bc_controller.py"
New-Item -ItemType File -Path "arz_model\simulation\state\__init__.py"
New-Item -ItemType File -Path "arz_model\simulation\state\state_manager.py"
New-Item -ItemType File -Path "arz_model\simulation\execution\__init__.py"
New-Item -ItemType File -Path "arz_model\simulation\execution\time_stepper.py"

# Backup
Copy-Item "arz_model\simulation\runner.py" "arz_model\simulation\runner_OLD_BACKUP.py"
```

---

## 🚨 SI PROBLÈME

### Erreur import Pydantic
```powershell
pip install --upgrade pydantic
```

### Erreur validation config
```python
# Debug mode
try:
    config = ConfigBuilder.section_7_6()
except Exception as e:
    print(f"❌ Validation error: {e}")
    import traceback
    traceback.print_exc()
```

### Runner crash après extraction
```powershell
# Rollback
Copy-Item "arz_model\simulation\runner_OLD_BACKUP.py" "arz_model\simulation\runner.py"
```

---

## 📈 RÉSULTAT ATTENDU

**AVANT (Jour 0)** :
- ❌ runner.py : 999 lignes, God Object
- ❌ Configuration : YAML fragile
- ❌ Validation : Runtime uniquement

**APRÈS (Jour 7)** :
- ✅ runner.py : ~664 lignes (-34%), orchestration uniquement
- ✅ 4 classes extraites : ICBuilder, BCController, StateManager, TimeStepper
- ✅ Configuration : 100% Python, type-safe, IDE autocomplete
- ✅ Validation : Immédiate (Pydantic)
- ✅ Tests : Tous passent
- ✅ Training Section 7.6 : Complété sans crash

---

## 🎯 COMMENCE PAR QUOI ?

**MA RECOMMANDATION** : 

**👉 COMMENCE PAR JOUR 1 (MAINTENANT !)**

```powershell
# 1. Créer structure
New-Item -ItemType Directory -Path "arz_model\config" -Force

# 2. Installer Pydantic
pip install pydantic

# 3. Créer premier fichier (GridConfig)
# (Copier contenu depuis ARCHITECTURE_FINALE_SANS_YAML.md)
```

**Dis-moi quand tu as fini Jour 1, je te guide pour Jour 2 !** 🚀

---

**TU VEUX QUE JE CRÉE LES FICHIERS POUR TOI ?** 

Dis juste "GO" et je crée tout de suite :
- ✅ Structure complète
- ✅ Tous les fichiers config Pydantic
- ✅ ConfigBuilder
- ✅ Tests unitaires

**OU TU VEUX LE FAIRE TOI-MÊME ?**

Suis ce plan jour par jour et je t'assiste si besoin !

🎯 **À TOI DE JOUER !**
