# 🎯 ARCHITECTURE FINALE : SANS YAML, TOUT EN PYTHON

**Date**: 2025-10-26  
**Philosophie**: YAML = source de bugs. Python = type-safe, validé, auto-complété par IDE  
**Objectif**: Configuration 100% en Python pur avec Pydantic pour validation

---

## ❌ POURQUOI ÉLIMINER YAML ?

### Problèmes du YAML actuel

```yaml
# ❌ PROBLÈME 1: Pas de validation de types
initial_conditions:
  rho_m: "0.1"  # String au lieu de float → crash runtime!

# ❌ PROBLÈME 2: Pas d'autocomplétion IDE
boundary_conditions:
  lft:  # Typo! Devrait être "left" → crash runtime!
    type: inflow

# ❌ PROBLÈME 3: Validation tardive (après 8h GPU)
simulation:
  N: -100  # Valeur invalide découverte après lancement!

# ❌ PROBLÈME 4: Pas de documentation inline
some_parameter: 42  # C'est quoi ce 42 ???

# ❌ PROBLÈME 5: Merge de configs fragile
# base.yml + scenario.yml → comportement imprévisible
```

### ✅ Solution : Python pur + Pydantic

```python
# ✅ AVANTAGE 1: Validation de types automatique
from pydantic import BaseModel, Field

class InitialConditions(BaseModel):
    rho_m: float = Field(gt=0, le=1, description="Motorcycle density [0,1]")
    # ↑ IDE sait que c'est un float
    # ↑ Validation automatique: 0 < rho_m ≤ 1
    # ↑ Documentation intégrée

# ✅ AVANTAGE 2: Autocomplétion IDE
ic = InitialConditions(
    rho_m=0.1,  # ← IDE autocomplete "rho_m"
    # ← IDE affiche docstring "Motorcycle density [0,1]"
)

# ✅ AVANTAGE 3: Validation immédiate (avant lancement)
ic = InitialConditions(rho_m=-0.5)  # ← ERREUR IMMÉDIATE!
# ValidationError: rho_m must be > 0

# ✅ AVANTAGE 4: Documentation = code
# Docstrings, type hints, Field descriptions

# ✅ AVANTAGE 5: Composition claire
config = SimulationConfig(
    grid=GridConfig(N=200, xmin=0, xmax=20),
    ic=InitialConditions(...),
    bc=BoundaryConditions(...)
)
```

---

## 🏗️ ARCHITECTURE FINALE : 8 MODULES

```
arz_model/
├── config/
│   ├── __init__.py
│   ├── simulation_config.py      # SimulationConfig (racine)
│   ├── grid_config.py             # GridConfig
│   ├── ic_config.py               # InitialConditionsConfig
│   ├── bc_config.py               # BoundaryConditionsConfig
│   ├── physics_config.py          # PhysicsConfig
│   ├── rl_config.py               # RLTrainingConfig
│   └── builders.py                # Config builders (helpers)
│
├── simulation/
│   ├── runner.py                  # SimulationRunner (ORCHESTRATION)
│   ├── initialization/
│   │   ├── ic_builder.py          # ICBuilder
│   │   └── grid_builder.py        # GridBuilder
│   ├── boundaries/
│   │   └── bc_controller.py       # BCController
│   ├── state/
│   │   └── state_manager.py       # StateManager
│   └── execution/
│       └── time_stepper.py        # TimeStepper
│
└── core/
    └── parameters.py              # DEPRECATED (legacy)
```

**CHANGEMENT MAJEUR** :
- ❌ `ModelParameters` avec YAML → **SUPPRIMÉ**
- ✅ `SimulationConfig` avec Pydantic → **NOUVEAU**

---

## 📦 MODULE 1 : Grid Configuration

**Fichier** : `arz_model/config/grid_config.py`

```python
"""
Grid Configuration Module

Defines spatial discretization parameters with validation
"""

from pydantic import BaseModel, Field, field_validator


class GridConfig(BaseModel):
    """
    Spatial grid configuration
    
    Defines the 1D spatial domain [xmin, xmax] discretized into N cells
    """
    
    N: int = Field(
        ...,  # Required
        gt=0,
        le=10000,
        description="Number of spatial cells"
    )
    
    xmin: float = Field(
        0.0,
        description="Left boundary of domain (km)"
    )
    
    xmax: float = Field(
        ...,  # Required
        gt=0,
        description="Right boundary of domain (km)"
    )
    
    ghost_cells: int = Field(
        2,
        ge=1,
        le=4,
        description="Number of ghost cells for boundary conditions"
    )
    
    @field_validator('xmax')
    @classmethod
    def xmax_must_be_greater_than_xmin(cls, v, info):
        """Validate xmax > xmin"""
        if 'xmin' in info.data and v <= info.data['xmin']:
            raise ValueError(f'xmax ({v}) must be > xmin ({info.data["xmin"]})')
        return v
    
    @property
    def dx(self) -> float:
        """Grid spacing (km)"""
        return (self.xmax - self.xmin) / self.N
    
    @property
    def total_cells(self) -> int:
        """Total cells including ghost cells"""
        return self.N + 2 * self.ghost_cells
    
    def __repr__(self):
        return (f"GridConfig(N={self.N}, domain=[{self.xmin}, {self.xmax}] km, "
                f"dx={self.dx:.4f} km, ghost_cells={self.ghost_cells})")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    # Valid configuration
    grid = GridConfig(N=200, xmin=0.0, xmax=20.0)
    print(grid)
    # Output: GridConfig(N=200, domain=[0.0, 20.0] km, dx=0.1000 km, ghost_cells=2)
    
    # Invalid configuration (caught immediately!)
    try:
        grid_bad = GridConfig(N=-100, xmin=0.0, xmax=20.0)
    except Exception as e:
        print(f"❌ Validation error: {e}")
        # ValidationError: N must be > 0
```

---

## 📦 MODULE 2 : Initial Conditions Configuration

**Fichier** : `arz_model/config/ic_config.py`

```python
"""
Initial Conditions Configuration Module

Defines initial state configuration with strong typing
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Optional
from enum import Enum


class ICType(str, Enum):
    """Initial condition types"""
    UNIFORM = "uniform"
    UNIFORM_EQUILIBRIUM = "uniform_equilibrium"
    RIEMANN = "riemann"
    GAUSSIAN_PULSE = "gaussian_pulse"
    FROM_FILE = "from_file"


class UniformIC(BaseModel):
    """Uniform initial conditions (constant state)"""
    
    type: Literal[ICType.UNIFORM] = ICType.UNIFORM
    
    rho_m: float = Field(gt=0, le=1, description="Motorcycle density")
    w_m: float = Field(gt=0, description="Motorcycle velocity (km/h)")
    rho_c: float = Field(gt=0, le=1, description="Car density")
    w_c: float = Field(gt=0, description="Car velocity (km/h)")


class UniformEquilibriumIC(BaseModel):
    """Uniform equilibrium initial conditions"""
    
    type: Literal[ICType.UNIFORM_EQUILIBRIUM] = ICType.UNIFORM_EQUILIBRIUM
    
    rho_m: float = Field(gt=0, le=1, description="Motorcycle density")
    rho_c: float = Field(gt=0, le=1, description="Car density")
    R_val: int = Field(gt=0, description="Road quality value [1-10]")


class RiemannIC(BaseModel):
    """Riemann problem initial conditions (discontinuity)"""
    
    type: Literal[ICType.RIEMANN] = ICType.RIEMANN
    
    x_discontinuity: float = Field(description="Position of discontinuity (km)")
    
    # Left state
    rho_m_left: float = Field(gt=0, le=1, description="Motorcycle density (left)")
    w_m_left: float = Field(gt=0, description="Motorcycle velocity (left)")
    rho_c_left: float = Field(gt=0, le=1, description="Car density (left)")
    w_c_left: float = Field(gt=0, description="Car velocity (left)")
    
    # Right state
    rho_m_right: float = Field(gt=0, le=1, description="Motorcycle density (right)")
    w_m_right: float = Field(gt=0, description="Motorcycle velocity (right)")
    rho_c_right: float = Field(gt=0, le=1, description="Car density (right)")
    w_c_right: float = Field(gt=0, description="Car velocity (right)")


class GaussianPulseIC(BaseModel):
    """Gaussian pulse initial conditions"""
    
    type: Literal[ICType.GAUSSIAN_PULSE] = ICType.GAUSSIAN_PULSE
    
    x_center: float = Field(description="Center position (km)")
    sigma: float = Field(gt=0, description="Pulse width (km)")
    amplitude: float = Field(gt=0, description="Pulse amplitude")
    
    background_rho_m: float = Field(gt=0, le=1, description="Background motorcycle density")
    background_rho_c: float = Field(gt=0, le=1, description="Background car density")


class FileBasedIC(BaseModel):
    """Initial conditions loaded from file"""
    
    type: Literal[ICType.FROM_FILE] = ICType.FROM_FILE
    
    filepath: str = Field(description="Path to initial conditions file (.npy or .txt)")


# Union type for all IC types
from typing import Union
InitialConditionsConfig = Union[
    UniformIC,
    UniformEquilibriumIC,
    RiemannIC,
    GaussianPulseIC,
    FileBasedIC
]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    # Example 1: Uniform equilibrium (most common for Section 7.6)
    ic_equilibrium = UniformEquilibriumIC(
        rho_m=0.1,
        rho_c=0.05,
        R_val=10
    )
    print(f"✅ {ic_equilibrium}")
    
    # Example 2: Riemann problem (for testing)
    ic_riemann = RiemannIC(
        x_discontinuity=10.0,
        rho_m_left=0.1, w_m_left=30.0, rho_c_left=0.05, w_c_left=40.0,
        rho_m_right=0.5, w_m_right=10.0, rho_c_right=0.3, w_c_right=15.0
    )
    print(f"✅ {ic_riemann}")
    
    # Example 3: Invalid config (caught immediately!)
    try:
        ic_bad = UniformEquilibriumIC(rho_m=-0.5, rho_c=0.05, R_val=10)
    except Exception as e:
        print(f"❌ Validation error: {e}")
```

---

## 📦 MODULE 3 : Boundary Conditions Configuration

**Fichier** : `arz_model/config/bc_config.py`

```python
"""
Boundary Conditions Configuration Module

Defines boundary condition configuration with strong typing
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Optional, Dict
from enum import Enum


class BCType(str, Enum):
    """Boundary condition types"""
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    PERIODIC = "periodic"
    REFLECTIVE = "reflective"


class BCState(BaseModel):
    """Boundary condition state vector [rho_m, w_m, rho_c, w_c]"""
    
    rho_m: float = Field(ge=0, le=1, description="Motorcycle density")
    w_m: float = Field(gt=0, description="Motorcycle velocity (km/h)")
    rho_c: float = Field(ge=0, le=1, description="Car density")
    w_c: float = Field(gt=0, description="Car velocity (km/h)")
    
    def to_array(self) -> List[float]:
        """Convert to [rho_m, w_m, rho_c, w_c] list"""
        return [self.rho_m, self.w_m, self.rho_c, self.w_c]


class BCScheduleItem(BaseModel):
    """Single item in BC schedule (time-dependent BC)"""
    
    time: float = Field(ge=0, description="Time to activate this phase (s)")
    phase_id: int = Field(ge=0, description="Phase ID to activate")


class InflowBC(BaseModel):
    """Inflow boundary condition"""
    
    type: Literal[BCType.INFLOW] = BCType.INFLOW
    state: BCState = Field(description="Inflow state")
    schedule: Optional[List[BCScheduleItem]] = Field(
        None,
        description="Optional time-dependent schedule"
    )


class OutflowBC(BaseModel):
    """Outflow boundary condition"""
    
    type: Literal[BCType.OUTFLOW] = BCType.OUTFLOW
    state: BCState = Field(description="Outflow state")


class PeriodicBC(BaseModel):
    """Periodic boundary condition"""
    
    type: Literal[BCType.PERIODIC] = BCType.PERIODIC


class ReflectiveBC(BaseModel):
    """Reflective (wall) boundary condition"""
    
    type: Literal[BCType.REFLECTIVE] = BCType.REFLECTIVE


# Union type for BC types
from typing import Union
BCSide = Union[InflowBC, OutflowBC, PeriodicBC, ReflectiveBC]


class BoundaryConditionsConfig(BaseModel):
    """Complete boundary conditions configuration"""
    
    left: BCSide = Field(description="Left boundary condition")
    right: BCSide = Field(description="Right boundary condition")
    
    # Optional: Traffic signal phases (for RL control)
    traffic_signal_phases: Optional[Dict[str, Dict[int, BCState]]] = Field(
        None,
        description="Traffic signal phase definitions for RL control"
    )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    # Example 1: Simple inflow/outflow (no schedule)
    bc_simple = BoundaryConditionsConfig(
        left=InflowBC(
            state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)
        ),
        right=OutflowBC(
            state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)
        )
    )
    print(f"✅ Simple BC: {bc_simple}")
    
    # Example 2: Time-dependent BC with schedule (Section 7.6 style)
    bc_scheduled = BoundaryConditionsConfig(
        left=InflowBC(
            state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0),
            schedule=[
                BCScheduleItem(time=0.0, phase_id=0),   # Normal flow
                BCScheduleItem(time=10.0, phase_id=1),  # High density (congestion)
                BCScheduleItem(time=50.0, phase_id=0)   # Back to normal
            ]
        ),
        right=OutflowBC(
            state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)
        ),
        traffic_signal_phases={
            'left': {
                0: BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0),  # Phase 0: Green
                1: BCState(rho_m=0.5, w_m=10.0, rho_c=0.3, w_c=15.0)    # Phase 1: Red (congestion)
            }
        }
    )
    print(f"✅ Scheduled BC: {bc_scheduled}")
    
    # Example 3: Periodic BC (for testing)
    bc_periodic = BoundaryConditionsConfig(
        left=PeriodicBC(),
        right=PeriodicBC()
    )
    print(f"✅ Periodic BC: {bc_periodic}")
```

---

## 📦 MODULE 4 : Physics Configuration

**Fichier** : `arz_model/config/physics_config.py`

```python
"""
Physics Configuration Module

Defines physical parameters with validation
"""

from pydantic import BaseModel, Field


class PhysicsConfig(BaseModel):
    """
    Physical parameters for ARZ model
    
    References:
    - λ parameters: Wong & Wong (2002)
    - V_max: Typical urban speeds
    """
    
    # Relaxation parameters
    lambda_m: float = Field(
        1.0,
        gt=0,
        description="Motorcycle relaxation parameter (1/s)"
    )
    
    lambda_c: float = Field(
        1.0,
        gt=0,
        description="Car relaxation parameter (1/s)"
    )
    
    # Maximum velocities
    V_max_m: float = Field(
        60.0,
        gt=0,
        le=200.0,
        description="Motorcycle maximum velocity (km/h)"
    )
    
    V_max_c: float = Field(
        80.0,
        gt=0,
        le=200.0,
        description="Car maximum velocity (km/h)"
    )
    
    # Lane interaction
    alpha: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Lane interaction coefficient [0,1]"
    )
    
    # Road quality (if not spatially varying)
    default_road_quality: int = Field(
        10,
        ge=1,
        le=10,
        description="Default road quality [1=bad, 10=excellent]"
    )
    
    def __repr__(self):
        return (f"PhysicsConfig(λ_m={self.lambda_m}, λ_c={self.lambda_c}, "
                f"V_max_m={self.V_max_m} km/h, V_max_c={self.V_max_c} km/h)")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    # Default physics parameters
    physics = PhysicsConfig()
    print(physics)
    
    # Custom parameters
    physics_custom = PhysicsConfig(
        lambda_m=1.5,
        lambda_c=1.2,
        V_max_m=50.0,
        V_max_c=70.0,
        alpha=0.7
    )
    print(physics_custom)
```

---

## 📦 MODULE 5 : Simulation Configuration (ROOT)

**Fichier** : `arz_model/config/simulation_config.py`

```python
"""
Simulation Configuration Module (ROOT)

This is the main configuration object that contains all simulation parameters
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

from arz_model.config.grid_config import GridConfig
from arz_model.config.ic_config import InitialConditionsConfig
from arz_model.config.bc_config import BoundaryConditionsConfig
from arz_model.config.physics_config import PhysicsConfig


class SimulationConfig(BaseModel):
    """
    Complete simulation configuration (ROOT)
    
    This replaces the old YAML-based ModelParameters
    """
    
    # ========================================================================
    # REQUIRED COMPONENTS
    # ========================================================================
    
    grid: GridConfig = Field(description="Spatial grid configuration")
    
    initial_conditions: InitialConditionsConfig = Field(
        description="Initial conditions configuration"
    )
    
    boundary_conditions: BoundaryConditionsConfig = Field(
        description="Boundary conditions configuration"
    )
    
    physics: PhysicsConfig = Field(
        default_factory=PhysicsConfig,
        description="Physical parameters"
    )
    
    # ========================================================================
    # TIME INTEGRATION
    # ========================================================================
    
    t_final: float = Field(
        gt=0,
        description="Final simulation time (s)"
    )
    
    output_dt: float = Field(
        1.0,
        gt=0,
        description="Output interval (s)"
    )
    
    cfl_number: float = Field(
        0.5,
        gt=0,
        le=1.0,
        description="CFL number for adaptive time stepping"
    )
    
    max_iterations: int = Field(
        100000,
        gt=0,
        description="Maximum number of time steps"
    )
    
    # ========================================================================
    # COMPUTATIONAL OPTIONS
    # ========================================================================
    
    device: Literal['cpu', 'gpu'] = Field(
        'cpu',
        description="Computation device"
    )
    
    quiet: bool = Field(
        False,
        description="Suppress output messages"
    )
    
    # ========================================================================
    # OPTIONAL: NETWORK SYSTEM
    # ========================================================================
    
    has_network: bool = Field(
        False,
        description="Enable network system (intersections)"
    )
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    @field_validator('output_dt')
    @classmethod
    def output_dt_must_be_less_than_t_final(cls, v, info):
        """Validate output_dt < t_final"""
        if 't_final' in info.data and v > info.data['t_final']:
            raise ValueError(f'output_dt ({v}) must be < t_final ({info.data["t_final"]})')
        return v
    
    def __repr__(self):
        return (f"SimulationConfig(\n"
                f"  grid={self.grid},\n"
                f"  ic={self.initial_conditions.type},\n"
                f"  bc=(left={self.boundary_conditions.left.type}, "
                f"right={self.boundary_conditions.right.type}),\n"
                f"  t_final={self.t_final}s, device={self.device}\n"
                f")")


# ============================================================================
# USAGE EXAMPLE: Section 7.6 Training Configuration
# ============================================================================

if __name__ == '__main__':
    from arz_model.config.ic_config import UniformEquilibriumIC
    from arz_model.config.bc_config import InflowBC, OutflowBC, BCState, BCScheduleItem
    
    # Section 7.6 configuration (without YAML!)
    config_section76 = SimulationConfig(
        grid=GridConfig(N=200, xmin=0.0, xmax=20.0),
        
        initial_conditions=UniformEquilibriumIC(
            rho_m=0.1,
            rho_c=0.05,
            R_val=10
        ),
        
        boundary_conditions=BoundaryConditionsConfig(
            left=InflowBC(
                state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0),
                schedule=[
                    BCScheduleItem(time=0.0, phase_id=0),
                    BCScheduleItem(time=100.0, phase_id=1)
                ]
            ),
            right=OutflowBC(
                state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)
            ),
            traffic_signal_phases={
                'left': {
                    0: BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0),
                    1: BCState(rho_m=0.5, w_m=10.0, rho_c=0.3, w_c=15.0)
                }
            }
        ),
        
        t_final=1000.0,
        output_dt=1.0,
        device='gpu'
    )
    
    print("=" * 80)
    print("SECTION 7.6 CONFIGURATION (NO YAML!)")
    print("=" * 80)
    print(config_section76)
    print("\n✅ Configuration validated successfully!")
    print(f"   Grid: N={config_section76.grid.N}, dx={config_section76.grid.dx:.4f} km")
    print(f"   IC: {config_section76.initial_conditions.type}")
    print(f"   BC: left={config_section76.boundary_conditions.left.type}, "
          f"right={config_section76.boundary_conditions.right.type}")
    print(f"   Time: t_final={config_section76.t_final}s, device={config_section76.device}")
```

---

## 📦 MODULE 6 : Configuration Builders (Ergonomics)

**Fichier** : `arz_model/config/builders.py`

```python
"""
Configuration Builders Module

Provides ergonomic helpers to construct common configurations
"""

from arz_model.config.simulation_config import SimulationConfig
from arz_model.config.grid_config import GridConfig
from arz_model.config.ic_config import UniformEquilibriumIC
from arz_model.config.bc_config import (
    BoundaryConditionsConfig, InflowBC, OutflowBC,
    BCState, BCScheduleItem
)
from arz_model.config.physics_config import PhysicsConfig


class ConfigBuilder:
    """Helper to build common configurations"""
    
    @staticmethod
    def section_7_6(N: int = 200, t_final: float = 1000.0, device: str = 'gpu') -> SimulationConfig:
        """
        Build Section 7.6 training configuration
        
        Default configuration for RL training with traffic signal control
        
        Args:
            N: Number of spatial cells
            t_final: Final simulation time (s)
            device: 'cpu' or 'gpu'
        
        Returns:
            SimulationConfig ready for RL training
        """
        return SimulationConfig(
            grid=GridConfig(N=N, xmin=0.0, xmax=20.0),
            
            initial_conditions=UniformEquilibriumIC(
                rho_m=0.1,
                rho_c=0.05,
                R_val=10
            ),
            
            boundary_conditions=BoundaryConditionsConfig(
                left=InflowBC(
                    state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0),
                    schedule=[
                        BCScheduleItem(time=0.0, phase_id=0),
                        BCScheduleItem(time=100.0, phase_id=1)
                    ]
                ),
                right=OutflowBC(
                    state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)
                ),
                traffic_signal_phases={
                    'left': {
                        0: BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0),   # Green
                        1: BCState(rho_m=0.5, w_m=10.0, rho_c=0.3, w_c=15.0)     # Red
                    }
                }
            ),
            
            t_final=t_final,
            output_dt=1.0,
            device=device
        )
    
    @staticmethod
    def simple_test(N: int = 100, t_final: float = 10.0) -> SimulationConfig:
        """
        Build simple test configuration
        
        Minimal configuration for quick testing
        
        Args:
            N: Number of spatial cells
            t_final: Final simulation time (s)
        
        Returns:
            SimulationConfig for testing
        """
        return SimulationConfig(
            grid=GridConfig(N=N, xmin=0.0, xmax=10.0),
            
            initial_conditions=UniformEquilibriumIC(
                rho_m=0.1,
                rho_c=0.05,
                R_val=10
            ),
            
            boundary_conditions=BoundaryConditionsConfig(
                left=InflowBC(
                    state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)
                ),
                right=OutflowBC(
                    state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)
                )
            ),
            
            t_final=t_final,
            output_dt=0.5,
            device='cpu'
        )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    # Build Section 7.6 config in ONE LINE!
    config = ConfigBuilder.section_7_6(N=200, t_final=1000.0, device='gpu')
    print("✅ Section 7.6 config created:")
    print(config)
    
    # Build test config in ONE LINE!
    config_test = ConfigBuilder.simple_test()
    print("\n✅ Test config created:")
    print(config_test)
```

---

## 🚀 NOUVEAU RUNNER.PY (SANS YAML)

**Fichier** : `arz_model/simulation/runner.py` (SIMPLIFIÉ)

```python
"""
Simulation Runner - NO YAML VERSION

Orchestrates simulation with Pydantic-based configuration
"""

import numpy as np
from typing import Optional, Dict, Any

from arz_model.config.simulation_config import SimulationConfig
from arz_model.simulation.initialization.ic_builder import ICBuilder
from arz_model.simulation.initialization.grid_builder import GridBuilder
from arz_model.simulation.boundaries.bc_controller import BCController
from arz_model.simulation.state.state_manager import StateManager
from arz_model.simulation.execution.time_stepper import TimeStepper


class SimulationRunner:
    """
    Orchestrate ARZ model simulation (NO YAML VERSION)
    
    This version uses Pydantic-based SimulationConfig instead of YAML
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize simulation runner
        
        Args:
            config: SimulationConfig object (validated by Pydantic)
        """
        self.config = config
        self.quiet = config.quiet
        
        # ====================================================================
        # STEP 1: Build Grid
        # ====================================================================
        if not self.quiet:
            print("🗺️  Building spatial grid...")
        
        self.grid = GridBuilder.build(config.grid)
        
        if not self.quiet:
            print(f"   ✅ Grid: N={self.grid.N}, dx={self.grid.dx:.4f} km")
        
        # ====================================================================
        # STEP 2: Create Initial Conditions
        # ====================================================================
        if not self.quiet:
            print("🎯 Creating initial conditions...")
        
        U0 = ICBuilder.build(config.initial_conditions, self.grid, config.physics)
        
        if not self.quiet:
            print(f"   ✅ IC type: {config.initial_conditions.type}")
        
        # ====================================================================
        # STEP 3: Initialize Boundary Conditions
        # ====================================================================
        if not self.quiet:
            print("🚪 Initializing boundary conditions...")
        
        self.bc_controller = BCController(config.boundary_conditions, config.physics)
        
        if not self.quiet:
            print(f"   ✅ BC: left={config.boundary_conditions.left.type}, "
                  f"right={config.boundary_conditions.right.type}")
        
        # ====================================================================
        # STEP 4: Initialize State Manager
        # ====================================================================
        if not self.quiet:
            print("💾 Initializing state manager...")
        
        self.state = StateManager(U0, device=config.device)
        
        if not self.quiet:
            print(f"   ✅ State initialized (device={config.device})")
        
        # ====================================================================
        # STEP 5: Initialize Time Stepper
        # ====================================================================
        if not self.quiet:
            print("⏰ Initializing time stepper...")
        
        self.time_stepper = TimeStepper(
            config.physics,
            self.grid,
            self.bc_controller,
            device=config.device
        )
        
        if not self.quiet:
            print("\n✅ Simulation runner initialized successfully!\n")
    
    def run(self) -> Dict[str, Any]:
        """
        Run simulation
        
        Returns:
            Dict with simulation results
        """
        config = self.config
        
        if not self.quiet:
            print(f"🚀 Starting simulation: t_final={config.t_final}s")
            print("=" * 80)
        
        # Main simulation loop
        next_output_time = config.output_dt
        
        while self.state.t < config.t_final and self.state.step_count < config.max_iterations:
            # Get current state
            U = self.state.get_current_state()
            
            # Perform time step
            U_new, dt = self.time_stepper.step(U, self.state.t)
            
            # Update state
            self.state.update_state(U_new)
            self.state.advance_time(dt)
            
            # Check for NaNs
            if self.state.check_for_nans():
                raise ValueError(f"NaN detected at t={self.state.t:.4f}s")
            
            # Store output if needed
            if self.state.t >= next_output_time:
                self.state.sync_from_gpu()
                U_phys = self.state.U[:, self.grid.physical_cell_indices]
                self.state.store_output(U_phys)
                next_output_time += config.output_dt
            
            # Progress output
            if not self.quiet and self.state.step_count % 100 == 0:
                print(f"   Step {self.state.step_count:5d} | t={self.state.t:7.2f}s")
        
        if not self.quiet:
            print("=" * 80)
            print(f"✅ Simulation completed!")
            print(f"   Final time: {self.state.t:.2f}s")
            print(f"   Total steps: {self.state.step_count}")
        
        # Return results
        return {
            'times': np.array(self.state.times),
            'states': np.array(self.state.states),
            'grid': self.grid,
            'config': self.config
        }
    
    # ========================================================================
    # LEGACY COMPATIBILITY (for RL environment)
    # ========================================================================
    
    @property
    def U(self):
        """Legacy: current state (CPU)"""
        return self.state.U
    
    @property
    def t(self):
        """Legacy: current time"""
        return self.state.t
    
    @property
    def times(self):
        """Legacy: times array"""
        return self.state.times
    
    @property
    def states(self):
        """Legacy: states array"""
        return self.state.states


# ============================================================================
# USAGE EXAMPLE: NO YAML!
# ============================================================================

if __name__ == '__main__':
    from arz_model.config.builders import ConfigBuilder
    
    # Build config (NO YAML!)
    config = ConfigBuilder.section_7_6(N=200, t_final=10.0, device='cpu')
    
    # Create runner (NO YAML!)
    runner = SimulationRunner(config)
    
    # Run simulation (NO YAML!)
    results = runner.run()
    
    print(f"\n✅ Simulation completed!")
    print(f"   Times: {len(results['times'])} timesteps")
    print(f"   States: {results['states'].shape}")
```

---

## 📊 COMPARAISON AVANT/APRÈS

### AVANT (AVEC YAML)

```yaml
# configs/section7_6.yml
simulation:
  N: 200
  device: gpu

initial_conditions:
  type: uniform_equilibrium
  rho_m: 0.1
  rho_c: 0.05
  R_val: 10

boundary_conditions:
  left:
    type: inflow
    state: [0.1, 30.0, 0.05, 40.0]
```

```python
# Code Python
from arz_model.core.parameters import ModelParameters
from arz_model.simulation.runner import SimulationRunner

# ❌ Charger YAML (erreurs découvertes au runtime!)
params = ModelParameters()
params.load_from_yaml('configs/section7_6.yml')

# ❌ Pas de validation avant création runner
runner = SimulationRunner(
    scenario_config_path='configs/section7_6.yml',
    base_config_path='configs/base.yml'
)
```

**Problèmes** :
- ❌ Typos dans YAML découverts au runtime
- ❌ Pas d'autocomplétion IDE
- ❌ Pas de validation avant lancement
- ❌ Merge de configs fragile

### APRÈS (SANS YAML)

```python
# Tout en Python pur!
from arz_model.config.builders import ConfigBuilder
from arz_model.simulation.runner import SimulationRunner

# ✅ Config validée par Pydantic (erreurs IMMÉDIATEMENT!)
config = ConfigBuilder.section_7_6(N=200, t_final=1000.0, device='gpu')

# ✅ Validation automatique avant création runner
runner = SimulationRunner(config)

# ✅ Run!
results = runner.run()
```

**Avantages** :
- ✅ **Validation immédiate** : Erreurs détectées AVANT lancement
- ✅ **Autocomplétion IDE** : `config.grid.` → IDE propose `N`, `xmin`, `xmax`, etc.
- ✅ **Type-safe** : `config.grid.N = "abc"` → Erreur de type IMMÉDIATE
- ✅ **Documentation intégrée** : Docstrings, Field descriptions
- ✅ **Pas de merge fragile** : Composition explicite

---

## 🎯 MIGRATION PLAN : YAML → Python

### Étape 1 : Créer les modules config (2h)

```bash
# Créer structure
mkdir -p arz_model/config
touch arz_model/config/__init__.py
touch arz_model/config/grid_config.py
touch arz_model/config/ic_config.py
touch arz_model/config/bc_config.py
touch arz_model/config/physics_config.py
touch arz_model/config/simulation_config.py
touch arz_model/config/builders.py
```

### Étape 2 : Installer Pydantic (1 min)

```bash
pip install pydantic
```

### Étape 3 : Adapter runner.py (1h)

- Modifier `__init__` pour accepter `SimulationConfig` au lieu de YAML paths
- Supprimer `ModelParameters.load_from_yaml()`
- Ajouter backward compatibility properties

### Étape 4 : Adapter ICBuilder, BCController, etc. (1h)

- Modifier pour accepter config Pydantic au lieu de `ModelParameters`

### Étape 5 : Tests (1h)

```python
# test_pydantic_config.py
from arz_model.config.builders import ConfigBuilder
from arz_model.simulation.runner import SimulationRunner

def test_section76_config():
    """Test Section 7.6 config creation"""
    config = ConfigBuilder.section_7_6()
    assert config.grid.N == 200
    assert config.device == 'gpu'
    print("✅ Config creation OK")

def test_runner_with_pydantic():
    """Test runner with Pydantic config"""
    config = ConfigBuilder.simple_test(N=50, t_final=1.0)
    runner = SimulationRunner(config)
    results = runner.run()
    assert len(results['times']) > 0
    print("✅ Runner with Pydantic config OK")

if __name__ == '__main__':
    test_section76_config()
    test_runner_with_pydantic()
```

### Étape 6 : Backward Compatibility (optionnel, 30min)

```python
# arz_model/config/yaml_loader.py (pour legacy code)
def load_config_from_yaml(yaml_path: str) -> SimulationConfig:
    """
    Load YAML and convert to Pydantic config
    
    For backward compatibility with old YAML files
    """
    import yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Convert YAML dict to Pydantic config
    # ... conversion logic ...
    
    return SimulationConfig(**data)
```

---

## 📈 RÉSUMÉ

| Aspect | AVEC YAML (AVANT) | SANS YAML (APRÈS) |
|---|---|---|
| **Validation** | Runtime (tard) | Immédiate (Pydantic) |
| **Type Safety** | ❌ Non | ✅ Oui (type hints) |
| **IDE Support** | ❌ Non | ✅ Autocomplétion complète |
| **Documentation** | ❌ Séparée | ✅ Intégrée (docstrings) |
| **Erreurs typo** | Runtime crash | Compile-time error |
| **Composition** | Merge fragile | ✅ Explicite |
| **Testing** | Difficile | ✅ Facile (unit tests) |
| **Maintenabilité** | ❌ Bas | ✅ Haut |

---

## ✅ PROCHAINES ÉTAPES

**Option A)** 🔥 **Créer TOUT maintenant** - Je crée les 6 modules config + nouveau runner (4-5h)

**Option B)** ⚡ **Créer juste SimulationConfig + Builder** - Minimum viable (2h)

**Option C)** 🧪 **Test d'abord** - On teste Bug 31 avec YAML, puis on migre vers Pydantic

**Dis-moi ce que tu veux !** 🚀
