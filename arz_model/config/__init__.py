"""
Configuration module for ARZ traffic model.

Pydantic-based configuration system.
"""

# ============================================================================
# PYDANTIC CONFIG SYSTEM
# ============================================================================

from .grid_config import GridConfig
from .ic_config import (
    ICConfig,
    InitialConditionsConfig,
    UniformIC,
    UniformEquilibriumIC,
    RiemannIC,
    GaussianPulseIC,
    FileBasedIC
)
from .bc_config import (
    BoundaryConditionsConfig,
    BCState,
    BCScheduleItem,
    InflowBC,
    OutflowBC,
    PeriodicBC,
    ReflectiveBC
)
from .physics_config import PhysicsConfig
from .simulation_config import SimulationConfig
from .network_simulation_config import (
    NetworkSimulationConfig,
    SegmentConfig,
    NodeConfig,
    LinkConfig
)
from .time_config import TimeConfig

__all__ = [
    # Pydantic config
    'SimulationConfig',
    'GridConfig',
    'PhysicsConfig',
    'InitialConditionsConfig',
    'UniformIC',
    'UniformEquilibriumIC',
    'RiemannIC',
    'GaussianPulseIC',
    'FileBasedIC',
    'BoundaryConditionsConfig',
    'BCState',
    'BCScheduleItem',
    'InflowBC',
    'OutflowBC',
    'PeriodicBC',
    'ReflectiveBC',
    
    # Network config (Pydantic)
    'NetworkSimulationConfig',
    'SegmentConfig',
    'NodeConfig',
    'LinkConfig',
]

__version__ = '0.3.0'  # Major update: NetworkGrid Pydantic integration
