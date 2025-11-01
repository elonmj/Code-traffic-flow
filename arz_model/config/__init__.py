"""
Configuration module for ARZ traffic model.

NEW: Pydantic-based configuration system (replaces YAML)
LEGACY: YAML-based network configuration (deprecated)

Author: ARZ Research Team
Date: 2025-10-26
"""

# ============================================================================
# PYDANTIC CONFIG SYSTEM (NEW - PREFERRED)
# ============================================================================

from arz_model.config.grid_config import GridConfig
from arz_model.config.ic_config import (
    InitialConditionsConfig,
    UniformIC,
    UniformEquilibriumIC,
    RiemannIC,
    GaussianPulseIC,
    FileBasedIC
)
from arz_model.config.bc_config import (
    BoundaryConditionsConfig,
    BCState,
    BCScheduleItem,
    InflowBC,
    OutflowBC,
    PeriodicBC,
    ReflectiveBC
)
from arz_model.config.physics_config import PhysicsConfig
from arz_model.config.simulation_config import SimulationConfig
from arz_model.config.network_simulation_config import (
    NetworkSimulationConfig,
    SegmentConfig,
    NodeConfig,
    LinkConfig
)
from arz_model.config.builders import ConfigBuilder, RLNetworkConfigBuilder

# ============================================================================
# LEGACY YAML CONFIG SYSTEM (DEPRECATED)
# ============================================================================

from .network_config import (
    NetworkConfig,
    NetworkConfigError,
    load_network_config
)

__all__ = [
    # NEW: Pydantic config
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
    'ConfigBuilder',
    
    # NEW: Network config (Pydantic)
    'NetworkSimulationConfig',
    'SegmentConfig',
    'NodeConfig',
    'LinkConfig',
    'RLNetworkConfigBuilder',
    
    # LEGACY: YAML config (deprecated)
    'NetworkConfig',
    'NetworkConfigError',
    'load_network_config'
]

__version__ = '0.3.0'  # Major update: NetworkGrid Pydantic integration
