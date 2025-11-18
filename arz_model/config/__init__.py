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
from .simulation_config import SimulationConfig  # DEPRECATED - use NetworkSimulationConfig
from .network_simulation_config import (
    NetworkSimulationConfig,
    SegmentConfig,
    NodeConfig,
    LinkConfig
)
from .time_config import TimeConfig
from .config_factory import (
    CityNetworkConfigFactory,
    VictoriaIslandConfigFactory,
    create_victoria_island_config,
    create_city_network_config
)
from .network_config_cache import NetworkConfigCache
from .rl_scenarios import (
    victoria_island_rl_config,
    victoria_island_quick_test,
    victoria_island_extended_training,
    custom_city_rl_config
)

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
    'TimeConfig',
    
    # Network config (Pydantic)
    'NetworkSimulationConfig',
    'SegmentConfig',
    'NodeConfig',
    'LinkConfig',
    
    # Config Factory & Cache
    'CityNetworkConfigFactory',
    'VictoriaIslandConfigFactory',
    'create_victoria_island_config',
    'create_city_network_config',
    'NetworkConfigCache',
    
    # RL Training Scenarios
    'simple_test_config',
    'lagos_training_config',
    'riemann_problem_config',
    'extended_lagos_config',
]

__version__ = '0.3.0'  # Major update: NetworkGrid Pydantic integration
