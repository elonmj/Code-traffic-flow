"""
Urban Traffic Scenarios Package
================================

Collection of real-world urban traffic scenarios for ARZ model validation.

Each scenario module provides:
- Real network topology (from OSM/CSV data)
- Calibrated parameters (from real traffic observations)
- create_grid() function for easy NetworkGrid creation
- Scenario metadata and documentation

Available Scenarios:
--------------------
- lagos_victoria_island: 75 segments, Lagos, Nigeria (September 2024 data)

Usage:
------
    >>> from scenarios.lagos_victoria_island import create_grid
    >>> grid = create_grid()
    >>> grid.initialize()
    >>> for t in range(3600):
    ...     grid.step(dt=0.1)

Architecture:
-------------
This package demonstrates the scalable architecture for managing
100+ scenarios without YAML bloat:

    10 scenarios = 10 Python modules (not 10 YAML files)
    ✅ Version-controlled
    ✅ Type-safe
    ✅ Git-friendly
    ✅ Easy to import and use

Future Scenarios (Roadmap):
----------------------------
- paris_champs_elysees: Paris arterial corridor
- nyc_manhattan_grid: Manhattan street network
- shanghai_yan_an: Shanghai elevated highway
- tokyo_shibuya: Tokyo crossing and surrounding streets
- london_oxford_street: London shopping district
- berlin_unter_den_linden: Berlin boulevard
- mumbai_marine_drive: Mumbai coastal road
- sao_paulo_paulista: São Paulo avenue
- cairo_tahrir: Cairo square and radiating streets
- sydney_george_street: Sydney CBD corridor
"""

__version__ = '1.0.0'
__author__ = 'ARZ Research Team'

# Import scenario creators for convenience
try:
    from .lagos_victoria_island import create_grid as create_lagos_grid
    from .lagos_victoria_island import get_scenario_info as get_lagos_info
    
    __all__ = [
        'create_lagos_grid',
        'get_lagos_info',
    ]
except ImportError:
    # Graceful degradation if scenarios not fully set up
    __all__ = []

# Scenario registry (for discovery and documentation)
SCENARIOS = {
    'lagos_victoria_island': {
        'name': 'Lagos Victoria Island',
        'location': 'Lagos, Nigeria',
        'segments': 75,
        'data_period': 'September 2024',
        'status': 'ready',
        'creator': 'create_lagos_grid'
    },
    # Future scenarios will be added here
}


def list_scenarios():
    """
    List all available scenarios with metadata.
    
    Returns:
        Dictionary of scenario metadata
    """
    return SCENARIOS


def get_scenario(name: str):
    """
    Get scenario creator function by name.
    
    Args:
        name: Scenario name (e.g., 'lagos_victoria_island')
    
    Returns:
        create_grid function for the scenario
    
    Raises:
        ValueError: If scenario not found
    
    Example:
        >>> creator = get_scenario('lagos_victoria_island')
        >>> grid = creator()
    """
    if name not in SCENARIOS:
        available = ', '.join(SCENARIOS.keys())
        raise ValueError(
            f"Scenario '{name}' not found. "
            f"Available scenarios: {available}"
        )
    
    if name == 'lagos_victoria_island':
        return create_lagos_grid
    
    # Future scenarios...
    raise NotImplementedError(f"Scenario '{name}' not yet implemented")
