"""
RL Training Scenario Helpers

Convenience functions for creating NetworkSimulationConfig instances for RL training.

CRITICAL ARCHITECTURE:
- Uses FULL NETWORK with traffic lights (exactly like main_network_simulation.py)
- Reuses create_victoria_island_config() and create_city_network_config()
- NO single-segment simplifications - RL agent must learn traffic light control!
- Traffic lights are at NodeConfig(type="signalized", traffic_light_config={...})

The RL agent controls traffic signals to optimize traffic flow on the complete network.
"""

from .config_factory import create_victoria_island_config, create_city_network_config
from .network_simulation_config import NetworkSimulationConfig
from typing import Optional


def victoria_island_rl_config(
    t_final: float = 450.0,
    output_dt: float = 15.0,
    cells_per_100m: int = 4,
    default_density: float = 20.0,
    inflow_density: float = 30.0,
    use_cache: bool = True
) -> NetworkSimulationConfig:
    """
    Victoria Island network configuration for RL training (Section 7.6).
    
    This is the COMPLETE network with:
    - All road segments from CSV topology
    - All signalized nodes with traffic_light_config
    - OSM-enriched traffic signal data
    - Regional defaults for West Africa
    
    The RL agent learns to control traffic signals at signalized nodes.
    
    Args:
        t_final: Simulation time (seconds, default 7.5 min)
        output_dt: Output interval (seconds, default 15s)
        cells_per_100m: Grid resolution (default 4)
        default_density: Baseline traffic density (veh/km)
        inflow_density: Entry traffic density (veh/km)
        use_cache: Enable config caching
    
    Returns:
        NetworkSimulationConfig with FULL Victoria Island network + traffic lights
    """
    return create_victoria_island_config(
        t_final=t_final,
        output_dt=output_dt,
        cells_per_100m=cells_per_100m,
        default_density=default_density,
        inflow_density=inflow_density,
        use_cache=use_cache
    )


def victoria_island_quick_test(
    t_final: float = 120.0,
    output_dt: float = 5.0,
    cells_per_100m: int = 2,
    use_cache: bool = True
) -> NetworkSimulationConfig:
    """
    Quick test configuration with FULL network but shorter time and coarser grid.
    
    For rapid sanity checks before full training.
    
    Args:
        t_final: Short simulation (default 2 min)
        output_dt: Frequent output (default 5s)
        cells_per_100m: Coarse grid (default 2)
        use_cache: Enable caching
    
    Returns:
        NetworkSimulationConfig with FULL network (fast parameters)
    """
    return create_victoria_island_config(
        t_final=t_final,
        output_dt=output_dt,
        cells_per_100m=cells_per_100m,
        default_density=15.0,  # Light traffic for quick test
        inflow_density=20.0,
        use_cache=use_cache
    )


def victoria_island_extended_training(
    t_final: float = 3600.0,
    output_dt: float = 30.0,
    cells_per_100m: int = 6,
    use_cache: bool = True
) -> NetworkSimulationConfig:
    """
    Extended training configuration with FULL network for longer episodes.
    
    For deep RL training with fine-grained grid.
    
    Args:
        t_final: Long simulation (default 1 hour)
        output_dt: Less frequent output (default 30s)
        cells_per_100m: Fine grid (default 6)
        use_cache: Enable caching
    
    Returns:
        NetworkSimulationConfig with FULL network (production parameters)
    """
    return create_victoria_island_config(
        t_final=t_final,
        output_dt=output_dt,
        cells_per_100m=cells_per_100m,
        default_density=25.0,  # Moderate traffic
        inflow_density=35.0,   # Higher inflow to create congestion
        use_cache=use_cache
    )


def custom_city_rl_config(
    city_name: str,
    csv_path: str,
    enriched_path: Optional[str] = None,
    region: str = 'west_africa',
    t_final: float = 450.0,
    output_dt: float = 15.0,
    cells_per_100m: int = 4,
    use_cache: bool = True,
    **kwargs
) -> NetworkSimulationConfig:
    """
    Generic RL configuration for any city network.
    
    Reuses create_city_network_config() for maximum flexibility.
    
    Args:
        city_name: Name of the city
        csv_path: Path to topology CSV
        enriched_path: Optional path to OSM-enriched data
        region: Traffic light region defaults
        t_final: Simulation time
        output_dt: Output interval
        cells_per_100m: Grid resolution
        use_cache: Enable caching
        **kwargs: Additional factory parameters
    
    Returns:
        NetworkSimulationConfig with FULL network for any city
    """
    return create_city_network_config(
        city_name=city_name,
        csv_path=csv_path,
        enriched_path=enriched_path,
        region=region,
        t_final=t_final,
        output_dt=output_dt,
        cells_per_100m=cells_per_100m,
        use_cache=use_cache,
        **kwargs
    )
