"""
RL-specific network configuration factory.

Wraps arz_model's create_victoria_island_config() with RL-specific parameters
for training traffic signal control agents.
"""
from typing import List, Dict, Optional
from pathlib import Path
import sys

# Add arz_model to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from arz_model.config.config_factory import create_victoria_island_config
from arz_model.config import NetworkSimulationConfig


class RLNetworkConfig:
    """
    Helper class to extract RL-specific metadata from NetworkSimulationConfig.
    
    Provides easy access to signalized segments and phase mappings for RL environments.
    """
    
    def __init__(self, simulation_config: NetworkSimulationConfig):
        """
        Initialize RL network config helper.
        
        Args:
            simulation_config: NetworkSimulationConfig from arz_model
        """
        self.config = simulation_config
        self._signalized_segment_ids = None
        self._phase_map = None
    
    @property
    def signalized_segment_ids(self) -> List[int]:
        """
        Extract segment IDs that connect to signalized nodes.
        
        Returns:
            List of segment IDs (as integers) for signalized boundaries
        """
        if self._signalized_segment_ids is None:
            self._signalized_segment_ids = []
            
            # Find all signalized nodes
            signalized_nodes = [
                node.node_id for node in self.config.nodes
                if hasattr(node, 'type') and node.type == 'signalized'
            ]
            
            # Find segments that have inflow to signalized nodes
            for segment in self.config.segments:
                # Check if segment flows into a signalized node
                if hasattr(segment, 'end_node') and segment.end_node in signalized_nodes:
                    self._signalized_segment_ids.append(int(segment.segment_id))
        
        return self._signalized_segment_ids
    
    @property
    def phase_map(self) -> Dict[int, str]:
        """
        Default phase mapping for 2-phase traffic signal control.
        
        Returns:
            Dict mapping RL action (0 or 1) to phase name
            
        Note:
            Phase 0: 'green_NS' - North-South green, East-West red
            Phase 1: 'green_EW' - East-West green, North-South red
        """
        if self._phase_map is None:
            self._phase_map = {
                0: 'green_NS',  # North-South green phase
                1: 'green_EW'   # East-West green phase
            }
        
        return self._phase_map
    
    def get_phase_updates(self, phase: int) -> Dict[int, str]:
        """
        Generate phase updates dict for all signalized segments.
        
        Args:
            phase: RL action (0 or 1)
            
        Returns:
            Dict mapping segment_id -> phase_name for use with
            runner.set_boundary_phases_bulk()
            
        Example:
            >>> rl_config = RLNetworkConfig(simulation_config)
            >>> updates = rl_config.get_phase_updates(phase=1)
            >>> runner.set_boundary_phases_bulk(updates)
        """
        phase_name = self.phase_map[phase]
        return {seg_id: phase_name for seg_id in self.signalized_segment_ids}


def create_rl_training_config(
    csv_topology_path: str,
    episode_duration: float = 3600.0,
    decision_interval: float = 15.0,
    observation_segment_ids: Optional[List[str]] = None,
    default_density: float = 20.0,
    default_velocity: float = 50.0,
    inflow_density: float = 30.0,
    inflow_velocity: float = 40.0,
    cells_per_100m: int = 10,
    v_max_m_kmh: float = 100.0,
    v_max_c_kmh: float = 120.0,
    road_quality: float = 0.8,
    alpha: float = 0.35,
    quiet: bool = False
) -> NetworkSimulationConfig:
    """
    Creates a network configuration optimized for RL training.
    
    This function wraps arz_model's create_victoria_island_config() and adds
    RL-specific metadata for training traffic signal control agents.
    
    Args:
        csv_topology_path: Path to network topology CSV file
        episode_duration: Episode maximum time in seconds (default: 3600.0 = 1 hour)
        decision_interval: Time between RL decisions in seconds (default: 15.0)
        observation_segment_ids: Segment IDs to observe (default: first 6)
        default_density: Initial density in veh/km (default: 20.0)
        default_velocity: Initial velocity in km/h (default: 50.0)
        inflow_density: Inflow boundary density in veh/km (default: 30.0)
        inflow_velocity: Inflow boundary velocity in km/h (default: 40.0)
        cells_per_100m: Spatial resolution (default: 10 cells per 100m)
        v_max_m_kmh: Motorcycle max speed in km/h (default: 100.0)
        v_max_c_kmh: Car max speed in km/h (default: 120.0)
        road_quality: Road quality coefficient [0, 1] (default: 0.8)
        alpha: Motorcycle fraction (default: 0.35 = 35%)
        quiet: Suppress output (default: False)
    
    Returns:
        NetworkSimulationConfig with RL metadata attached
        
    Example:
        >>> config = create_rl_training_config(
        ...     csv_topology_path='data/victoria_island_topology.csv',
        ...     episode_duration=1800.0,  # 30 min episodes
        ...     decision_interval=15.0,
        ...     default_density=25.0
        ... )
        >>> env = TrafficSignalEnvDirectV2(simulation_config=config)
    """
    # Generate base configuration using arz_model factory
    config = create_victoria_island_config(
        csv_path=csv_topology_path,
        default_density=default_density,
        default_velocity=default_velocity,
        inflow_density=inflow_density,
        inflow_velocity=inflow_velocity,
        t_final=episode_duration,
        output_dt=decision_interval,
        cells_per_100m=cells_per_100m,
        v_max_m_kmh=v_max_m_kmh,
        v_max_c_kmh=v_max_c_kmh,
        road_quality=road_quality,
        alpha=alpha
    )
    
    # Determine observation segments
    if observation_segment_ids is None:
        # Default: first 6 segments
        observation_segment_ids = [seg.id for seg in config.segments[:6]]
    
    # Attach RL-specific metadata
    # Note: Pydantic strict mode may reject extra fields, so we use a workaround
    # by storing in a dict that can be accessed separately
    rl_metadata = {
        'observation_segment_ids': observation_segment_ids,
        'decision_interval': decision_interval,
        'episode_duration': episode_duration,
        'created_for': 'rl_training'
    }
    
    # Store as custom attribute (Python allows this even on Pydantic models)
    config.rl_metadata = rl_metadata
    
    if not quiet:
        print(f"✅ RL Training Configuration Created:")
        print(f"   Topology: {csv_topology_path}")
        print(f"   Segments: {len(config.segments)}")
        print(f"   Nodes: {len(config.nodes)}")
        print(f"   Episode duration: {episode_duration}s")
        print(f"   Decision interval: {decision_interval}s")
        print(f"   Observation segments: {len(observation_segment_ids)}")
        print(f"   Initial density: {default_density} veh/km")
        print(f"   Motorcycle fraction: {alpha:.0%}")
    
    return config


def create_simple_corridor_config(
    corridor_length: float = 500.0,
    episode_duration: float = 600.0,
    decision_interval: float = 10.0,
    initial_density: float = 30.0,
    initial_velocity: float = 50.0,
    cells_per_100m: int = 10,
    quiet: bool = False
) -> NetworkSimulationConfig:
    """
    Creates a simple single-corridor configuration for quick RL testing.
    
    Useful for debugging and rapid prototyping before scaling to full network.
    
    Args:
        corridor_length: Length of corridor in meters (default: 500.0)
        episode_duration: Episode max time in seconds (default: 600.0 = 10 min)
        decision_interval: Time between RL decisions in seconds (default: 10.0)
        initial_density: Initial density in veh/km (default: 30.0)
        initial_velocity: Initial velocity in km/h (default: 50.0)
        cells_per_100m: Spatial resolution (default: 10)
        quiet: Suppress output (default: False)
        
    Returns:
        NetworkSimulationConfig for single corridor
        
    Example:
        >>> config = create_simple_corridor_config(
        ...     corridor_length=1000.0,  # 1 km
        ...     episode_duration=300.0   # 5 min
        ... )
    """
    from arz_model.config import (
        NetworkSimulationConfig, SegmentConfig, NodeConfig,
        PhysicsConfig, TimeConfig, ICConfig, UniformIC,
        BoundaryConditionsConfig, InflowBC, OutflowBC
    )
    
    N_cells = int(corridor_length / 100.0 * cells_per_100m)
    
    config = NetworkSimulationConfig(
        segments=[
            SegmentConfig(
                id='corridor-0',
                x_min=0.0,
                x_max=corridor_length,
                N=N_cells,
                initial_conditions=ICConfig(
                    config=UniformIC(
                        density=initial_density,
                        velocity=initial_velocity
                    )
                ),
                boundary_conditions=BoundaryConditionsConfig(
                    type='inflow',
                    config=InflowBC(
                        density=initial_density * 1.2,  # Slightly higher inflow
                        velocity=initial_velocity * 0.9
                    )
                ),
                start_node=None,
                end_node='exit-0'
            ),
            SegmentConfig(
                id='corridor-1',
                x_min=0.0,
                x_max=corridor_length,
                N=N_cells,
                initial_conditions=ICConfig(
                    config=UniformIC(
                        density=initial_density,
                        velocity=initial_velocity
                    )
                ),
                boundary_conditions=BoundaryConditionsConfig(
                    type='outflow',
                    config=OutflowBC()
                ),
                start_node='exit-0',
                end_node=None
            )
        ],
        nodes=[
            NodeConfig(
                id='exit-0',
                type='signalized',
                position=[corridor_length, 0.0],
                incoming_segments=['corridor-0'],
                outgoing_segments=['corridor-1'],
                traffic_light_config={
                    'cycle_time': 60.0,
                    'green_time': 30.0
                }
            )
        ],
        physics=PhysicsConfig(
            alpha=0.35,
            rho_max=0.00025,  # 250 veh/km
            V0_m=100.0 / 3.6,  # 100 km/h → m/s
            V0_c=120.0 / 3.6,  # 120 km/h → m/s
            tau_m=10.0,
            tau_c=18.0,
            epsilon=1.0,
            k_m=1.0,
            gamma_m=2.0,
            k_c=1.0,
            gamma_c=2.0,
            weno_order=5,
            weno_ghost_cells=3,
            default_road_quality=0.8
        ),
        time=TimeConfig(
            t_start=0.0,
            t_final=episode_duration,
            output_dt=decision_interval,
            cfl_factor=0.8,
            dt_min=1e-6
        )
    )
    
    # Add RL metadata
    config.rl_metadata = {
        'observation_segment_ids': ['corridor-0', 'corridor-1'],
        'decision_interval': decision_interval,
        'episode_duration': episode_duration,
        'created_for': 'rl_testing_simple_corridor'
    }
    
    if not quiet:
        print(f"✅ Simple Corridor Configuration Created:")
        print(f"   Length: {corridor_length}m")
        print(f"   Cells: {N_cells}")
        print(f"   Episode duration: {episode_duration}s")
        print(f"   Decision interval: {decision_interval}s")
    
    return config
