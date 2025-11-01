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
from arz_model.config.network_simulation_config import (
    NetworkSimulationConfig, SegmentConfig, NodeConfig, LinkConfig
)


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


class RLNetworkConfigBuilder:
    """
    Builder for RL-compatible network configurations.
    
    Provides factory methods to create multi-segment networks optimized
    for reinforcement learning training and evaluation.
    
    This replaces YAML-based NetworkConfig with type-safe Pydantic models,
    avoiding serialization errors and providing better IDE support.
    
    Usage:
        >>> # Quick test (2 segments, 400m)
        >>> config = RLNetworkConfigBuilder.simple_corridor(segments=2)
        >>> env = TrafficSignalEnvDirect(simulation_config=config)
        >>> 
        >>> # Medium network (10 segments, 2km)
        >>> config = RLNetworkConfigBuilder.medium_network(segments=10)
        >>> 
        >>> # Full Lagos network (75 segments, 15km) - Future
        >>> config = RLNetworkConfigBuilder.lagos_network()
    
    Academic Reference:
        - Garavello & Piccoli (2005): "Traffic Flow on Networks"
        - Network formulation replaces single-segment Grid1D approach
    """
    
    @staticmethod
    def simple_corridor(
        segments: int = 2,
        segment_length: float = 200.0,
        N_per_segment: int = 40,
        device: str = 'cpu',
        decision_interval: float = 15.0
    ) -> NetworkSimulationConfig:
        """
        Build simple linear corridor for quick RL testing.
        
        Creates a straight road divided into segments with traffic signals
        at junctions. This solves the "20m domain problem" by providing
        realistic space for congestion to form.
        
        Why this works:
        - 2 segments × 200m = 400m domain (vs 20m single segment)
        - Traffic signal at junction creates bottleneck
        - Queue formation visible → Agent gets learning signal
        - Computation: ~15-20x realtime (vs 280x for single segment)
        
        Args:
            segments: Number of segments (2-10 recommended)
            segment_length: Length of each segment (m)
            N_per_segment: Spatial cells per segment (dx = segment_length/N)
            device: 'cpu' or 'gpu'
            decision_interval: Agent decision interval (s)
        
        Returns:
            NetworkSimulationConfig ready for RL training
        
        Example:
            >>> config = RLNetworkConfigBuilder.simple_corridor(segments=2)
            >>> # Domain: 400m (2 × 200m)
            >>> # Signal: At x=200m (junction between segments)
            >>> # Upstream: seg_0 (0-200m)
            >>> # Downstream: seg_1 (200-400m)
            >>> # Expected: Queue forms during RED, releases during GREEN
        
        Raises:
            ValueError: If segments < 2 or > 10
        """
        if segments < 2:
            raise ValueError("Need at least 2 segments for corridor (junction required)")
        if segments > 10:
            raise ValueError("Use medium_network() for >10 segments")
        
        # Build segments dictionary
        seg_configs = {}
        for i in range(segments):
            x_min = i * segment_length
            x_max = (i + 1) * segment_length
            
            # First segment: no start_node (boundary)
            # Last segment: no end_node (boundary)
            # Middle segments: connect to junction nodes
            start_node = None if i == 0 else f'node_{i}'
            end_node = None if i == segments - 1 else f'node_{i+1}'
            
            seg_configs[f'seg_{i}'] = SegmentConfig(
                x_min=x_min,
                x_max=x_max,
                N=N_per_segment,
                start_node=start_node,
                end_node=end_node
            )
        
        # Build nodes dictionary
        node_configs = {}
        
        # NOTE: Do NOT create boundary nodes as Node objects!
        # NetworkGrid handles boundaries through segment endpoints automatically.
        # Only create intermediate nodes (junctions with traffic signals).
        
        # Intermediate nodes (traffic signals at ALL junctions)
        for i in range(1, segments):
            # ✅ FIX (2025-01-XX): Use segment_id for green_segments, not geographic orientations
            # The node_solver._calculate_outgoing_flux() checks:
            #     has_green_light = segment_id in green_segments
            # So green_segments must contain actual segment IDs ('seg_0', 'seg_1')
            # NOT geographic orientations ('north', 'south')
            outgoing_seg = f'seg_{i}'  # The segment that gets the green light
            
            node_configs[f'node_{i}'] = NodeConfig(
                type='signalized',
                position=[i * segment_length, 0.0],
                incoming_segments=[f'seg_{i-1}'],
                outgoing_segments=[outgoing_seg],
                traffic_light_config={
                    'cycle_time': 60.0,     # 60s cycle time
                    'green_time': 25.0,     # 25s green phase
                    'yellow_time': 3.0,     # 3s yellow transition
                    'all_red_time': 2.0,    # 2s all-red clearance
                    'phases': [
                        {
                            'id': 0, 
                            'name': 'RED',  # ✅ FIX: Phase 0 = RED (action 0 = RED in env)
                            'duration': 30.0,
                            'green_segments': [],  # No green segments during RED
                            'yellow_segments': []
                        },
                        {
                            'id': 1, 
                            'name': 'GREEN',  # ✅ FIX: Phase 1 = GREEN (action 1 = GREEN in env)
                            'duration': 25.0,
                            'green_segments': [outgoing_seg],  # Use actual segment_id
                            'yellow_segments': []
                        }
                    ]
                }
            )
        
        # Build links (sequential connections)
        link_configs = []
        for i in range(segments - 1):
            link_configs.append(LinkConfig(
                from_segment=f'seg_{i}',
                to_segment=f'seg_{i+1}',
                via_node=f'node_{i+1}',
                coupling_type='theta_k'  # Garavello & Piccoli junction coupling
            ))
        
        # Controlled nodes (all intermediate signalized nodes)
        controlled_nodes = [f'node_{i}' for i in range(1, segments)]
        
        # Define global parameters (physics + initial conditions)
        global_params = {
            # Physics parameters
            'rho_jam': 0.2,  # jam density (veh/m)
            'gamma_m': 2.0,
            'gamma_c': 2.0,
            'K_m': 20.0 / 3.6,  # critical speed motorcycles (m/s)
            'K_c': 20.0 / 3.6,  # critical speed cars (m/s)
            'tau_m': 1.0,  # relaxation time motorcycles (s)
            'tau_c': 1.0,  # relaxation time cars (s)
            'V_creeping': 0.1,  # creeping speed (m/s)
            'red_light_factor': 0.05,  # ✅ FIX: Strong blocking (5% flow) during RED signal - more numerically stable than 0.01
            
            # Initial conditions (uniform equilibrium - moderate traffic)
            'rho_m_init': 0.08,  # 8% of jam density (40 veh/km motorcycles)
            'rho_c_init': 0.04,  # 4% of jam density (20 veh/km cars)
            'V0_m': 8.89,  # Free-flow speed motos (m/s) = 32 km/h (for IC)
            'V0_c': 13.89,  # Free-flow speed cars (m/s) = 50 km/h (for IC)
        }
        
        # ✅ FIX (2025-10-29): Add boundary conditions to enable traffic inflow
        # Without BC, network is CLOSED → no congestion formation → RL can't learn!
        # Enable continuous traffic inflow → congestion forms → rewards vary
        boundary_conditions = BoundaryConditionsConfig(
            left=InflowBC(
                state=BCState(
                    rho_m=0.12,    # 60 veh/km inflow (realistic urban demand)
                    w_m=10.668,    # w_m = v_m + p_m (ARZ Lagrangian variable)
                                   # With rho_m=0.12, rho_c=0.06: p_m ≈ 1.78
                                   # v_m = 8.89 m/s → w_m = 8.89 + 1.78 = 10.67
                    rho_c=0.06,    # 30 veh/km inflow
                    w_c=15.335     # w_c = v_c + p_c
                                   # With rho_c=0.06: p_c ≈ 1.45
                                   # v_c = 13.89 m/s → w_c = 13.89 + 1.45 = 15.34
                )
            ),
            right=OutflowBC(
                state=BCState(
                    rho_m=0.08, w_m=8.89,  # Matches equilibrium IC
                    rho_c=0.04, w_c=13.89
                )
            )
        )
        
        # Create config
        config = NetworkSimulationConfig(
            segments=seg_configs,
            nodes=node_configs,
            links=link_configs,
            dt=0.1,              # 100ms timestep for numerical stability
            t_final=3600.0,      # 1 hour simulation
            output_dt=1.0,       # 1s output interval
            device=device,
            decision_interval=decision_interval,  # 15s agent decisions
            controlled_nodes=controlled_nodes,
            global_params=global_params,  # Physics + IC
            boundary_conditions=boundary_conditions  # ✅ BC for traffic inflow/outflow
        )
        
        # Validate topology
        config.validate_network_topology()
        
        return config
    
    @staticmethod
    def medium_network(
        segments: int = 10,
        segment_length: float = 200.0,
        N_per_segment: int = 40,
        device: str = 'cpu'
    ) -> NetworkSimulationConfig:
        """
        Build medium-sized urban network (10-20 segments, 2-4km).
        
        Creates a more complex corridor with:
        - Multiple traffic signals (every segment junction)
        - Realistic urban corridor dimensions
        - Signal coordination opportunities
        
        Args:
            segments: Number of segments (10-20 recommended)
            segment_length: Average segment length (m)
            N_per_segment: Spatial cells per segment
            device: 'cpu' or 'gpu'
        
        Returns:
            NetworkSimulationConfig for medium network training
        
        Example:
            >>> config = RLNetworkConfigBuilder.medium_network(segments=10)
            >>> # Domain: 2km (10 × 200m)
            >>> # Signals: 9 junctions (every 200m)
            >>> # Challenge: Coordinate signals to minimize delay
        
        Note:
            For now, uses simple_corridor() with more segments.
            Future: Add grid/tree topology variants.
        """
        if segments < 10:
            raise ValueError("Use simple_corridor() for <10 segments")
        if segments > 20:
            raise ValueError("Medium network supports up to 20 segments")
        
        # Use simple_corridor with more segments
        # TODO: Add more complex topologies (grid, tree, etc.)
        return RLNetworkConfigBuilder.simple_corridor(
            segments=segments,
            segment_length=segment_length,
            N_per_segment=N_per_segment,
            device=device
        )
    
    @staticmethod
    def lagos_network(
        csv_path: str = 'donnees_trafic_75_segments (2).csv',
        device: str = 'cpu'
    ) -> NetworkSimulationConfig:
        """
        Build Lagos Victoria Island network from TomTom data.
        
        Creates full 75-segment urban network with:
        - Real street topology (Akin Adesola, Adeola Odeku, etc.)
        - Calibrated parameters from speed data
        - Multiple arterial/residential road types
        
        Args:
            csv_path: Path to Lagos network CSV
            device: 'cpu' or 'gpu'
        
        Returns:
            NetworkSimulationConfig for Lagos network
        
        Note:
            This requires CSV file and NetworkBuilder conversion.
            For testing, use simple_corridor(segments=2) first.
            Once working, implement Lagos conversion (75 segments).
        
        Raises:
            NotImplementedError: Lagos conversion pending
            FileNotFoundError: If CSV not found
        """
        import os
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Lagos CSV not found: {csv_path}\n"
                f"Use simple_corridor() or medium_network() for testing"
            )
        
        # TODO: Implement NetworkGrid → NetworkSimulationConfig conversion
        # This requires extracting segments/nodes/links from existing NetworkGrid
        raise NotImplementedError(
            "Lagos network Pydantic conversion pending.\n\n"
            "Current status:\n"
            "✅ NetworkGrid exists (75 segments from CSV)\n"
            "✅ NetworkSimulationConfig Pydantic model exists\n"
            "⏳ Conversion logic needed\n\n"
            "For now, use:\n"
            "  - simple_corridor(segments=2) for quick testing (400m)\n"
            "  - medium_network(segments=10) for validation (2km)\n\n"
            "Once these work, we'll add Lagos conversion (75 segments, 15km)."
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
    
    # Build network configs (NEW!)
    print("\n" + "="*70)
    print("NETWORK CONFIGURATION EXAMPLES")
    print("="*70)
    
    # Simple corridor (2 segments, 400m)
    net_config = RLNetworkConfigBuilder.simple_corridor(segments=2)
    print("\n✅ Simple corridor config created:")
    print(f"   Segments: {len(net_config.segments)}")
    print(f"   Nodes: {len(net_config.nodes)}")
    print(f"   Links: {len(net_config.links)}")
    print(f"   Domain: {len(net_config.segments) * 200}m")
    print(f"   Controlled nodes: {net_config.controlled_nodes}")
    
    # Medium network (10 segments, 2km)
    net_config_medium = RLNetworkConfigBuilder.medium_network(segments=10)
    print("\n✅ Medium network config created:")
    print(f"   Segments: {len(net_config_medium.segments)}")
    print(f"   Nodes: {len(net_config_medium.nodes)}")
    print(f"   Domain: {len(net_config_medium.segments) * 200}m")
    print(f"   Controlled nodes: {len(net_config_medium.controlled_nodes)} signals")
