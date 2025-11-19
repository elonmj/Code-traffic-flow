"""
Configuration Factory for Victoria Island Corridor Network.

This module provides a reusable system for automatically generating complete
NetworkSimulationConfig objects from topology CSV files. It embodies a "global
reflection" approach where the entire network is analyzed to intelligently
determine boundary conditions, node types, and connectivity.

The factory eliminates manual segment-by-segment configuration by:
1. Reading network topology from CSV
2. Analyzing graph structure to identify entry/exit points and junctions
3. Automatically creating SegmentConfig for all edges
4. Automatically creating NodeConfig for all nodes with correct types
5. Applying intelligent default boundary and initial conditions

This is a REUSABLE system that can work with any corridor network.
"""
import os
import pandas as pd
import networkx as nx
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path

from arz_model.config import (
    NetworkSimulationConfig,
    TimeConfig,
    PhysicsConfig,
    SegmentConfig,
    NodeConfig,
    BoundaryConditionsConfig,
    InflowBC,
    OutflowBC,
    UniformIC,
    ICConfig
)
from arz_model.config.network_config_cache import NetworkConfigCache

# Regional traffic light defaults
REGIONAL_TRAFFIC_LIGHT_DEFAULTS = {
    'west_africa': {
        'cycle_time': 90.0,  # Lagos standard
        'green_time': 35.0,
        'amber_time': 3.0,
        'red_time': 52.0
    },
    'europe': {
        'cycle_time': 120.0,
        'green_time': 50.0,
        'amber_time': 3.0,
        'red_time': 67.0
    },
    'north_america': {
        'cycle_time': 100.0,
        'green_time': 40.0,
        'amber_time': 4.0,
        'red_time': 56.0
    },
    'asia': {
        'cycle_time': 150.0,
        'green_time': 60.0,
        'amber_time': 3.0,
        'red_time': 87.0
    }
}


class CityNetworkConfigFactory:
    """
    Factory for generating complete network simulation configurations from CSV topology.
    
    This factory implements a "global reflection" approach where the entire network
    structure is analyzed to make intelligent decisions about boundary conditions,
    node types, and simulation parameters.
    
    Now supports multiple cities, OSM integration, and intelligent caching.
    """
    
    def __init__(
        self,
        city_name: str,
        csv_path: str,
        enriched_path: Optional[str] = None,  # Path to OSM-enriched Excel file
        region: str = 'west_africa',  # Traffic light region defaults
        use_cache: bool = True,  # Enable/disable caching
        default_density: float = 20.0,  # veh/km - light traffic baseline
        default_velocity: float = 50.0,  # km/h - moderate speed baseline
        inflow_density: float = 30.0,  # veh/km - slightly higher at entry
        inflow_velocity: float = 40.0,  # km/h - slower at entry
        outflow_density: float = 15.0,  # veh/km - lighter at exit
        outflow_velocity: float = 60.0,  # km/h - faster at exit
        # --- STRATEGIC PARAMETERS FOR FEASIBLE SIMULATION ---
        # These have been adjusted from scientific-grade to practical-grade
        # to ensure completion within Kaggle's 9-hour limit.
        
        cells_per_100m: int = 4,      # Grid resolution: 4 cells/100m (dx=25m). Coarser but faster.
        t_final: float = 450.0,       # 7.5 minutes simulation. Sufficient to observe dynamics.
        output_dt: float = 15.0,      # Output every 15s -> 30 frames for animation.
        cfl_factor: float = 0.8,      # Aggressive but stable CFL for speed.
        v_max_m_kmh: float = 100.0,   # Max speed motorcycles
        v_max_c_kmh: float = 120.0,   # Max speed cars
        road_quality: float = 0.8     # Good road quality (0-1 scale)
    ):
        """
        Initialize the configuration factory.
        
        Args:
            city_name: Name of the city (e.g., "Victoria Island", "Paris")
            csv_path: Path to the topology CSV file
            enriched_path: Optional path to OSM-enriched Excel file
            region: Regional defaults for traffic lights (west_africa, europe, asia, north_america)
            use_cache: Enable caching to avoid regenerating configs
            default_density: Default initial density for all segments (veh/km)
            default_velocity: Default initial velocity for all segments (km/h)
            inflow_density: Density for inflow boundary conditions (veh/km)
            inflow_velocity: Velocity for inflow boundary conditions (km/h)
            outflow_density: Density for outflow boundary conditions (veh/km)
            outflow_velocity: Velocity for outflow boundary conditions (km/h)
            cells_per_100m: Number of grid cells per 100 meters
            t_final: Total simulation time (seconds)
            output_dt: Output interval (seconds)
            cfl_factor: CFL factor for time stepping
            v_max_m_kmh: Maximum motorcycle speed (km/h)
            v_max_c_kmh: Maximum car speed (km/h)
            road_quality: Road quality coefficient (0-1)
        """
        self.city_name = city_name
        self.csv_path = Path(csv_path)
        self.enriched_path = Path(enriched_path) if enriched_path else None
        self.region = region
        self.use_cache = use_cache
        
        # Simulation parameters
        self.default_density = default_density
        self.default_velocity = default_velocity
        self.inflow_density = inflow_density
        self.inflow_velocity = inflow_velocity
        self.outflow_density = outflow_density
        self.outflow_velocity = outflow_velocity
        self.cells_per_100m = cells_per_100m
        self.t_final = t_final
        self.output_dt = output_dt
        self.cfl_factor = cfl_factor
        self.v_max_m_kmh = v_max_m_kmh
        self.v_max_c_kmh = v_max_c_kmh
        self.road_quality = road_quality
        
        # Cache manager
        self.cache = NetworkConfigCache() if use_cache else None
        
        # Data structures populated during analysis
        self.df: Optional[pd.DataFrame] = None
        self.graph: Optional[nx.DiGraph] = None
        self.entry_nodes: Set[str] = set()
        self.exit_nodes: Set[str] = set()
        self.junction_nodes: Set[str] = set()
        self.signalized_nodes: Set[str] = set()  # OSM traffic signals
        
    def _load_topology(self) -> None:
        """Load and validate the topology CSV file."""
        print(f"   📊 Loading topology from: {self.csv_path}", flush=True)
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Topology CSV not found: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        
        # Validate required columns
        required_cols = ['u', 'v', 'length']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        
        # Convert node IDs to strings for consistency
        self.df['u'] = self.df['u'].astype(str)
        self.df['v'] = self.df['v'].astype(str)
        
        print(f"   ✅ Loaded {len(self.df)} edges from topology", flush=True)
        
    def _build_graph(self) -> None:
        """Build NetworkX directed graph from topology data."""
        print("   🔗 Building directed graph...", flush=True)
        
        self.graph = nx.DiGraph()
        
        # Add edges with length attribute
        for _, row in self.df.iterrows():
            u, v, length = row['u'], row['v'], row['length']
            self.graph.add_edge(u, v, length=length, data=row.to_dict())
        
        print(f"   ✅ Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges", flush=True)
        
    def _analyze_network_structure(self) -> None:
        """
        Analyze graph structure to identify entry points, exit points, and junctions.
        
        This is the "global reflection" logic:
        - Entry nodes: in-degree = 0 (no incoming edges, traffic enters here)
        - Exit nodes: out-degree = 0 (no outgoing edges, traffic exits here)
        - Junction nodes: in-degree > 1 OR out-degree > 1 (merges/diverges)
        """
        print("   🧠 Analyzing network structure (global reflection)...", flush=True)
        
        # Identify entry and exit points
        for node in self.graph.nodes():
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            
            if in_degree == 0:
                self.entry_nodes.add(node)
            if out_degree == 0:
                self.exit_nodes.add(node)
            if in_degree > 1 or out_degree > 1:
                self.junction_nodes.add(node)
        
        print(f"   ✅ Network analysis complete:", flush=True)
        print(f"      - Entry points: {len(self.entry_nodes)}", flush=True)
        print(f"      - Exit points: {len(self.exit_nodes)}", flush=True)
        print(f"      - Junctions: {len(self.junction_nodes)}", flush=True)
        print(f"      - Simple pass-through nodes: "
              f"{self.graph.number_of_nodes() - len(self.entry_nodes) - len(self.exit_nodes) - len(self.junction_nodes)}",
              flush=True)
    
    def _load_osm_signalized_nodes(self) -> Set[str]:
        """
        Load signalized nodes from OSM-enriched Excel file.
        
        Returns:
            Set of node IDs that have traffic signals
        """
        if not self.enriched_path or not self.enriched_path.exists():
            return set()
        
        try:
            # Read enriched data
            df_enriched = pd.read_excel(self.enriched_path)
            
            signalized = set()
            
            # Check for has_signal columns
            if 'u_has_signal' in df_enriched.columns:
                # Get u nodes that have signals
                u_signals = df_enriched[df_enriched['u_has_signal'] == True]['u'].astype(str).unique()
                signalized.update(u_signals)
            
            if 'v_has_signal' in df_enriched.columns:
                # Get v nodes that have signals
                v_signals = df_enriched[df_enriched['v_has_signal'] == True]['v'].astype(str).unique()
                signalized.update(v_signals)
            
            if signalized:
                print(f"   🚦 Detected {len(signalized)} signalized nodes from OSM data", flush=True)
            
            return signalized
        
        except Exception as e:
            print(f"   ⚠️  Could not load OSM signals: {e}", flush=True)
            return set()
    
    def _create_traffic_light_config(self, node_id: str) -> Dict[str, Any]:
        """
        Create traffic light configuration with regional defaults.
        
        Args:
            node_id: Node identifier
        
        Returns:
            Dictionary with traffic light timing parameters
        """
        defaults = REGIONAL_TRAFFIC_LIGHT_DEFAULTS.get(self.region, REGIONAL_TRAFFIC_LIGHT_DEFAULTS['west_africa'])
        
        return {
            'cycle_time': defaults['cycle_time'],
            'green_time': defaults['green_time'],
            'amber_time': defaults['amber_time'],
            'red_time': defaults['red_time'],
            'initial_phase': 'green'  # Start in green phase
        }
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get factory parameters for fingerprinting.
        
        Returns:
            Dictionary of all parameters that affect config generation
        """
        return {
            'default_density': self.default_density,
            'default_velocity': self.default_velocity,
            'inflow_density': self.inflow_density,
            'inflow_velocity': self.inflow_velocity,
            'outflow_density': self.outflow_density,
            'outflow_velocity': self.outflow_velocity,
            'cells_per_100m': self.cells_per_100m,
            't_final': self.t_final,
            'output_dt': self.output_dt,
            'cfl_factor': self.cfl_factor,
            'v_max_m_kmh': self.v_max_m_kmh,
            'v_max_c_kmh': self.v_max_c_kmh,
            'road_quality': self.road_quality,
            'region': self.region
        }
        
    def _create_segment_config(self, u: str, v: str, edge_data: Dict) -> SegmentConfig:
        """
        Create a SegmentConfig for a single edge.
        
        Args:
            u: Source node ID
            v: Target node ID
            edge_data: Edge attributes from the graph
            
        Returns:
            Configured SegmentConfig object
        """
        length = edge_data['length']
        
        # Calculate number of cells based on resolution
        N = max(10, int(length / 100.0 * self.cells_per_100m))
        
        # Create segment ID
        seg_id = f"{u}->{v}"
        
        # CRITICAL FIX: A segment's start_node and end_node should ALWAYS be set
        # to the actual node IDs from the graph. The boundary conditions handle
        # entry/exit behavior, not the node connectivity itself.
        # Setting these to None breaks the topology validation.
        start_node = u
        end_node = v
        
        # Initial conditions: uniform baseline
        ic = ICConfig(config=UniformIC(
            density=self.default_density,
            velocity=self.default_velocity
        ))
        
        # Boundary conditions logic based on network position
        if u in self.entry_nodes:
            # Entry segment: inflow on left
            left_bc = InflowBC(density=self.inflow_density, velocity=self.inflow_velocity)
        else:
            # Internal segment: node will handle left boundary
            left_bc = OutflowBC(density=self.default_density, velocity=self.default_velocity)
        
        if v in self.exit_nodes:
            # Exit segment: outflow on right
            right_bc = OutflowBC(density=self.outflow_density, velocity=self.outflow_velocity)
        else:
            # Internal segment: node will handle right boundary
            right_bc = OutflowBC(density=self.default_density, velocity=self.default_velocity)
        
        bc = BoundaryConditionsConfig(left=left_bc, right=right_bc)
        
        return SegmentConfig(
            id=seg_id,
            x_min=0.0,
            x_max=length,
            N=N,
            start_node=start_node,
            end_node=end_node,
            initial_conditions=ic,
            boundary_conditions=bc
        )
    
    def _create_node_config(self, node_id: str) -> NodeConfig:
        """
        Create a NodeConfig for a single node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Configured NodeConfig object
        """
        # Get incoming and outgoing edges
        incoming = [f"{pred}->{node_id}" for pred in self.graph.predecessors(node_id)]
        outgoing = [f"{node_id}->{succ}" for succ in self.graph.successors(node_id)]
        
        # Determine node type based on structure and OSM data
        if node_id in self.signalized_nodes:
            node_type = "signalized"
            traffic_light_config = self._create_traffic_light_config(node_id)
        elif node_id in self.entry_nodes or node_id in self.exit_nodes:
            node_type = "boundary"
            traffic_light_config = None
        elif node_id in self.junction_nodes:
            node_type = "junction"
            traffic_light_config = None
        else:
            node_type = "junction"  # Simple pass-through treated as junction
            traffic_light_config = None
        
        return NodeConfig(
            id=node_id,
            type=node_type,
            incoming_segments=incoming,
            outgoing_segments=outgoing,
            traffic_light_config=traffic_light_config
        )
    
    def create_config(self) -> NetworkSimulationConfig:
        """
        Generate complete NetworkSimulationConfig from CSV topology.
        
        This is the main factory method that orchestrates the entire configuration
        generation process using "global reflection" logic.
        
        Returns:
            Complete, validated NetworkSimulationConfig ready for simulation
        """
        # Check cache first
        if self.cache:
            fingerprint = self.cache.compute_fingerprint(
                csv_path=self.csv_path,
                enriched_path=self.enriched_path,
                factory_params=self.get_params()
            )
            
            cached_config = self.cache.load(self.city_name, fingerprint)
            if cached_config is not None:
                return cached_config
            
            print(f"   🔄 Cache MISS: Generating config from scratch", flush=True)
        
        print("\n" + "=" * 70, flush=True)
        print(f"🏭 {self.city_name.upper()} CONFIG FACTORY - GLOBAL CONFIGURATION GENERATION", flush=True)
        print("=" * 70, flush=True)
        
        # Step 1: Load topology data
        self._load_topology()
        
        # Step 2: Build graph representation
        self._build_graph()
        
        # Step 3: Analyze network structure (global reflection)
        self._analyze_network_structure()
        
        # Step 3.5: Load OSM signalized nodes
        self.signalized_nodes = self._load_osm_signalized_nodes()
        
        # Step 4: Generate segment configurations
        print("\n   🔧 Generating segment configurations...", flush=True)
        segments = []
        for u, v, edge_data in self.graph.edges(data=True):
            seg_config = self._create_segment_config(u, v, edge_data)
            segments.append(seg_config)
        print(f"   ✅ Created {len(segments)} segment configurations", flush=True)
        
        # Step 5: Generate node configurations
        print("\n   🔧 Generating node configurations...", flush=True)
        nodes = []
        for node_id in self.graph.nodes():
            # Skip nodes with no connections (shouldn't happen in a proper network)
            if self.graph.degree(node_id) == 0:
                continue
            node_config = self._create_node_config(node_id)
            nodes.append(node_config)
        print(f"   ✅ Created {len(nodes)} node configurations", flush=True)
        
        # Step 6: Create time and physics configurations
        print("\n   ⚙️  Setting up time and physics parameters...", flush=True)
        time_config = TimeConfig(
            t_final=self.t_final,
            output_dt=self.output_dt,
            cfl_factor=self.cfl_factor
        )
        
        physics_config = PhysicsConfig(
            v_max_m_kmh=self.v_max_m_kmh,
            v_max_c_kmh=self.v_max_c_kmh,
            default_road_quality=self.road_quality
        )
        
        # Step 7: Assemble complete configuration
        print("\n   🔨 Assembling complete network configuration...", flush=True)
        network_config = NetworkSimulationConfig(
            time=time_config,
            physics=physics_config,
            segments=segments,
            nodes=nodes
        )
        
        # Save to cache if enabled
        if self.cache:
            self.cache.save(
                config=network_config,
                city_name=self.city_name,
                fingerprint=fingerprint,
                csv_path=self.csv_path,
                enriched_path=self.enriched_path,
                factory_params=self.get_params()
            )
        
        print("\n" + "=" * 70, flush=True)
        print("✅ CONFIGURATION GENERATION COMPLETE", flush=True)
        print("=" * 70, flush=True)
        print(f"   Total Segments: {len(segments)}", flush=True)
        print(f"   Total Nodes: {len(nodes)}", flush=True)
        print(f"   Entry Points: {len(self.entry_nodes)}", flush=True)
        print(f"   Exit Points: {len(self.exit_nodes)}", flush=True)
        print(f"   Junctions: {len(self.junction_nodes)}", flush=True)
        print(f"   Signalized Nodes: {len(self.signalized_nodes)}", flush=True)
        print(f"   Simulation Duration: {self.t_final}s ({self.t_final/60:.1f} min)", flush=True)
        print(f"   Grid Resolution: {self.cells_per_100m} cells/100m", flush=True)
        print("=" * 70 + "\n", flush=True)
        
        return network_config


# Alias for backward compatibility
VictoriaIslandConfigFactory = CityNetworkConfigFactory


def create_city_network_config(
    city_name: str,
    csv_path: str,
    enriched_path: Optional[str] = None,
    **kwargs
) -> NetworkSimulationConfig:
    """
    Create a network configuration for any city.
    
    Args:
        city_name: Name of the city (e.g., "Victoria Island", "Paris")
        csv_path: Path to topology CSV file
        enriched_path: Optional path to OSM-enriched Excel file
        **kwargs: Additional parameters to pass to CityNetworkConfigFactory
        
    Returns:
        Complete NetworkSimulationConfig ready for simulation
        
    Example:
        >>> # Victoria Island with OSM data
        >>> config = create_city_network_config(
        ...     city_name="Victoria Island",
        ...     csv_path="data/victoria_island_topology.csv",
        ...     enriched_path="data/fichier_de_travail_complet_enriched.xlsx"
        ... )
        >>> 
        >>> # Paris with custom parameters
        >>> config = create_city_network_config(
        ...     city_name="Paris",
        ...     csv_path="data/paris_topology.csv",
        ...     region='europe',
        ...     v_max_c_kmh=130.0
        ... )
    """
    factory = CityNetworkConfigFactory(
        city_name=city_name,
        csv_path=csv_path,
        enriched_path=enriched_path,
        **kwargs
    )
    return factory.create_config()


def create_victoria_island_config(
    csv_path: Optional[str] = None,
    enriched_path: Optional[str] = None,
    **kwargs
) -> NetworkSimulationConfig:
    """
    Convenience function to create a Victoria Island network configuration.
    
    Args:
        csv_path: Path to topology CSV. If None, uses default location.
        enriched_path: Optional path to OSM-enriched Excel file
        **kwargs: Additional parameters to pass to CityNetworkConfigFactory
        
    Returns:
        Complete NetworkSimulationConfig ready for simulation
        
    Example:
        >>> config = create_victoria_island_config()
        >>> # Or with OSM data:
        >>> config = create_victoria_island_config(
        ...     enriched_path="data/fichier_de_travail_complet_enriched.xlsx"
        ... )
        >>> # Or with custom parameters:
        >>> config = create_victoria_island_config(
        ...     default_density=30.0,
        ...     t_final=3600.0,
        ...     cells_per_100m=20
        ... )
    """
    if csv_path is None:
        # Default path relative to this file
        config_dir = Path(__file__).parent
        csv_path = config_dir.parent / 'data' / 'fichier_de_travail_corridor_utf8.csv'
    
    if enriched_path is None:
        # Try default enriched file location
        config_dir = Path(__file__).parent
        # Updated to point to the correct file (corridor) which exists
        default_enriched = config_dir.parent / 'data' / 'fichier_de_travail_corridor_enriched.xlsx'
        if default_enriched.exists():
            enriched_path = str(default_enriched)
    
    return create_city_network_config(
        city_name="Victoria Island",
        csv_path=str(csv_path),
        enriched_path=enriched_path,
        region='west_africa',
        **kwargs
    )
