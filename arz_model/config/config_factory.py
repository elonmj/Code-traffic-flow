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
from typing import Dict, List, Set, Optional, Tuple
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


class VictoriaIslandConfigFactory:
    """
    Factory for generating complete network simulation configurations from CSV topology.
    
    This factory implements a "global reflection" approach where the entire network
    structure is analyzed to make intelligent decisions about boundary conditions,
    node types, and simulation parameters.
    """
    
    def __init__(
        self,
        csv_path: str,
        default_density: float = 20.0,  # veh/km - light traffic baseline
        default_velocity: float = 50.0,  # km/h - moderate speed baseline
        inflow_density: float = 30.0,  # veh/km - slightly higher at entry
        inflow_velocity: float = 40.0,  # km/h - slower at entry
        outflow_density: float = 15.0,  # veh/km - lighter at exit
        outflow_velocity: float = 60.0,  # km/h - faster at exit
        cells_per_100m: int = 10,  # Grid resolution: 10 cells per 100m
        t_final: float = 1800.0,  # 30 minutes simulation
        output_dt: float = 10.0,  # Output every 10 seconds
        cfl_factor: float = 0.4,  # Conservative CFL for stability
        v_max_m_kmh: float = 100.0,  # Max speed motorcycles
        v_max_c_kmh: float = 120.0,  # Max speed cars
        road_quality: float = 0.8  # Good road quality (0-1 scale)
    ):
        """
        Initialize the configuration factory.
        
        Args:
            csv_path: Path to the topology CSV file
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
        self.csv_path = Path(csv_path)
        
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
        
        # Data structures populated during analysis
        self.df: Optional[pd.DataFrame] = None
        self.graph: Optional[nx.DiGraph] = None
        self.entry_nodes: Set[str] = set()
        self.exit_nodes: Set[str] = set()
        self.junction_nodes: Set[str] = set()
        
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
        
        # Determine node type based on structure
        if node_id in self.entry_nodes or node_id in self.exit_nodes:
            node_type = "boundary"
        elif node_id in self.junction_nodes:
            node_type = "junction"
        else:
            node_type = "boundary"  # Simple pass-through treated as boundary
        
        return NodeConfig(
            id=node_id,
            type=node_type,
            incoming_segments=incoming,
            outgoing_segments=outgoing
        )
    
    def create_config(self) -> NetworkSimulationConfig:
        """
        Generate complete NetworkSimulationConfig from CSV topology.
        
        This is the main factory method that orchestrates the entire configuration
        generation process using "global reflection" logic.
        
        Returns:
            Complete, validated NetworkSimulationConfig ready for simulation
        """
        print("\n" + "=" * 70, flush=True)
        print("🏭 VICTORIA ISLAND CONFIG FACTORY - GLOBAL CONFIGURATION GENERATION", flush=True)
        print("=" * 70, flush=True)
        
        # Step 1: Load topology data
        self._load_topology()
        
        # Step 2: Build graph representation
        self._build_graph()
        
        # Step 3: Analyze network structure (global reflection)
        self._analyze_network_structure()
        
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
        
        print("\n" + "=" * 70, flush=True)
        print("✅ CONFIGURATION GENERATION COMPLETE", flush=True)
        print("=" * 70, flush=True)
        print(f"   Total Segments: {len(segments)}", flush=True)
        print(f"   Total Nodes: {len(nodes)}", flush=True)
        print(f"   Entry Points: {len(self.entry_nodes)}", flush=True)
        print(f"   Exit Points: {len(self.exit_nodes)}", flush=True)
        print(f"   Junctions: {len(self.junction_nodes)}", flush=True)
        print(f"   Simulation Duration: {self.t_final}s ({self.t_final/60:.1f} min)", flush=True)
        print(f"   Grid Resolution: {self.cells_per_100m} cells/100m", flush=True)
        print("=" * 70 + "\n", flush=True)
        
        return network_config


def create_victoria_island_config(
    csv_path: Optional[str] = None,
    **kwargs
) -> NetworkSimulationConfig:
    """
    Convenience function to create a Victoria Island network configuration.
    
    Args:
        csv_path: Path to topology CSV. If None, uses default location.
        **kwargs: Additional parameters to pass to VictoriaIslandConfigFactory
        
    Returns:
        Complete NetworkSimulationConfig ready for simulation
        
    Example:
        >>> config = create_victoria_island_config()
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
    
    factory = VictoriaIslandConfigFactory(csv_path=str(csv_path), **kwargs)
    return factory.create_config()
