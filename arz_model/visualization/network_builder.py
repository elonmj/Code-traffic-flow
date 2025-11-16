"""
Network Builder Module for ARZ Traffic Simulation Topology

This module provides tools for constructing NetworkX graph representations
from CSV topology files, following the Separation of Concerns principle.

Responsibility: Graph Construction (Concern 2)
- Load network topology from CSV files
- Build NetworkX DiGraph with edge attributes
- Compute node positions using various layout algorithms

Usage:
    builder = NetworkTopologyBuilder('network_topology.csv')
    builder.load_topology()
    graph = builder.get_graph()
    positions = builder.compute_layout(layout_type='spring')
"""

import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List


class NetworkTopologyBuilder:
    """
    Build NetworkX graph from CSV topology files.
    
    This class handles all graph construction operations, creating a
    directed graph with edge attributes from CSV data representing
    road network topology.
    
    Attributes:
        csv_file (Path): Path to the CSV topology file
        df (pd.DataFrame): Loaded topology data
        graph (nx.DiGraph): NetworkX directed graph
    """
    
    def __init__(self, csv_file: str):
        """
        Initialize the network topology builder.
        
        Args:
            csv_file: Path to the CSV file containing network topology
                     Expected columns: u, v, name_clean, highway, length, oneway
        """
        self.csv_file = Path(csv_file)
        self.df: Optional[pd.DataFrame] = None
        self.graph: Optional[nx.DiGraph] = None
        
    def load_topology(self) -> 'NetworkTopologyBuilder':
        """
        Load topology from CSV and build NetworkX graph.
        
        Returns:
            self: For method chaining
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV structure is invalid
        """
        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"Topology CSV file not found: {self.csv_file}"
            )
            
        # Load CSV with error handling
        try:
            self.df = pd.read_csv(self.csv_file)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
            
        # Validate required columns
        required_cols = ['u', 'v']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}. "
                f"Available columns: {list(self.df.columns)}"
            )
            
        # Build the graph
        self._build_graph()
        
        print(f"✓ Loaded topology: {len(self.graph.nodes)} nodes, "
              f"{len(self.graph.edges)} edges from {self.csv_file.name}")
        
        return self
        
    def _build_graph(self) -> None:
        """
        Construct NetworkX directed graph from CSV data.
        
        Creates a DiGraph where:
        - Nodes are intersections (from u, v columns)
        - Edges are road segments with attributes
        """
        self.graph = nx.DiGraph()
        
        # Add edges with all available attributes
        for idx, row in self.df.iterrows():
            u = row['u']
            v = row['v']
            
            # Build edge attributes dictionary
            edge_attrs = {}
            
            # Add standard attributes if they exist
            attr_mapping = {
                'name_clean': 'name',
                'highway': 'highway',
                'length': 'length',
                'oneway': 'oneway',
                'lanes_manual': 'lanes',
                'Rx_manual': 'rx',
                'maxspeed_manual_kmh': 'maxspeed'
            }
            
            for csv_col, edge_attr in attr_mapping.items():
                if csv_col in row.index and pd.notna(row[csv_col]):
                    edge_attrs[edge_attr] = row[csv_col]
                    
            # Add edge to graph
            self.graph.add_edge(u, v, **edge_attrs)
            
    def compute_layout(
        self,
        layout_type: str = 'spring',
        **kwargs
    ) -> Dict[int, Tuple[float, float]]:
        """
        Compute node positions using specified layout algorithm.
        
        Args:
            layout_type: Layout algorithm to use
                        Options: 'spring', 'kamada_kawai', 'circular', 'random'
            **kwargs: Additional arguments for the layout algorithm
            
        Returns:
            Dictionary mapping node IDs to (x, y) positions
            
        Raises:
            ValueError: If graph hasn't been built or layout type invalid
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call load_topology() first.")
            
        if len(self.graph.nodes) == 0:
            raise ValueError("Graph has no nodes")
            
        # Set default parameters for spring layout
        if layout_type == 'spring':
            default_kwargs = {
                'k': 0.5,           # Optimal distance between nodes
                'iterations': 100,   # Number of iterations
                'seed': 42          # For reproducibility
            }
            default_kwargs.update(kwargs)
            kwargs = default_kwargs
            
            pos = nx.spring_layout(self.graph, **kwargs)
            
        elif layout_type == 'kamada_kawai':
            # Good for small to medium graphs
            pos = nx.kamada_kawai_layout(self.graph)
            
        elif layout_type == 'circular':
            # Arranges nodes in a circle
            pos = nx.circular_layout(self.graph)
            
        elif layout_type == 'random':
            # Random positions (useful for testing)
            seed = kwargs.get('seed', 42)
            pos = nx.random_layout(self.graph, seed=seed)
            
        else:
            raise ValueError(
                f"Unknown layout type: '{layout_type}'. "
                f"Valid options: 'spring', 'kamada_kawai', 'circular', 'random'"
            )
            
        print(f"✓ Computed {layout_type} layout for {len(pos)} nodes")
        
        return pos
        
    def get_graph(self) -> nx.DiGraph:
        """
        Get the constructed NetworkX graph.
        
        Returns:
            NetworkX directed graph
            
        Raises:
            ValueError: If graph hasn't been built
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call load_topology() first.")
            
        return self.graph
        
    def get_edge_attributes(self, attribute: str) -> Dict[Tuple[int, int], Any]:
        """
        Get a specific attribute for all edges.
        
        Args:
            attribute: Name of the edge attribute to retrieve
            
        Returns:
            Dictionary mapping edge tuples (u, v) to attribute values
            
        Raises:
            ValueError: If graph hasn't been built
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call load_topology() first.")
            
        return nx.get_edge_attributes(self.graph, attribute)
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get network topology statistics.
        
        Returns:
            Dictionary with network statistics
            
        Raises:
            ValueError: If graph hasn't been built
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call load_topology() first.")
            
        stats = {
            'num_nodes': len(self.graph.nodes),
            'num_edges': len(self.graph.edges),
            'is_connected': nx.is_weakly_connected(self.graph),
            'num_connected_components': nx.number_weakly_connected_components(self.graph),
        }
        
        # Get edge attribute statistics if available
        highway_types = self.get_edge_attributes('highway')
        if highway_types:
            from collections import Counter
            stats['highway_type_counts'] = dict(Counter(highway_types.values()))
            
        return stats

    def create_subgraph_from_simulated(self, simulated_segment_names: List[str]) -> nx.DiGraph:
        """
        Creates a subgraph containing only the edges present in the simulation.

        Args:
            simulated_segment_names: A list of segment names (e.g., from 'name_clean' column)
                                     that were included in the simulation.

        Returns:
            A new NetworkX DiGraph containing only the simulated edges and their nodes.
            
        Raises:
            ValueError: If the graph has not been built yet.
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call load_topology() first.")

        # Get all edges that have a 'name' attribute matching the simulated segment names
        edges_to_include = [
            (u, v) for u, v, data in self.graph.edges(data=True)
            if data.get('name') in simulated_segment_names
        ]

        if not edges_to_include:
            print("⚠️ WARNING: No matching edges found for the simulated segments. Returning an empty graph.")
            return nx.DiGraph()

        # Create the subgraph from the list of edges
        subgraph = self.graph.edge_subgraph(edges_to_include).copy()
        
        print(f"✓ Created subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges "
              f"based on {len(simulated_segment_names)} simulated segments.")

        return subgraph
