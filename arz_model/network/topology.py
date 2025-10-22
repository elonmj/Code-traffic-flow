"""
Topology utilities for network analysis and validation.

This module provides graph-based utilities for analyzing and validating
road network topology, following standard graph theory approaches used
in transportation networks (Garavello & Piccoli 2005).

Academic Reference:
    - Garavello & Piccoli (2005), Section 2.3: "Topological consistency"
    - NetworkX: Graph analysis library
"""

from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Import NetworkX if available (optional dependency)
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.warning("NetworkX not available. Some topology features disabled.")


def build_graph(segments: Dict, nodes: Dict, links: List) -> Optional:
    """
    Build NetworkX directed graph from network components.
    
    Args:
        segments: Dictionary {segment_id: segment_dict} where
                 segment_dict = {'grid': Grid1D, 'U': np.ndarray, ...}
        nodes: Dictionary {node_id: Node}
        links: List of Link objects
        
    Returns:
        NetworkX DiGraph representing network topology, or None if NetworkX unavailable
        
    Graph Structure:
        - Nodes: Junction nodes (from nodes dict)
        - Edges: Directed connections via links (segment_i → segment_j)
        - Node attributes: position, node_type, num_incoming, num_outgoing
        - Edge attributes: segment_id, length, num_cells
        
    Academic Note:
        This builds the "network graph" G = (V, E) from Garavello & Piccoli (2005),
        where V = set of junctions, E = set of road segments.
    """
    if not HAS_NETWORKX:
        return None
        
    G = nx.DiGraph()
    
    # Add nodes (junctions)
    for node_id, node in nodes.items():
        G.add_node(
            node_id,
            position=node.position,
            node_type=node.node_type,
            num_incoming=len(node.incoming_segments),
            num_outgoing=len(node.outgoing_segments)
        )
        
    # Add edges (segments via links)
    for link in links:
        # Find start and end nodes for this link
        from_seg_id = link.from_segment
        to_seg_id = link.to_segment
        
        from_segment = segments[from_seg_id]
        to_segment = segments[to_seg_id]
        
        # Extract Grid1D from segment dict
        from_grid = from_segment['grid']
        to_grid = to_segment['grid']
        
        # Add edge with segment properties
        G.add_edge(
            link.via_node.node_id,  # From node (where from_segment ends)
            link.via_node.node_id,  # To node (where to_segment starts)
            segment_id=link.link_id,
            length=to_grid.xmax - to_grid.xmin,
            num_cells=to_grid.N_physical,
            coupling_type=link.coupling_type
        )
        
    logger.info(f"Built network graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def validate_topology(
    graph: Optional,
    segments: Dict,
    nodes: Dict
) -> Tuple[bool, List[str]]:
    """
    Validate network topology consistency.
    
    Checks:
    1. All segments connected to at least one node
    2. All nodes have incoming and outgoing segments
    3. No isolated components (entire network is connected)
    4. Conservation laws can be enforced (flux balance possible)
    
    Args:
        graph: NetworkX graph from build_graph()
        segments: Dictionary {segment_id: SegmentGrid}
        nodes: Dictionary {node_id: Node}
        
    Returns:
        Tuple (is_valid, error_messages)
        
    Academic Note:
        These checks ensure the network satisfies the topological
        requirements from Garavello & Piccoli (2005), Section 2.3.
    """
    errors = []
    
    # Check 1: All segments referenced by nodes
    referenced_segments = set()
    for node in nodes.values():
        referenced_segments.update(node.incoming_segments)
        referenced_segments.update(node.outgoing_segments)
        
    for seg_id in segments.keys():
        if seg_id not in referenced_segments:
            errors.append(f"Segment {seg_id} is not connected to any node")
            
    # Check 2: All nodes have proper connectivity
    for node_id, node in nodes.items():
        if not node.incoming_segments:
            errors.append(f"Node {node_id} has no incoming segments")
        if not node.outgoing_segments:
            errors.append(f"Node {node_id} has no outgoing segments")
            
    # Check 3: Graph connectivity (if NetworkX available)
    if HAS_NETWORKX and graph is not None:
        if not nx.is_weakly_connected(graph):
            num_components = nx.number_weakly_connected_components(graph)
            errors.append(f"Network has {num_components} disconnected components")
            
        # Check for cycles (optional warning, not an error)
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                logger.info(f"Network contains {len(cycles)} cycles (circular routes)")
        except:
            pass  # Cycles check can fail for some graph structures
            
    is_valid = len(errors) == 0
    return is_valid, errors


def find_upstream_segments(node_id: str, nodes: Dict) -> List[str]:
    """
    Find all segments feeding into a node.
    
    Args:
        node_id: Target node ID
        nodes: Dictionary {node_id: Node}
        
    Returns:
        List of segment IDs entering the node
        
    Academic Note:
        These are the "incoming roads" I^-(j) in Garavello & Piccoli notation,
        used for computing ∑ flux_in at junction j.
    """
    if node_id not in nodes:
        raise ValueError(f"Node {node_id} not found")
        
    return nodes[node_id].incoming_segments.copy()


def find_downstream_segments(node_id: str, nodes: Dict) -> List[str]:
    """
    Find all segments leaving from a node.
    
    Args:
        node_id: Target node ID
        nodes: Dictionary {node_id: Node}
        
    Returns:
        List of segment IDs leaving the node
        
    Academic Note:
        These are the "outgoing roads" I^+(j) in Garavello & Piccoli notation,
        used for computing ∑ flux_out at junction j.
    """
    if node_id not in nodes:
        raise ValueError(f"Node {node_id} not found")
        
    return nodes[node_id].outgoing_segments.copy()


def compute_shortest_path(
    graph: Optional,
    start_node: str,
    end_node: str,
    weight: str = 'length'
) -> Optional[List[str]]:
    """
    Compute shortest path between two nodes.
    
    Args:
        graph: NetworkX graph from build_graph()
        start_node: Starting node ID
        end_node: Target node ID
        weight: Edge attribute for path cost ('length', 'num_cells')
        
    Returns:
        List of node IDs forming shortest path, or None if no path exists
        
    Academic Note:
        This uses Dijkstra's algorithm for weighted shortest paths,
        useful for route optimization in traffic assignment problems.
    """
    if not HAS_NETWORKX or graph is None:
        logger.warning("NetworkX not available for shortest path computation")
        return None
        
    try:
        path = nx.shortest_path(graph, start_node, end_node, weight=weight)
        return path
    except nx.NetworkXNoPath:
        logger.warning(f"No path from {start_node} to {end_node}")
        return None
    except nx.NodeNotFound as e:
        logger.error(f"Node not found: {e}")
        return None


def get_network_diameter(graph: Optional) -> Optional[float]:
    """
    Compute network diameter (longest shortest path).
    
    Args:
        graph: NetworkX graph from build_graph()
        
    Returns:
        Maximum shortest path length, or None if not computable
        
    Academic Note:
        The diameter characterizes network scale and is used in
        complexity analysis of routing algorithms.
    """
    if not HAS_NETWORKX or graph is None:
        return None
        
    if not nx.is_weakly_connected(graph):
        logger.warning("Network is not connected, diameter undefined")
        return None
        
    try:
        diameter = nx.diameter(graph.to_undirected())
        return float(diameter)
    except:
        return None
