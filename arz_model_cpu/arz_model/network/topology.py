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


def build_graph(segments: Dict, nodes: Dict) -> Optional:
    """
    Build NetworkX directed graph from network components.
    
    Args:
        segments: Dictionary {segment_id: segment_dict} where
                 segment_dict contains 'start_node' and 'end_node' IDs.
        nodes: Dictionary {node_id: Node} for adding node attributes.
        
    Returns:
        NetworkX DiGraph representing network topology, or None if NetworkX unavailable.
    """
    if not HAS_NETWORKX:
        return None
        
    G = nx.DiGraph()
    
    # Add nodes from the keys of the nodes dictionary
    for node_id, node in nodes.items():
        G.add_node(
            node_id,
            position=node.position,
            node_type=node.node_type,
            num_incoming=len(node.incoming_segments),
            num_outgoing=len(node.outgoing_segments)
        )
        
    # Add edges by iterating through segments
    for seg_id, segment_data in segments.items():
        start_node = segment_data.get('start_node')
        end_node = segment_data.get('end_node')
        
        if start_node and end_node:
            # Ensure nodes exist in the graph before adding an edge
            if not G.has_node(start_node):
                G.add_node(start_node) # Add with no attributes if not in main nodes list
            if not G.has_node(end_node):
                G.add_node(end_node) # Add with no attributes if not in main nodes list

            grid = segment_data['grid']
            G.add_edge(
                start_node,
                end_node,
                segment_id=seg_id,
                length=grid.xmax - grid.xmin,
                num_cells=grid.N_physical
            )
        
    logger.info(f"Built network graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def validate_topology(segments: Dict, nodes: Dict) -> None:
    """
    Validates the topological consistency of the network.

    This function checks that the connectivity information stored in the nodes
    matches the connections defined by the segments. It ensures that every
    segment's start and end nodes are correctly registered in the corresponding
    node's `outgoing_segments` and `incoming_segments` lists.

    Raises:
        ValueError: If any topological inconsistencies are found.
    """
    errors = []

    # Check 1: Verify that every segment's start_node has it in its outgoing list
    for seg_id, segment in segments.items():
        start_node_id = segment.get('start_node')
        if not start_node_id:
            errors.append(f"Segment '{seg_id}' is missing a start_node.")
            continue
        
        start_node = nodes.get(start_node_id)
        if not start_node:
            errors.append(f"Segment '{seg_id}' points to a non-existent start_node '{start_node_id}'.")
            continue
            
        if seg_id not in start_node.outgoing_segments:
            errors.append(
                f"Topological error at node '{start_node_id}': "
                f"Segment '{seg_id}' starts here, but is not in the node's outgoing_segments list. "
                f"Node's outgoing: {start_node.outgoing_segments}"
            )

    # Check 2: Verify that every segment's end_node has it in its incoming list
    for seg_id, segment in segments.items():
        end_node_id = segment.get('end_node')
        if not end_node_id:
            errors.append(f"Segment '{seg_id}' is missing an end_node.")
            continue

        end_node = nodes.get(end_node_id)
        if not end_node:
            errors.append(f"Segment '{seg_id}' points to a non-existent end_node '{end_node_id}'.")
            continue
            
        if seg_id not in end_node.incoming_segments:
            errors.append(
                f"Topological error at node '{end_node_id}': "
                f"Segment '{seg_id}' ends here, but is not in the node's incoming_segments list. "
                f"Node's incoming: {end_node.incoming_segments}"
            )
            
    # Check 3: Verify node-centric consistency
    for node_id, node in nodes.items():
        # Every incoming segment must point to this node as its end_node
        for seg_id in node.incoming_segments:
            segment = segments.get(seg_id)
            if not segment or segment.get('end_node') != node_id:
                errors.append(
                    f"Node '{node_id}' lists '{seg_id}' as incoming, but the segment does not end here."
                )
        
        # Every outgoing segment must point to this node as its start_node
        for seg_id in node.outgoing_segments:
            segment = segments.get(seg_id)
            if not segment or segment.get('start_node') != node_id:
                errors.append(
                    f"Node '{node_id}' lists '{seg_id}' as outgoing, but the segment does not start here."
                )

    if errors:
        error_summary = "\n - ".join(errors)
        raise ValueError(f"Invalid network topology. Found {len(errors)} errors:\n - {error_summary}")

    # Optional: Use NetworkX for deeper graph analysis if available
    graph = build_graph(segments, nodes)
    if graph:
        # Check for isolated components
        if not nx.is_weakly_connected(graph):
            num_components = nx.number_weakly_connected_components(graph)
            logger.warning(f"Network has {num_components} isolated sub-graphs. This may be intentional.")
            
        # Check for nodes with no connections (degree 0)
        isolated_nodes = [node for node, degree in graph.degree() if degree == 0]
        if isolated_nodes:
            logger.warning(f"Found isolated nodes with no connections: {isolated_nodes}")
            
    logger.info("Network topology validation passed.")
    return True, []


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
        used for computing sum of flux_in at junction j.
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
        used for computing sum of flux_out at junction j.
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
