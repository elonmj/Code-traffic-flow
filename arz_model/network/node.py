"""
Node class: Junction wrapper with network topology.

This class wraps the existing Intersection class and adds network topology
information (incoming/outgoing segments, node connectivity). It follows the
SUMO MSJunction pattern where nodes are connection points between road segments.

Academic Reference:
    - Garavello & Piccoli (2005), Section 2.2: "Nodes and Junctions"
    - SUMO's MSJunction design pattern
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.intersection import Intersection
from ..core.traffic_lights import TrafficLightController
from ..core.parameters import ModelParameters


class Node:
    """
    Network node representing a junction between road segments.
    
    This class wraps an Intersection object and adds explicit network topology,
    enabling multi-segment network simulation following Garavello & Piccoli (2005).
    
    Attributes:
        node_id: Unique identifier for this node
        position: (x, y) coordinates in network space
        intersection: Wrapped Intersection object (handles flux resolution)
        incoming_segments: List of segment IDs feeding into this node
        outgoing_segments: List of segment IDs leaving this node
        node_type: Junction type ('signalized', 'priority', 'secondary', 'roundabout')
        traffic_lights: Optional traffic light controller
        
    Academic Note:
        This implements the "node" concept from Garavello & Piccoli (2005),
        where nodes are connection points with conservation laws:
            ∑ flux_in = ∑ flux_out  (mass conservation at junctions)
    """
    
    def __init__(
        self,
        node_id: str,
        position: Tuple[float, float],
        intersection: Intersection,
        incoming_segments: List[str],
        outgoing_segments: List[str],
        node_type: str = 'signalized',
        traffic_lights: Optional[TrafficLightController] = None
    ):
        """
        Initialize network node with topology.
        
        Args:
            node_id: Unique node identifier (e.g., 'node_1', 'junction_A')
            position: (x, y) spatial coordinates
            intersection: Existing Intersection object for flux resolution
            incoming_segments: List of segment IDs entering this node
            outgoing_segments: List of segment IDs leaving this node
            node_type: Junction type ('signalized', 'priority', 'secondary', 'roundabout')
            traffic_lights: Optional traffic light controller
            
        Raises:
            ValueError: If incoming/outgoing segments are empty
        """
        if not incoming_segments:
            raise ValueError(f"Node {node_id} must have at least one incoming segment")
        if not outgoing_segments:
            raise ValueError(f"Node {node_id} must have at least one outgoing segment")
            
        self.node_id = node_id
        self.position = position
        self.intersection = intersection
        self.incoming_segments = incoming_segments
        self.outgoing_segments = outgoing_segments
        self.node_type = node_type
        self.traffic_lights = traffic_lights
        
    def get_incoming_states(self, segments: Dict) -> Dict[str, np.ndarray]:
        """
        Get boundary states from all incoming segments.
        
        Args:
            segments: Dictionary {segment_id: segment_dict} from NetworkGrid
                     where segment_dict = {'grid': Grid1D, 'U': np.ndarray, ...}
            
        Returns:
            Dictionary {segment_id: U_boundary} where U_boundary is the
            4-component state [ρ_m, w_m, ρ_c, w_c] at segment exit
            
        Academic Note:
            These are the "upstream conditions" U^-(x_node) from each
            incoming road in Garavello & Piccoli notation.
        """
        states = {}
        for seg_id in self.incoming_segments:
            segment = segments[seg_id]
            # Get rightmost cell state (segment exit = node entrance)
            U_boundary = segment['U'][:, -1]  # Shape: (4,)
            states[seg_id] = U_boundary
        return states
        
    def get_outgoing_capacities(self, segments: Dict) -> Dict[str, float]:
        """
        Get receiving capacities from all outgoing segments.
        
        Args:
            segments: Dictionary {segment_id: segment_dict} from NetworkGrid
            
        Returns:
            Dictionary {segment_id: capacity} where capacity is the
            maximum inflow the segment can accept (Daganzo's "demand")
            
        Academic Note:
            These are the supply/demand values S(ρ) from Daganzo (1995)
            used for flux distribution at junctions.
        """
        capacities = {}
        for seg_id in self.outgoing_segments:
            segment = segments[seg_id]
            grid = segment['grid']
            U = segment['U']
            
            # Get leftmost cell state (segment entrance)
            first_ghost = grid.num_ghost_cells
            rho_m = U[0, first_ghost]
            rho_c = U[2, first_ghost]
            
            # Compute receiving capacity (simplified: free-flow - current)
            # TODO Phase 5: Use proper Daganzo supply function
            rho_jam = 0.2  # Should come from params
            capacity = max(0.0, (rho_jam - rho_m - rho_c))
            capacities[seg_id] = capacity
        return capacities
        
    def is_signalized(self) -> bool:
        """Check if this is a signalized junction."""
        return self.node_type == 'signalized' and self.traffic_lights is not None
        
    def get_traffic_light_state(self) -> Optional[str]:
        """
        Get current traffic light state if signalized.
        
        Returns:
            'green', 'yellow', 'red', or None if not signalized
        """
        if self.traffic_lights is None:
            return None
        return self.traffic_lights.current_state
        
    def update_traffic_lights(self, dt: float):
        """
        Update traffic light controller (if present).
        
        Note: TrafficLightController is stateless and doesn't need updates.
        This method is kept for API compatibility and future extensions.
        """
        if self.traffic_lights is not None:
            # TrafficLightController is stateless - phase is computed from time
            # No update needed, but we could call update_stats() if needed
            self.traffic_lights.update_stats(dt)
            
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"Node(id={self.node_id}, type={self.node_type}, "
                f"in={len(self.incoming_segments)}, out={len(self.outgoing_segments)})")
