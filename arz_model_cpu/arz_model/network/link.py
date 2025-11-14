"""
Link class: Segment connection with θ_k behavioral coupling.

This class represents a directed connection from one segment to another through
a node, managing the phenomenological coupling parameter θ_k that controls
behavioral memory preservation (Kolb et al. 2018).

Academic Reference:
    - Kolb et al. (2018): "Phenomenological coupling for ARZ traffic model"
    - Thesis Chapter 4: Two-step network coupling (flux + behavior)
"""

from typing import Optional
import numpy as np

from ..core.parameters import ModelParameters
from ..core.node_solver import _get_coupling_parameter, _apply_behavioral_coupling
from .node import Node


class Link:
    """
    Directed link connecting two segments through a node.
    
    This class manages the θ_k behavioral coupling between an upstream segment
    (from_segment) and a downstream segment (to_segment) passing through a
    junction (via_node). It implements the two-step coupling:
        1. Flux resolution (handled by Node's Intersection)
        2. Behavioral coupling w_out = w_eq_out + θ_k(w_in - w_eq_in)
        
    Attributes:
        link_id: Unique identifier
        from_segment: ID of upstream segment
        to_segment: ID of downstream segment  
        via_node: Node object representing the junction
        coupling_type: Type of coupling ('sequential', 'diverging', 'merging')
        
    Academic Note:
        The coupling_type determines flux distribution at multi-branch junctions:
        - 'sequential': Single input → single output (θ_k applies directly)
        - 'diverging': Single input → multiple outputs (flux split by turning ratios)
        - 'merging': Multiple inputs → single output (flux aggregation)
    """
    
    def __init__(
        self,
        link_id: str,
        from_segment: str,
        to_segment: str,
        via_node: Node,
        coupling_type: str = 'sequential',
        params: Optional[ModelParameters] = None
    ):
        """
        Initialize link with coupling configuration.
        
        Args:
            link_id: Unique link identifier
            from_segment: Upstream segment ID
            to_segment: Downstream segment ID
            via_node: Junction node connecting the segments
            coupling_type: 'sequential', 'diverging', or 'merging'
            params: Model parameters (contains θ_k values)
            
        Raises:
            ValueError: If from_segment not in node's incoming or to_segment not in outgoing
        """
        if from_segment not in via_node.incoming_segments:
            raise ValueError(f"Segment {from_segment} not in node {via_node.node_id} incoming segments")
        if to_segment not in via_node.outgoing_segments:
            raise ValueError(f"Segment {to_segment} not in node {via_node.node_id} outgoing segments")
            
        self.link_id = link_id
        self.from_segment = from_segment
        self.to_segment = to_segment
        self.via_node = via_node
        self.coupling_type = coupling_type
        self.params = params
        
    def apply_coupling(
        self,
        U_in: np.ndarray,
        U_out: np.ndarray,
        vehicle_class: str = 'motorcycle',
        time: float = 0.0
    ) -> np.ndarray:
        """
        Apply θ_k behavioral coupling from upstream to downstream segment.
        
        This implements the thesis equation:
            w_out = w_eq_out + θ_k * (w_in - w_eq_in)
        where θ_k depends on junction type and traffic light state.
        
        Args:
            U_in: Upstream state [ρ_m, w_m, ρ_c, w_c] from from_segment exit
            U_out: Downstream state [ρ_m, w_m, ρ_c, w_c] at to_segment entrance
            vehicle_class: 'motorcycle' or 'car'
            time: Current simulation time (seconds)
            
        Returns:
            Coupled state U_coupled with adjusted w values
            
        Academic Note:
            The θ_k parameter is selected based on:
            - Junction type (signalized, priority, secondary, roundabout)
            - Traffic light state (green → θ_signalized, red → 0.0)
            - Vehicle class (motorcycles have higher θ at green lights)
            
        References:
            - Kolb et al. (2018), Equation (15): Phenomenological coupling
            - Thesis Chapter 4, Section 4.2: θ_k parameter selection
        """
        if self.params is None:
            # No coupling if parameters not provided
            return U_out.copy()
            
        # Step 1: Determine θ_k based on junction characteristics
        theta_k = _get_coupling_parameter(
            self.via_node.intersection,
            self.to_segment,  # segment_id
            vehicle_class,
            self.params,
            time
        )
        
        # DEBUG: Afficher θ_k pour diagnostiquer les feux rouges (limité aux 3 premiers pas)
        if time < 0.3:  # Afficher seulement les 3 premiers pas (0.0s, 0.1s, 0.2s)
            try:
                if self.via_node.intersection.traffic_lights is not None:
                    green_segs = self.via_node.intersection.traffic_lights.get_current_green_segments(time)
                    print(f"[COUPLING_DEBUG] time={repr(time)}, link={repr(self.link_id)}, theta_k={repr(theta_k)}, to_seg={repr(self.to_segment)}, green_segs={repr(green_segs)}")
                else:
                    print(f"[COUPLING_DEBUG] time={repr(time)}, link={repr(self.link_id)}, theta_k={repr(theta_k)}, to_seg={repr(self.to_segment)}, TL=NULL")
            except Exception as e:
                import traceback
                print(f"[COUPLING_DEBUG_ERROR] error: {e}")
                traceback.print_exc()
        
        # Step 2: Apply behavioral coupling
        U_coupled = _apply_behavioral_coupling(
            U_in,
            U_out,
            theta_k,
            self.params,
            vehicle_class
        )
        
        return U_coupled
        
    def get_coupling_strength(self, vehicle_class: str = 'motorcycle', time: float = 0.0) -> float:
        """
        Get current θ_k coupling strength.
        
        Useful for diagnostics and visualization.
        
        Args:
            vehicle_class: 'motorcycle' or 'car'
            time: Current simulation time (seconds)
            
        Returns:
            Current θ_k value in [0, 1]
        """
        if self.params is None:
            return 0.0
            
        return _get_coupling_parameter(
            self.via_node.intersection,
            self.to_segment,
            vehicle_class,
            self.params,
            time
        )
        
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"Link(id={self.link_id}, {self.from_segment}→{self.to_segment}, "
                f"via={self.via_node.node_id}, type={self.coupling_type})")
