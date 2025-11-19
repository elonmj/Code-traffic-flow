"""
Node class: Junction wrapper with network topology.

This class manages the state and logic of a junction in the traffic network.

Attributes:
    node_id: Unique identifier for this node
    position: (x, y) coordinates in network space
    node_type: Type of node (e.g., 'junction', 'source', 'sink')
    incoming_segments: List of Link objects entering this node
    outgoing_segments: List of Link objects leaving this node
    traffic_light_controller: Optional controller for traffic lights
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.node_solver_gpu import solve_node_fluxes_gpu
from ..core.parameters import ModelParameters
from .traffic_lights import TrafficLightController


class Node:
    """
    Network node representing a junction between road segments.
    
    This class manages the state and logic of a junction in the traffic network.
    
    Attributes:
        node_id: Unique identifier for this node
        position: (x, y) coordinates in network space
        node_type: Type of node (e.g., 'junction', 'source', 'sink')
        incoming_segments: List of Link objects entering this node
        outgoing_segments: List of Link objects leaving this node
        traffic_light_controller: Optional controller for traffic lights
    """
    @classmethod
    def from_config(cls, config: 'NodeConfig') -> 'Node':
        """
        Creates a Node instance from a NodeConfig object.
        """
        traffic_lights = None
        if config.traffic_light_config:
            traffic_lights = TrafficLightController(
                node_id=config.id,
                config=config.traffic_light_config,
                incoming_segments=config.incoming_segments or []
            )

        return cls(
            node_id=config.id,
            position=config.position if config.position else (0.0, 0.0),
            node_type=config.type,
            incoming_segments=config.incoming_segments,
            outgoing_segments=config.outgoing_segments,
            traffic_lights=traffic_lights
        )

    def __init__(self, node_id: str, position: Tuple[float, float] = (0.0, 0.0), node_type: str = 'junction', incoming_segments: List[str] = None, outgoing_segments: List[str] = None, traffic_lights: Optional[TrafficLightController] = None):
        self.node_id = node_id
        self.position = position
        self.node_type = node_type
        self.incoming_segments: List[str] = incoming_segments or []
        self.outgoing_segments: List[str] = outgoing_segments or []
        self.traffic_lights = traffic_lights

    def add_incoming_segment(self, segment: 'Link'):
        self.incoming_segments.append(segment)

    def add_outgoing_segment(self, segment: 'Link'):
        self.outgoing_segments.append(segment)

    def get_boundary_states(self) -> Dict[str, np.ndarray]:
        """
        Gathers the boundary states from all incoming segments.
        This is the input for the Riemann solver at the junction.
        """
        states = {}
        for seg in self.incoming_segments:
            # The boundary state is the last physical cell of the incoming segment
            states[seg.segment_id] = seg.grid.get_downstream_boundary_state()
        return states

    def get_outgoing_capacities(self) -> Dict[str, float]:
        """
        Gathers the capacities of all outgoing segments.
        This determines the maximum flow that can be accepted by each outgoing road.
        """
        capacities = {}
        for seg in self.outgoing_segments:
            # Capacity is related to the maximum density and speed
            capacities[seg.segment_id] = seg.grid.get_capacity()
        return capacities

    def solve_fluxes(self, params: ModelParameters, t: float) -> Dict[str, np.ndarray]:
        """
        Solves the Riemann problem at the junction to determine outgoing fluxes.
        
        This is the core of the network coupling logic. It takes the state of all
        incoming roads and calculates the resulting flow into each outgoing road.
        """
        incoming_states = self.get_boundary_states()
        outgoing_capacities = self.get_outgoing_capacities()
        
        # Get traffic light state if applicable
        green_mask = None
        
        # Prepare input for the GPU node solver
        # This part will be adapted for the GPU-native solver
        
        # For now, we simulate the logic that will be on the GPU
        # The actual GPU call will be in `apply_network_coupling_gpu`
        
        # This is a placeholder for the complex logic of solve_node_fluxes_gpu
        # In the real implementation, this would involve a sophisticated solver.
        
        # The actual `solve_node_fluxes_gpu` would be a GPU kernel
        # Here we call a placeholder for the logic
        outgoing_fluxes_flat = solve_node_fluxes_gpu(
            self.node_id, 
            np.array(list(incoming_states.values())), 
            len(self.outgoing_segments), 
            params
        )
        
        # Map the flat array of fluxes back to the outgoing segments
        # This logic needs to be adapted based on the actual return of the GPU solver
        # For now, assuming it returns a tuple of (q_m, q_c) per outgoing link
        
        outgoing_fluxes = {}
        for i, seg in enumerate(self.outgoing_segments):
            # This is a simplified placeholder
            # The real implementation would construct a full flux vector
            q_m, q_c = outgoing_fluxes_flat # Simplified assumption
            # This needs to be a full state vector, not just fluxes
            # This part of the code is incomplete and needs the real GPU solver logic
            flux_vector = np.zeros(4) 
            outgoing_fluxes[seg.segment_id] = flux_vector
        
        return outgoing_fluxes

    def __repr__(self):
        return (
            f"Node(id={self.node_id}, type={self.node_type}, "
            f"in={len(self.incoming_segments)}, out={len(self.outgoing_segments)})"
        )
