"""
Network Simulation Configuration - Pydantic Model

Pydantic equivalent of the YAML-based NetworkConfig for multi-segment networks.
Enables type-safe configuration of traffic networks for RL training without YAML.

This module provides:
- SegmentConfig: Individual road segment configuration
- NodeConfig: Junction/boundary node configuration  
- LinkConfig: Directional connection between segments
- NetworkSimulationConfig: Complete network configuration

Author: ARZ Research Team
Date: 2025-10-28 (Phase 6 - NetworkGrid Integration)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any
from .bc_config import BoundaryConditionsConfig


class SegmentConfig(BaseModel):
    """
    Configuration for a single road segment.
    
    A segment represents a continuous section of road with uniform parameters.
    Segments are connected at nodes (junctions).
    
    Attributes:
        x_min: Segment start position (meters)
        x_max: Segment end position (meters)
        N: Number of spatial cells for discretization
        start_node: ID of upstream node
        end_node: ID of downstream node
        parameters: Optional segment-specific parameters (V0, tau, rho_max, etc.)
    
    Example:
        >>> seg = SegmentConfig(
        ...     x_min=0.0, x_max=200.0, N=40,
        ...     start_node='node_0', end_node='node_1',
        ...     parameters={'V0_c': 13.89, 'tau_c': 18.0}
        ... )
    """
    x_min: float = Field(ge=0, description="Segment start position (m)")
    x_max: float = Field(gt=0, description="Segment end position (m)")
    N: int = Field(ge=10, description="Number of spatial cells")
    start_node: Optional[str] = Field(default=None, description="Upstream node ID (None for boundary)")
    end_node: Optional[str] = Field(default=None, description="Downstream node ID (None for boundary)")
    parameters: Optional[Dict[str, float]] = Field(
        default=None,
        description="Segment-specific parameters (V0, tau, rho_max, etc.)"
    )
    
    @field_validator('x_max')
    @classmethod
    def validate_x_max(cls, v, info):
        """Validate that x_max > x_min."""
        if 'x_min' in info.data and v <= info.data['x_min']:
            raise ValueError('x_max must be > x_min')
        return v
    
    @field_validator('N')
    @classmethod
    def validate_N(cls, v):
        """Validate spatial cells are sufficient."""
        if v < 10:
            raise ValueError('N must be >= 10 for numerical stability')
        return v
    
    model_config = {"extra": "forbid"}


class NodeConfig(BaseModel):
    """
    Configuration for a junction or boundary node.
    
    Nodes represent connection points where segments meet. Types:
    - boundary: Entry/exit point (inflow/outflow boundary conditions)
    - signalized: Traffic light controlled intersection
    - stop_sign: Stop sign controlled intersection (future)
    
    Attributes:
        type: Node type (boundary/signalized/stop_sign)
        position: [x, y] coordinates for visualization
        incoming_segments: List of segment IDs entering this node
        outgoing_segments: List of segment IDs leaving this node
        traffic_light_config: Traffic light parameters (cycle, phases, etc.)
    
    Example:
        >>> node = NodeConfig(
        ...     type='signalized',
        ...     position=[200.0, 0.0],
        ...     incoming_segments=['seg_0'],
        ...     outgoing_segments=['seg_1'],
        ...     traffic_light_config={
        ...         'cycle_time': 60.0,
        ...         'green_time': 25.0,
        ...         'phases': [
        ...             {'id': 0, 'name': 'GREEN'},
        ...             {'id': 1, 'name': 'RED'}
        ...         ]
        ...     }
        ... )
    """
    type: str = Field(description="Node type: boundary/signalized/stop_sign")
    position: List[float] = Field(description="[x, y] position coordinates")
    incoming_segments: Optional[List[str]] = Field(
        default=None,
        description="Segment IDs entering this node"
    )
    outgoing_segments: Optional[List[str]] = Field(
        default=None,
        description="Segment IDs leaving this node"
    )
    traffic_light_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Traffic light parameters (cycle_time, phases, etc.)"
    )
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """Validate node type is recognized."""
        valid_types = ['boundary', 'signalized', 'stop_sign']
        if v not in valid_types:
            raise ValueError(f'type must be one of {valid_types}, got {v}')
        return v
    
    @field_validator('position')
    @classmethod
    def validate_position(cls, v):
        """Validate position is [x, y] format."""
        if len(v) != 2:
            raise ValueError('position must be [x, y] with exactly 2 values')
        if not all(isinstance(coord, (int, float)) for coord in v):
            raise ValueError('position coordinates must be numeric')
        return v
    
    @field_validator('traffic_light_config')
    @classmethod
    def validate_traffic_light_config(cls, v, info):
        """Validate traffic light config is present for signalized nodes."""
        if 'type' in info.data and info.data['type'] == 'signalized':
            if v is None:
                raise ValueError('signalized nodes must have traffic_light_config')
            # Validate required keys
            required_keys = ['cycle_time', 'green_time', 'phases']
            missing = [k for k in required_keys if k not in v]
            if missing:
                raise ValueError(f'traffic_light_config missing keys: {missing}')
        return v
    
    model_config = {"extra": "forbid"}


class LinkConfig(BaseModel):
    """
    Configuration for a directional connection between segments.
    
    Links define how traffic flows through the network. Each link represents
    a single directional connection from one segment to another via a node.
    
    Attributes:
        from_segment: Source segment ID
        to_segment: Destination segment ID
        via_node: Junction node ID connecting the segments
        coupling_type: Junction coupling algorithm (theta_k/flux_matching)
    
    Example:
        >>> link = LinkConfig(
        ...     from_segment='seg_0',
        ...     to_segment='seg_1',
        ...     via_node='node_1',
        ...     coupling_type='theta_k'
        ... )
    """
    from_segment: str = Field(description="Source segment ID")
    to_segment: str = Field(description="Destination segment ID")
    via_node: str = Field(description="Junction node ID")
    coupling_type: Optional[str] = Field(
        default='theta_k',
        description="Coupling algorithm: theta_k or flux_matching"
    )
    
    @field_validator('coupling_type')
    @classmethod
    def validate_coupling_type(cls, v):
        """Validate coupling algorithm is recognized."""
        valid_types = ['theta_k', 'flux_matching']
        if v not in valid_types:
            raise ValueError(f'coupling_type must be one of {valid_types}, got {v}')
        return v
    
    model_config = {"extra": "forbid"}


class NetworkSimulationConfig(BaseModel):
    """
    Complete network simulation configuration (Pydantic).
    
    This is the Pydantic equivalent of NetworkConfig (YAML loader).
    Used for programmatic creation of network configurations without YAML files.
    
    Replaces the 2-file YAML architecture:
    - network.yml → segments, nodes, links (THIS CLASS)
    - traffic_control.yml → traffic_light_config in NodeConfig
    
    Benefits over YAML:
    - Type safety: Errors caught at config creation, not runtime
    - No file I/O: Faster, no serialization bugs
    - Programmatic: Easy to generate variations for experiments
    - IDE support: Autocomplete, type hints
    
    Attributes:
        segments: Dictionary of segment configurations
        nodes: Dictionary of node configurations
        links: List of segment connections
        global_params: Global default parameters (V0, tau, rho_max, etc.)
        dt: Simulation timestep (seconds)
        t_final: Final simulation time (seconds)
        output_dt: Output interval (seconds)
        device: Computation device ('cpu' or 'gpu')
        decision_interval: RL agent decision interval (seconds)
        controlled_nodes: Node IDs controlled by RL agent
    
    Usage:
        >>> # Simple corridor (2 segments, 1 signal)
        >>> config = RLNetworkConfigBuilder.simple_corridor(segments=2)
        >>> 
        >>> # Medium network (10 segments, 3 signals)
        >>> config = RLNetworkConfigBuilder.medium_network(segments=10)
        >>> 
        >>> # Use in RL environment
        >>> env = TrafficSignalEnvDirect(simulation_config=config)
    
    Academic Reference:
        - Garavello & Piccoli (2005): "Traffic Flow on Networks"
        - Network formulation: I (segments) + J (junctions) + conservation laws
    """
    segments: Dict[str, SegmentConfig] = Field(
        description="Dictionary of segment configurations {seg_id: SegmentConfig}"
    )
    nodes: Dict[str, NodeConfig] = Field(
        description="Dictionary of node configurations {node_id: NodeConfig}"
    )
    links: List[LinkConfig] = Field(
        default_factory=list,
        description="List of segment connections (directional)"
    )
    
    # Global parameters (shared across network)
    global_params: Dict[str, float] = Field(
        default_factory=lambda: {
            'V0_c': 13.89,      # 50 km/h free-flow speed cars (m/s)
            'V0_m': 15.28,      # 55 km/h free-flow speed motorcycles (m/s)
            'tau_c': 18.0,      # Relaxation time cars (s)
            'tau_m': 20.0,      # Relaxation time motorcycles (s)
            'rho_max_c': 200.0, # Max density cars (veh/km)
            'rho_max_m': 150.0, # Max density motorcycles (veh/km)
            'kappa': 1.5,       # Anticipation coefficient
            'nu': 2.0,          # Interaction exponent
            'delta': 0.3        # Anisotropy parameter
        },
        description="Global default parameters for all segments"
    )
    
    # Simulation parameters
    dt: float = Field(default=0.1, gt=0, description="Simulation timestep (s)")
    t_final: float = Field(default=3600.0, gt=0, description="Final time (s)")
    output_dt: float = Field(default=1.0, gt=0, description="Output interval (s)")
    device: str = Field(default='cpu', description="Computation device: cpu or gpu")
    
    # RL-specific parameters
    decision_interval: float = Field(
        default=15.0,
        gt=0,
        description="Agent decision interval (s) - from Bug #27 validation"
    )
    controlled_nodes: Optional[List[str]] = Field(
        default=None,
        description="Node IDs controlled by RL agent (default: all signalized)"
    )
    
    # Boundary conditions
    boundary_conditions: Optional[BoundaryConditionsConfig] = Field(
        default=None,
        description="Boundary conditions for network edges (inflow/outflow)"
    )
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        """Validate device is recognized."""
        valid_devices = ['cpu', 'gpu']
        if v not in valid_devices:
            raise ValueError(f'device must be one of {valid_devices}, got {v}')
        return v
    
    @field_validator('controlled_nodes')
    @classmethod
    def validate_controlled_nodes(cls, v, info):
        """Auto-populate controlled_nodes with all signalized nodes if None."""
        if v is None and 'nodes' in info.data:
            # Extract all signalized node IDs
            signalized = [
                node_id for node_id, node in info.data['nodes'].items()
                if node.type == 'signalized'
            ]
            return signalized if signalized else None
        return v
    
    def validate_network_topology(self) -> None:
        """
        Validate network topology consistency.
        
        Checks:
        - All links reference existing segments and nodes
        - Segment start/end nodes exist
        - No orphaned segments
        
        Raises:
            ValueError: If topology is invalid
        """
        # Check link references
        for link in self.links:
            if link.from_segment not in self.segments:
                raise ValueError(f"Link references unknown from_segment: {link.from_segment}")
            if link.to_segment not in self.segments:
                raise ValueError(f"Link references unknown to_segment: {link.to_segment}")
            if link.via_node not in self.nodes:
                raise ValueError(f"Link references unknown via_node: {link.via_node}")
        
        # Check segment node references
        for seg_id, segment in self.segments.items():
            # Allow None for boundary segments
            if segment.start_node is not None and segment.start_node not in self.nodes:
                raise ValueError(f"Segment {seg_id} references unknown start_node: {segment.start_node}")
            if segment.end_node is not None and segment.end_node not in self.nodes:
                raise ValueError(f"Segment {seg_id} references unknown end_node: {segment.end_node}")
    
    model_config = {"extra": "forbid"}


# Export public API
__all__ = [
    'NetworkSimulationConfig',
    'SegmentConfig',
    'NodeConfig',
    'LinkConfig'
]
