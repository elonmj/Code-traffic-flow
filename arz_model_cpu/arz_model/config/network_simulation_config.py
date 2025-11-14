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
from .grid_config import GridConfig
from .physics_config import PhysicsConfig
from .time_config import TimeConfig
from .ic_config import ICConfig
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
        boundary_condition: Boundary condition configuration for source/sink nodes
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
    position: Optional[List[float]] = Field(default=None, description="[x, y] position coordinates")
    incoming_segments: Optional[List[str]] = Field(
        default=None,
        description="Segment IDs entering this node"
    )
    outgoing_segments: Optional[List[str]] = Field(
        default=None,
        description="Segment IDs leaving this node"
    )
    boundary_condition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Boundary condition configuration for source/sink nodes"
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
        if v is not None:
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
        return v

    @field_validator('boundary_condition')
    @classmethod
    def validate_boundary_condition(cls, v, info):
        """Validate boundary condition is present for boundary nodes."""
        if 'type' in info.data and info.data['type'] == 'boundary':
            if v is None:
                raise ValueError('boundary nodes must have a boundary_condition config')
            if 'type' not in v or v['type'] not in ['inflow', 'outflow']:
                raise ValueError("Boundary condition must have a 'type' of 'inflow' or 'outflow'")
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
    Complete configuration for a network simulation.

    This model aggregates all other configuration models (time, physics, grid,
    segments, nodes, links, and boundary conditions) into a single, unified
    object that can be passed throughout the simulation system.
    """
    device: str = Field("cpu", description="Computation device ('cpu' or 'gpu')")
    time: TimeConfig
    physics: PhysicsConfig
    grid: GridConfig
    ic: ICConfig
    segments: Dict[str, SegmentConfig]
    nodes: Dict[str, NodeConfig]
    links: Dict[str, LinkConfig]
    boundary_conditions: BoundaryConditionsConfig

    @field_validator('links')
    @classmethod
    def validate_links(cls, v: Dict[str, 'LinkConfig']) -> Dict[str, 'LinkConfig']:
        """Validate that 'from_segment' is unique across all links."""
        from_segments = set()
        for link in v.values():
            if link.from_segment in from_segments:
                raise ValueError(f"Duplicate 'from_segment' found in links: {link.from_segment}")
            from_segments.add(link.from_segment)
        return v
    
    model_config = {"extra": "forbid"}


# Export public API
__all__ = [
    'NetworkSimulationConfig',
    'SegmentConfig',
    'NodeConfig',
    'LinkConfig'
]
