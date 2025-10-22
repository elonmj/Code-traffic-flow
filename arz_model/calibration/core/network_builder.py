"""
Network Builder for ARZ Calibration
==================================

This module transforms CSV corridor data into ARZ network objects
that can be used for simulation and calibration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ..data.group_manager import SegmentInfo
from ...core.parameter_manager import ParameterManager
# TODO: Import when core modules are available
# from ..core.intersection import Intersection
# from ..core.traffic_lights import TrafficLightController


@dataclass
class RoadSegment:
    """Represents a road segment in the network"""
    segment_id: str
    start_node: str
    end_node: str
    name: str
    length: float  # meters
    highway_type: str  # 'primary', 'secondary', 'tertiary'
    oneway: bool
    lanes: Optional[int] = None
    maxspeed: Optional[float] = None  # km/h

    @property
    def direction(self) -> str:
        """Get segment direction for identification"""
        return f"{self.start_node}_{self.end_node}"


@dataclass
class NetworkNode:
    """Represents a node in the network (intersection or endpoint)"""
    node_id: str
    position: Tuple[float, float] = (0.0, 0.0)  # x, y coordinates
    connected_segments: Optional[List[str]] = None
    is_intersection: bool = False
    # traffic_lights: Optional[TrafficLightController] = None  # TODO: Uncomment when available

    def __post_init__(self):
        if self.connected_segments is None:
            self.connected_segments = []


class NetworkBuilder:
    """
    Builds ARZ network objects from CSV corridor data.

    Transforms the graph representation (u,v segments) into:
    - RoadSegment objects for each road piece
    - NetworkNode objects for intersections
    - Intersection objects for traffic flow resolution
    - ParameterManager for heterogeneous parameters (Phase 6 integration)
    
    Architecture:
        NetworkBuilder + ParameterManager → NetworkGrid (direct, no YAML intermediate)
    """

    def __init__(self, global_params: Optional[Dict[str, float]] = None):
        """
        Initialize NetworkBuilder with integrated ParameterManager.
        
        Args:
            global_params: Global default parameters. If None, uses standard ARZ defaults.
        """
        self.segments: Dict[str, RoadSegment] = {}
        self.nodes: Dict[str, NetworkNode] = {}
        self.intersections: List[Intersection] = []
        
        # Phase 6 Integration: ParameterManager for heterogeneous parameters
        if global_params is None:
            global_params = {
                'V0_c': 13.89,      # 50 km/h default
                'V0_m': 15.28,      # 55 km/h default
                'tau_c': 18.0,      # Relaxation time cars (s)
                'tau_m': 20.0,      # Relaxation time motorcycles (s)
                'rho_max_c': 200.0, # Max density cars (veh/km)
                'rho_max_m': 150.0  # Max density motorcycles (veh/km)
            }
        
        self.parameter_manager = ParameterManager(global_params=global_params)
    
    def set_segment_params(
        self,
        segment_id: str,
        parameters: Dict[str, float]
    ) -> None:
        """
        Set local parameters for a segment (overrides global defaults).
        
        This method applies calibrated or custom parameters to specific segments,
        enabling heterogeneous networks (e.g., arterial vs residential speeds).
        
        Args:
            segment_id: Segment identifier (e.g., 'seg_arterial_1')
            parameters: Dictionary of parameter overrides
                       e.g., {'V0_c': 16.67, 'tau_c': 15.0}
        
        Example:
            >>> builder.set_segment_params('seg_main_1', {
            ...     'V0_c': 16.67,  # 60 km/h for arterial
            ...     'tau_c': 15.0   # Lower relaxation for high-speed
            ... })
        """
        for param_name, value in parameters.items():
            self.parameter_manager.set_local(segment_id, param_name, value)
    
    def get_segment_params(
        self,
        segment_id: str
    ) -> Dict[str, float]:
        """
        Get effective parameters for a segment (local overrides or global defaults).
        
        Returns complete set of ARZ parameters for the specified segment,
        combining local overrides with global defaults.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            Dictionary with all ARZ parameters for this segment
            
        Example:
            >>> params = builder.get_segment_params('seg_main_1')
            >>> print(params['V0_c'])  # 16.67 (if set) or 13.89 (global default)
        """
        param_names = ['V0_c', 'V0_m', 'tau_c', 'tau_m', 'rho_max_c', 'rho_max_m']
        return {
            name: self.parameter_manager.get(segment_id, name)
            for name in param_names
        }

    def build_from_csv(self, corridor_file: str) -> Dict[str, Any]:
        """
        Build network from CSV corridor file.

        Args:
            corridor_file: Path to CSV file with corridor data

        Returns:
            Dictionary containing network components
        """
        # Load corridor data
        df = pd.read_csv(corridor_file)

        # Build segments
        self._build_segments(df)

        # Build nodes and identify intersections
        self._build_nodes()

        # Create ARZ intersection objects
        self._create_intersections()

        return {
            'segments': self.segments,
            'nodes': self.nodes,
            'intersections': self.intersections,
            'segment_count': len(self.segments),
            'node_count': len(self.nodes),
            'intersection_count': len(self.intersections)
        }

    def build_from_segments(self, segments: List['SegmentInfo']) -> Dict[str, Any]:
        """
        Build network from list of SegmentInfo objects.

        Args:
            segments: List of SegmentInfo objects from a network group

        Returns:
            Dictionary containing network components
        """
        # Convert SegmentInfo to DataFrame for compatibility
        segment_data = []
        for seg in segments:
            segment_data.append({
                'u': seg.start_node,
                'v': seg.end_node,
                'name_clean': seg.name,
                'highway': seg.highway_type,
                'length': seg.length,
                'oneway': seg.oneway,
                'lanes_manual': seg.lanes,
                'maxspeed_manual_kmh': seg.max_speed
            })

        df = pd.DataFrame(segment_data)

        # Build segments
        self._build_segments(df)

        # Build nodes and identify intersections
        self._build_nodes()

        # Create ARZ intersection objects
        self._create_intersections()

        return {
            'segments': self.segments,
            'nodes': self.nodes,
            'intersections': self.intersections,
            'segment_count': len(self.segments),
            'node_count': len(self.nodes),
            'intersection_count': len(self.intersections)
        }

    def _build_segments(self, df: pd.DataFrame):
        """Build RoadSegment objects from DataFrame"""
        for _, row in df.iterrows():
            segment_id = f"{row['u']}_{row['v']}"

            segment = RoadSegment(
                segment_id=segment_id,
                start_node=str(row['u']),
                end_node=str(row['v']),
                name=row['name_clean'],
                length=float(row['length']),
                highway_type=row['highway'],
                oneway=bool(row['oneway']),
                lanes=row['lanes_manual'] if pd.notna(row['lanes_manual']) else None,
                maxspeed=row['maxspeed_manual_kmh'] if pd.notna(row['maxspeed_manual_kmh']) else None
            )

            self.segments[segment_id] = segment

    def _build_nodes(self):
        """Build NetworkNode objects and identify intersections"""
        # Collect all node connections
        node_connections = defaultdict(list)

        for segment in self.segments.values():
            node_connections[segment.start_node].append(segment.segment_id)
            if not segment.oneway:
                node_connections[segment.end_node].append(segment.segment_id)

        # Create nodes
        for node_id, connected_segments in node_connections.items():
            is_intersection = len(connected_segments) > 1

            node = NetworkNode(
                node_id=node_id,
                connected_segments=connected_segments,
                is_intersection=is_intersection
            )

            self.nodes[node_id] = node

    def _create_intersections(self):
        """Create ARZ Intersection objects for nodes with multiple connections"""
        for node in self.nodes.values():
            if node.is_intersection:
                # TODO: Create intersection objects when core modules are available
                # Convert position to float for ARZ
                # position = 0.0  # Will be set based on spatial layout later

                # intersection = Intersection(
                #     node_id=node.node_id,
                #     position=position,
                #     segments=node.connected_segments,
                #     traffic_lights=node.traffic_lights
                # )

                # self.intersections.append(intersection)
                pass

    def get_segment_by_nodes(self, start_node: str, end_node: str) -> Optional[RoadSegment]:
        """Get segment by start and end nodes"""
        segment_id = f"{start_node}_{end_node}"
        return self.segments.get(segment_id)

    def get_node_connections(self, node_id: str) -> List[str]:
        """Get all segments connected to a node"""
        node = self.nodes.get(node_id)
        return node.connected_segments if node and node.connected_segments else []

    def get_intersection_segments(self, intersection_id: str) -> List[str]:
        """Get segments connected to an intersection"""
        for intersection in self.intersections:
            if intersection.node_id == intersection_id:
                return intersection.segments
        return []

    def validate_network(self) -> List[str]:
        """Validate network integrity"""
        errors = []

        # Check for disconnected segments
        for segment_id, segment in self.segments.items():
            if segment.start_node not in self.nodes:
                errors.append(f"Segment {segment_id}: start node {segment.start_node} not found")
            if segment.end_node not in self.nodes:
                errors.append(f"Segment {segment_id}: end node {segment.end_node} not found")

        # Check for isolated nodes
        for node_id, node in self.nodes.items():
            if not node.connected_segments:
                errors.append(f"Node {node_id}: no connected segments")

        return errors

    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        highway_types = {}
        total_length = 0.0

        for segment in self.segments.values():
            highway_types[segment.highway_type] = highway_types.get(segment.highway_type, 0) + 1
            total_length += segment.length

        return {
            'total_segments': len(self.segments),
            'total_nodes': len(self.nodes),
            'total_intersections': len(self.intersections),
            'total_length_km': total_length / 1000,
            'highway_type_distribution': highway_types,
            'avg_segment_length_m': total_length / len(self.segments) if self.segments else 0
        }
