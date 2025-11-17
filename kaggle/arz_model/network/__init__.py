"""
Network infrastructure for multi-segment road networks.

This module implements professional network architecture inspired by SUMO's MSNet
and CityFlow's RoadNet patterns, following the academic formulation from 
Garavello & Piccoli (2005) "Traffic Flow on Networks".

Architecture:
    - NetworkGrid: Top-level coordinator managing segments/nodes/links
    - Node: Junction wrapper with topology (wraps existing Intersection class)
    - Link: Segment connection managing θ_k behavioral coupling
    - topology: Graph utilities for network validation and analysis

Academic References:
    - Garavello & Piccoli (2005): "Traffic Flow on Networks - Conservation Laws Models"
    - Kolb et al. (2018): Phenomenological network coupling with memory parameter
    - Göttlich et al. (2021): Second-order traffic models on networks

Author: ARZ Model Development Team
Date: 2025-01-21
"""

from .network_grid import NetworkGrid
from .node import Node
from .link import Link
from .topology import build_graph, validate_topology, find_upstream_segments, find_downstream_segments

__all__ = [
    'NetworkGrid',
    'Node', 
    'Link',
    'build_graph',
    'validate_topology',
    'find_upstream_segments',
    'find_downstream_segments'
]

__version__ = '0.1.0'
