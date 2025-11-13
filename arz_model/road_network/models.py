"""
Core Pydantic models for representing the road network structure.

This module defines the canonical, validated data structures for Nodes, Links,
and the overall RoadNetwork. These models are the "source of truth" and are
used throughout the simulation pipeline.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any

class Node(BaseModel):
    """
    Represents a node (intersection or boundary) in the road network.
    """
    node_id: str = Field(..., description="Unique identifier for the node.")
    x: float = Field(0.0, description="X-coordinate of the node.")
    y: float = Field(0.0, description="Y-coordinate of the node.")
    node_type: str = Field("junction", description="Type of node (e.g., 'junction', 'boundary').")

class Link(BaseModel):
    """
    Represents a directed link (road segment) between two nodes.
    """
    link_id: str = Field(..., description="Unique identifier for the link.")
    name: str = Field(..., description="Name of the road.")
    start_node_id: str = Field(..., description="ID of the starting node.")
    end_node_id: str = Field(..., description="ID of the ending node.")
    length_m: float = Field(..., description="Length of the link in meters.")
    lanes: int = Field(..., description="Number of lanes.")
    road_quality: int = Field(..., description="Road quality category (e.g., 1-5).")
    max_speed_kmh: float = Field(..., description="Posted speed limit in km/h.")
    oneway: bool = Field(False, description="Indicates if the link is one-way.")

    @validator('length_m', 'lanes', 'road_quality', 'max_speed_kmh')
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Link attributes (length, lanes, etc.) must be positive.")
        return v

class RoadNetwork(BaseModel):
    """
    Represents the entire road network, composed of nodes and links.
    """
    nodes: Dict[str, Node] = Field(..., description="Dictionary of all nodes in the network.")
    links: List[Link] = Field(..., description="List of all links in the network.")

    @validator('links')
    def check_node_references(cls, links, values):
        """
        Ensures that all links refer to nodes that actually exist in the network.
        """
        node_ids = values.get('nodes', {}).keys()
        for link in links:
            if link.start_node_id not in node_ids:
                raise ValueError(f"Link '{link.link_id}' refers to a non-existent start node '{link.start_node_id}'.")
            if link.end_node_id not in node_ids:
                raise ValueError(f"Link '{link.link_id}' refers to a non-existent end node '{link.end_node_id}'.")
        return links
