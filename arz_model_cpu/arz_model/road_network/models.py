"""
Core Pydantic models for representing the road network structure.

This module defines the canonical, validated data structures for Nodes, Links,
and the overall RoadNetwork. These models are the "source of truth" and are
used throughout the simulation pipeline.
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class Node(BaseModel):
    """Represents a single node (intersection or point) in the road network."""
    id: int = Field(..., description="Unique identifier for the node (from OSM u/v).")
    position: List[float] = Field(..., description="[x, y] coordinates of the node.")

class Link(BaseModel):
    """
    Represents a directed road segment (edge) between two nodes.
    
    This model captures the static properties of a road segment as defined in the
    source data. It includes robust handling of optional fields with sensible defaults.
    """
    u: int = Field(..., description="The identifier of the origin node.")
    v: int = Field(..., description="The identifier of the destination node.")
    name: str = Field(..., description="The name of the road (e.g., 'Akin Adesola Street').")
    length_m: float = Field(..., description="The length of the link in meters.")
    highway_type: str = Field(..., description="The type of highway (e.g., 'primary', 'secondary').")
    oneway: bool = Field(False, description="Flag indicating if the link is one-way.")
    
    # Optional fields with default values
    lanes: Optional[int] = Field(1, description="Number of lanes on the link. Defaults to 1.")
    max_speed_kmh: Optional[float] = Field(50.0, description="Maximum speed limit in km/h. Defaults to 50.0.")
    
    # Unused fields from CSV, captured for completeness but not used in simulation yet.
    rx_manual: Optional[str] = Field(None, description="Placeholder for the 'Rx_manual' column.")

class RoadNetwork(BaseModel):
    """
    Represents the entire road network as a collection of nodes and links.
    
    This acts as the main container, providing a graph-like structure that can be
    easily accessed and traversed.
    """
    nodes: Dict[int, Node] = Field(..., description="A dictionary mapping node IDs to Node objects.")
    links: List[Link] = Field(..., description="A list of all links in the network.")

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
