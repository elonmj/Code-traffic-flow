"""
CSV Parser for Road Network Data.

This module provides a robust mechanism for parsing the road network data from
a CSV file and transforming it into a structured `RoadNetwork` object using
the Pydantic models.

The parsing strategy is designed to be fault-tolerant, handling missing values
and potential data type inconsistencies gracefully.
"""
import pandas as pd
from typing import Dict, List
from .models import Node, Link, RoadNetwork

def parse_csv_to_road_network(file_path: str) -> RoadNetwork:
    """
    Parses a CSV file and converts it into a RoadNetwork object.

    Args:
        file_path: The absolute path to the CSV file.

    Returns:
        A validated RoadNetwork object containing all nodes and links from the file.
    """
    df = pd.read_csv(file_path)

    # Clean and rename columns to match Pydantic model fields
    df.rename(columns={
        'name_clean': 'name',
        'length': 'length_m',
        'highway': 'highway_type',
        'maxspeed_manual_kmh': 'max_speed_kmh',
        'lanes_manual': 'lanes'
    }, inplace=True)

    # --- Data Cleaning and Type Conversion ---
    # Fill missing optional values with defaults that can be processed
    df['lanes'] = pd.to_numeric(df['lanes'], errors='coerce').fillna(1).astype(int)
    df['max_speed_kmh'] = pd.to_numeric(df['max_speed_kmh'], errors='coerce').fillna(50.0).astype(float)
    df['oneway'] = df['oneway'].astype(bool)

    links: List[Link] = []
    nodes: Dict[int, Node] = {}

    for _, row in df.iterrows():
        # Create Link object from row data
        link_data = row.to_dict()
        
        # Ensure 'u' and 'v' are treated as integers
        u_node_id = int(row['u'])
        v_node_id = int(row['v'])
        
        link_data['u'] = u_node_id
        link_data['v'] = v_node_id
        
        # Keep only fields that are part of the Link model
        valid_fields = {f for f in Link.model_fields}
        filtered_link_data = {k: v for k, v in link_data.items() if k in valid_fields}

        link = Link(**filtered_link_data)
        links.append(link)

        # Add nodes to the node dictionary, ensuring no duplicates
        # Use placeholder positions as coordinates are not in the CSV
        if u_node_id not in nodes:
            nodes[u_node_id] = Node(id=u_node_id, position=[0.0, 0.0])
        if v_node_id not in nodes:
            nodes[v_node_id] = Node(id=v_node_id, position=[0.0, 0.0])

    return RoadNetwork(nodes=nodes, links=links)
