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
import logging

from .models import RoadNetwork, Node, Link

logger = logging.getLogger(__name__)

def _get_safe_value(value, default=0):
    """Safely convert value to float, returning default if conversion fails."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def parse_csv_to_road_network(file_path: str) -> RoadNetwork:
    """
    Parses a CSV file containing road network data into a RoadNetwork object.

    The function performs the following steps:
    1. Reads the CSV into a pandas DataFrame.
    2. Extracts unique nodes from the 'u' and 'v' columns.
    3. Creates Link objects for each row in the DataFrame.
    4. Validates the entire structure using the RoadNetwork Pydantic model.

    Args:
        file_path: The absolute path to the CSV file.

    Returns:
        A validated RoadNetwork object.
    """
    logger.info(f"Parsing road network data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        raise

    # --- Node Extraction ---
    # We assume nodes don't have their own coordinates in this file format.
    # We will create placeholder nodes based on their IDs.
    nodes: Dict[str, Node] = {}
    all_node_ids = pd.unique(df[['u', 'v']].values.ravel('K'))
    
    for node_id in all_node_ids:
        # Since we don't have coordinates, we use a placeholder.
        # In a real scenario, you might have a separate nodes file.
        nodes[str(node_id)] = Node(node_id=str(node_id), x=0.0, y=0.0, node_type="junction")

    # --- Link Creation ---
    links: List[Link] = []
    for _, row in df.iterrows():
        start_node_id = str(row['u'])
        end_node_id = str(row['v'])
        
        # Create a unique ID for the link
        link_id = f"{start_node_id}_{end_node_id}"
        
        # Safely get numerical values
        lanes = int(_get_safe_value(row.get('lanes_manual'), default=1))
        road_quality = int(_get_safe_value(row.get('Rx_manual'), default=3))
        max_speed = _get_safe_value(row.get('maxspeed_manual_kmh'), default=50.0)

        link = Link(
            link_id=link_id,
            name=row.get('name_clean', 'Unknown'),
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            length_m=float(row['length']),
            lanes=lanes,
            road_quality=road_quality,
            max_speed_kmh=max_speed,
            oneway=bool(row.get('oneway', False))
        )
        links.append(link)

    # --- Final Assembly and Validation ---
    road_network = RoadNetwork(nodes=nodes, links=links)
    
    logger.info(f"Successfully parsed {len(nodes)} nodes and {len(links)} links.")
    
    return road_network
