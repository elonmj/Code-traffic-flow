"""
Victoria Island Corridor Group Configuration
==========================================

Configuration for the Victoria Island corridor network group.
This group contains the main roads in the Victoria Island area of Lagos.
"""

from .group_manager import GroupManager, NetworkGroup
import os

# Configuration for Victoria Island corridor group
VICTORIA_ISLAND_CONFIG = {
    'group_id': 'victoria_island_corridor',
    'name': 'Victoria Island Corridor',
    'description': 'Main corridor network in Victoria Island, Lagos including Akin Adesola Street, Ahmadu Bello Way, Adeola Odeku Street, and Saka Tinubu Street',
    'parameters': {
        # Group-specific ARZ parameters
        'max_speed_primary': 60.0,      # km/h for primary roads
        'max_speed_secondary': 50.0,    # km/h for secondary roads
        'max_speed_tertiary': 40.0,     # km/h for tertiary roads

        # Traffic flow parameters by road type
        'flow_capacity_primary': 1800,   # vehicles/hour/lane
        'flow_capacity_secondary': 1500,
        'flow_capacity_tertiary': 1200,

        # Lane change parameters
        'lane_change_probability': 0.1,

        # Intersection parameters
        'intersection_delay_primary': 15.0,   # seconds
        'intersection_delay_secondary': 20.0,
        'intersection_delay_tertiary': 25.0,

        # Speed adaptation parameters
        'speed_adaptation_factor': 0.8,
        'congestion_threshold': 0.7,    # fraction of capacity

        # Time-of-day factors
        'peak_hour_factor': 1.2,
        'off_peak_factor': 0.9,

        # Weather/situation factors
        'weather_impact_factor': 1.0,
        'incident_impact_factor': 0.5
    }
}

# Road type mappings for the corridor
ROAD_TYPE_MAPPING = {
    'primary': {
        'max_speed': 60.0,
        'capacity_per_lane': 1800,
        'lane_change_penalty': 2.0,
        'intersection_penalty': 15.0
    },
    'secondary': {
        'max_speed': 50.0,
        'capacity_per_lane': 1500,
        'lane_change_penalty': 3.0,
        'intersection_penalty': 20.0
    },
    'tertiary': {
        'max_speed': 40.0,
        'capacity_per_lane': 1200,
        'lane_change_penalty': 4.0,
        'intersection_penalty': 25.0
    }
}

def create_victoria_island_group(csv_file: str, save_format: str = 'json') -> str:
    """
    Create and save the Victoria Island corridor group

    Args:
        csv_file: Path to the corridor CSV file
        save_format: Format to save the group ('json' or 'yaml')

    Returns:
        Path to saved group file
    """
    manager = GroupManager()

    # Create group from CSV
    group = manager.create_group_from_csv(csv_file, VICTORIA_ISLAND_CONFIG)

    # Add additional metadata
    group.metadata.update({
        'location': 'Victoria Island, Lagos',
        'country': 'Nigeria',
        'data_source': 'OpenStreetMap',
        'last_updated': '2024-01-01',
        'coordinate_system': 'WGS84',
        'road_types_present': list(set(s.highway_type for s in group.segments)),
        'main_streets': [
            'Akin Adesola Street',
            'Ahmadu Bello Way',
            'Adeola Odeku Street',
            'Saka Tinubu Street'
        ]
    })

    # Save group
    saved_path = manager.save_group(group, save_format)
    print(f"Victoria Island corridor group saved to: {saved_path}")

    return saved_path

def load_victoria_island_group() -> 'NetworkGroup':
    """
    Load the Victoria Island corridor group

    Returns:
        NetworkGroup object
    """
    manager = GroupManager()
    return manager.load_group('victoria_island_corridor')

def get_group_statistics(group_id: str = 'victoria_island_corridor') -> dict:
    """
    Get statistics for a network group

    Args:
        group_id: ID of the group

    Returns:
        Dictionary with group statistics
    """
    manager = GroupManager()
    return manager.get_group_summary(group_id)

if __name__ == "__main__":
    # Example usage
    csv_path = "Code_RL/data/fichier_de_travail_corridor.csv"

    if os.path.exists(csv_path):
        # Create and save the group
        group_path = create_victoria_island_group(csv_path)

        # Load and display statistics
        stats = get_group_statistics()
        print("\nVictoria Island Corridor Statistics:")
        print(f"- Segments: {stats['segment_count']}")
        print(".2f")
        print(f"- Road types: {', '.join(stats['road_types_present'])}")
        print(f"- Main streets: {len(stats.get('main_streets', []))}")
    else:
        print(f"CSV file not found: {csv_path}")
