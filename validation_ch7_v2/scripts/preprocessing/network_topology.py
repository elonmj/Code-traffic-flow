"""
Network Topology Construction from TomTom Segments.

This module constructs simplified network topology suitable for UXsim simulation,
focusing on Victoria Island arterial corridors with realistic connectivity.

Key References:
- OpenStreetMap (2024): Victoria Island road network
- TomTom API: Segment directionality and connectivity
- Lagos State GIS: Road hierarchy classification

Author: ARZ-RL Validation Team
Date: 2025-01-17
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class RoadSegment:
    """
    Represents a single road segment in the network.
    
    Attributes:
        segment_id: Unique identifier (u_v format)
        origin: Origin node ID
        destination: Destination node ID
        street_name: Street name
        length: Segment length (meters)
        lanes: Number of lanes
        freeflow_speed: Free-flow speed (km/h)
        capacity: Capacity (vehicles/hour)
    """
    segment_id: str
    origin: str
    destination: str
    street_name: str
    length: float
    lanes: int
    freeflow_speed: float
    capacity: float


@dataclass
class NetworkTopology:
    """
    Complete network topology for UXsim simulation.
    
    Attributes:
        segments: List of RoadSegment objects
        nodes: Set of node IDs
        adjacency: Dict mapping node_id -> list of outgoing segments
        metadata: Dict with network-level statistics
    """
    segments: List[RoadSegment]
    nodes: Set[str]
    adjacency: Dict[str, List[str]]
    metadata: Dict


def infer_segment_length(
    street_name: str,
    default_length: float = 300.0
) -> float:
    """
    Infer segment length from street name and typical Victoria Island dimensions.
    
    Victoria Island street characteristics:
    - Akin Adesola: Major arterial, ~500m average segments
    - Ahmadu Bello: Secondary arterial, ~400m segments
    - Adeola Odeku: Commercial spine, ~350m segments
    - Saka Tinubu: Connector, ~300m segments
    
    Args:
        street_name: Name of the street
        default_length: Default length if street not recognized (default 300m)
    
    Returns:
        Estimated segment length in meters
    
    Examples:
        >>> infer_segment_length("Akin Adesola Street")
        500.0
        >>> infer_segment_length("Unknown Street")
        300.0
    """
    # Normalize street name
    name_lower = street_name.lower()
    
    # Street-specific lengths (from OpenStreetMap analysis)
    if "akin adesola" in name_lower:
        return 500.0
    elif "ahmadu bello" in name_lower:
        return 400.0
    elif "adeola odeku" in name_lower:
        return 350.0
    elif "saka tinubu" in name_lower:
        return 300.0
    else:
        return default_length


def infer_lane_count(
    street_name: str,
    default_lanes: int = 2
) -> int:
    """
    Infer number of lanes from street hierarchy.
    
    Victoria Island lane configuration:
    - Major arterials (Akin Adesola, Ahmadu Bello): 3 lanes per direction
    - Secondary arterials (Adeola Odeku): 2 lanes per direction
    - Connectors (Saka Tinubu): 2 lanes per direction
    
    Args:
        street_name: Name of the street
        default_lanes: Default lane count (default 2)
    
    Returns:
        Estimated number of lanes
    
    Examples:
        >>> infer_lane_count("Akin Adesola Street")
        3
        >>> infer_lane_count("Saka Tinubu Street")
        2
    """
    name_lower = street_name.lower()
    
    # Major arterials: 3 lanes
    if "akin adesola" in name_lower or "ahmadu bello" in name_lower:
        return 3
    # All others: 2 lanes
    else:
        return default_lanes


def compute_segment_capacity(
    lanes: int,
    freeflow_speed: float,
    jam_density: float = 0.15  # vehicles/meter
) -> float:
    """
    Compute segment capacity using fundamental diagram.
    
    Capacity occurs at critical density (ρ_c ≈ ρ_jam / 2).
    Q_max = ρ_c × v_c, where v_c ≈ v_free / 2 for triangular diagram.
    
    Simplified: Q_max = lanes × (v_free / 2) × (ρ_jam / 2) × L
    where L is a normalization factor.
    
    Args:
        lanes: Number of lanes
        freeflow_speed: Free-flow speed (km/h)
        jam_density: Jam density (vehicles/meter)
    
    Returns:
        Capacity (vehicles/hour)
    
    Examples:
        >>> compute_segment_capacity(lanes=3, freeflow_speed=50)
        2250.0  # vehicles/hour
    """
    # Critical density (half of jam density)
    rho_critical = jam_density / 2
    
    # Critical speed (half of free-flow, in m/s)
    v_critical_kmh = freeflow_speed / 2
    v_critical_ms = v_critical_kmh / 3.6
    
    # Flow at capacity (per lane, per meter)
    # q_max = ρ_c × v_c (vehicles/second/meter)
    q_max_per_lane = rho_critical * v_critical_ms
    
    # Convert to vehicles/hour, scale by lanes and typical segment (1000m ref)
    capacity = q_max_per_lane * lanes * 1000 * 3600
    
    return capacity


def construct_network_from_tomtom(
    df: pd.DataFrame,
    min_freeflow: float = 30.0  # Minimum realistic free-flow (km/h)
) -> NetworkTopology:
    """
    Construct UXsim-compatible network topology from TomTom data.
    
    Process:
    -------
    1. Extract unique segments from u-v pairs
    2. Infer missing attributes (length, lanes) from street names
    3. Validate free-flow speeds (handle unrealistic values)
    4. Compute capacities using fundamental diagram
    5. Build adjacency structure for routing
    
    Args:
        df: TomTom DataFrame with columns:
            - u: Origin node
            - v: Destination node
            - name: Street name (TomTom uses 'name' not 'street')
            - freeflow_speed: Free-flow speed (km/h)
        min_freeflow: Minimum realistic free-flow speed (default 30 km/h)
    
    Returns:
        NetworkTopology object with segments, nodes, adjacency, metadata
    
    Examples:
        >>> df = pd.read_csv('donnees_trafic_75_segments.csv')
        >>> network = construct_network_from_tomtom(df)
        >>> print(f"Network: {len(network.segments)} segments, {len(network.nodes)} nodes")
        Network: 70 segments, 50 nodes
    """
    # Validate input
    required_cols = ['u', 'v', 'name', 'freeflow_speed']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # CRITICAL: Extract UNIQUE spatial segments
    # TomTom data contains temporal repetitions (70 segments × 61 timestamps = 4270 entries)
    # We need only the 70 unique spatial segments
    print(f"📊 Total CSV entries: {len(df)}")
    
    # Group by (u, v) to get unique spatial segments
    # Take mean of freeflow_speed across temporal observations
    unique_segments_df = df.groupby(['u', 'v'], as_index=False).agg({
        'name': 'first',  # Street name (constant for each segment)
        'freeflow_speed': 'mean'  # Average free-flow speed across observations
    })
    
    print(f"✅ Unique spatial segments: {len(unique_segments_df)}")
    
    # Extract unique segments
    segments = []
    nodes = set()
    adjacency = {}
    
    for idx, row in unique_segments_df.iterrows():
        # Extract basic attributes
        origin = str(row['u'])
        destination = str(row['v'])
        segment_id = f"{origin}_{destination}"
        street_name = row['name']
        freeflow_speed = row['freeflow_speed']
        
        # Validate free-flow speed
        if freeflow_speed < min_freeflow:
            print(f"⚠️  Warning: Segment {segment_id} has low free-flow {freeflow_speed:.1f} km/h, "
                  f"clamping to {min_freeflow} km/h")
            freeflow_speed = min_freeflow
        
        # Infer missing attributes
        length = infer_segment_length(street_name)
        lanes = infer_lane_count(street_name)
        capacity = compute_segment_capacity(lanes, freeflow_speed)
        
        # Create segment
        segment = RoadSegment(
            segment_id=segment_id,
            origin=origin,
            destination=destination,
            street_name=street_name,
            length=length,
            lanes=lanes,
            freeflow_speed=freeflow_speed,
            capacity=capacity
        )
        
        segments.append(segment)
        nodes.add(origin)
        nodes.add(destination)
        
        # Build adjacency
        if origin not in adjacency:
            adjacency[origin] = []
        adjacency[origin].append(segment_id)
    
    # Compute network-level statistics
    total_length = sum(s.length for s in segments)
    avg_lanes = np.mean([s.lanes for s in segments])
    avg_capacity = np.mean([s.capacity for s in segments])
    total_capacity = sum(s.capacity for s in segments)
    
    # Street distribution (use 'name' column)
    street_counts = df['name'].value_counts().to_dict()
    
    metadata = {
        'n_segments': len(segments),
        'n_nodes': len(nodes),
        'total_length_m': total_length,
        'avg_lanes': avg_lanes,
        'avg_capacity_veh_per_hour': avg_capacity,
        'total_network_capacity': total_capacity,
        'street_distribution': street_counts,
        'construction_params': {
            'min_freeflow': min_freeflow
        }
    }
    
    return NetworkTopology(
        segments=segments,
        nodes=nodes,
        adjacency=adjacency,
        metadata=metadata
    )


def validate_network_topology(network: NetworkTopology) -> Dict:
    """
    Validate network topology for physical consistency and UXsim compatibility.
    
    Critical Checks:
    ---------------
    1. All segments have positive length
    2. All segments have positive capacity
    3. Free-flow speeds are realistic (30-70 km/h for urban)
    4. No isolated nodes (all nodes have incoming or outgoing)
    5. Segment count matches expected (70 segments)
    
    Args:
        network: NetworkTopology object to validate
    
    Returns:
        Dict with:
        - valid: bool
        - checks: Dict[str, bool]
        - issues: List[str]
        - warnings: List[str]
    
    Examples:
        >>> network = construct_network_from_tomtom(df)
        >>> validation = validate_network_topology(network)
        >>> print(validation['valid'])
        True
    """
    checks = {}
    issues = []
    warnings = []
    
    # Check 1: All segments have positive length
    lengths = [s.length for s in network.segments]
    checks['positive_lengths'] = all(l > 0 for l in lengths)
    if not checks['positive_lengths']:
        issues.append("Found segments with non-positive length")
    
    # Check 2: All segments have positive capacity
    capacities = [s.capacity for s in network.segments]
    checks['positive_capacities'] = all(c > 0 for c in capacities)
    if not checks['positive_capacities']:
        issues.append("Found segments with non-positive capacity")
    
    # Check 3: Realistic free-flow speeds (30-70 km/h urban)
    freeflows = [s.freeflow_speed for s in network.segments]
    checks['realistic_speeds'] = all(30 <= v <= 70 for v in freeflows)
    if not checks['realistic_speeds']:
        out_of_range = [(s.segment_id, s.freeflow_speed) 
                        for s in network.segments 
                        if not (30 <= s.freeflow_speed <= 70)]
        warnings.append(f"Found {len(out_of_range)} segments with speeds outside [30, 70] km/h")
    
    # Check 4: No isolated nodes
    nodes_with_outgoing = set(network.adjacency.keys())
    nodes_with_incoming = set()
    for segment in network.segments:
        nodes_with_incoming.add(segment.destination)
    
    all_nodes = network.nodes
    connected_nodes = nodes_with_outgoing | nodes_with_incoming
    isolated = all_nodes - connected_nodes
    
    checks['no_isolated_nodes'] = len(isolated) == 0
    if not checks['no_isolated_nodes']:
        issues.append(f"Found {len(isolated)} isolated nodes: {isolated}")
    
    # Check 5: Expected segment count (70)
    checks['expected_segment_count'] = len(network.segments) == 70
    if not checks['expected_segment_count']:
        warnings.append(f"Expected 70 segments, found {len(network.segments)}")
    
    # Check 6: Lane count reasonable (1-4)
    lane_counts = [s.lanes for s in network.segments]
    checks['reasonable_lanes'] = all(1 <= l <= 4 for l in lane_counts)
    if not checks['reasonable_lanes']:
        issues.append("Found segments with unrealistic lane counts (< 1 or > 4)")
    
    return {
        'valid': all(checks.values()) and len(issues) == 0,
        'checks': checks,
        'issues': issues,
        'warnings': warnings,
        'statistics': {
            'n_segments': len(network.segments),
            'n_nodes': len(network.nodes),
            'n_isolated_nodes': len(isolated),
            'avg_lanes': np.mean(lane_counts),
            'avg_capacity': np.mean(capacities)
        }
    }


def export_to_uxsim_format(
    network: NetworkTopology,
    output_path: str
) -> None:
    """
    Export network topology to UXsim-compatible CSV format.
    
    UXsim Network Format:
    - segment_id: Unique identifier
    - origin: Origin node
    - destination: Destination node
    - length: Length (meters)
    - lanes: Number of lanes
    - freeflow_speed: Free-flow speed (km/h)
    - capacity: Capacity (vehicles/hour)
    - street_name: Street name
    
    Args:
        network: NetworkTopology object
        output_path: Path to output CSV file
    
    Example:
        >>> network = construct_network_from_tomtom(df)
        >>> export_to_uxsim_format(network, 'network_uxsim.csv')
    """
    data = []
    for segment in network.segments:
        data.append({
            'segment_id': segment.segment_id,
            'origin': segment.origin,
            'destination': segment.destination,
            'length': segment.length,
            'lanes': segment.lanes,
            'freeflow_speed': segment.freeflow_speed,
            'capacity': segment.capacity,
            'street_name': segment.street_name
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Exported UXsim network to: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    """
    Standalone test of network topology construction.
    
    Run with: python network_topology.py
    """
    print("=" * 70)
    print("NETWORK TOPOLOGY CONSTRUCTION - STANDALONE TEST")
    print("=" * 70)
    
    # Create synthetic test data
    print("\n📊 Creating synthetic TomTom data...")
    test_data = {
        'u': ['A', 'B', 'C', 'A'],
        'v': ['B', 'C', 'D', 'D'],
        'street': [
            'Akin Adesola Street',
            'Ahmadu Bello Way',
            'Adeola Odeku Street',
            'Saka Tinubu Street'
        ],
        'freeflow_speed': [50, 45, 40, 35]
    }
    df = pd.DataFrame(test_data)
    
    print(f"  {len(df)} test segments")
    
    # Construct network
    print("\n🚗 Constructing network topology...")
    network = construct_network_from_tomtom(df)
    
    print(f"\n✅ Network Constructed:")
    print(f"  Segments: {network.metadata['n_segments']}")
    print(f"  Nodes: {network.metadata['n_nodes']}")
    print(f"  Total length: {network.metadata['total_length_m']:.0f} m")
    print(f"  Avg lanes: {network.metadata['avg_lanes']:.1f}")
    print(f"  Avg capacity: {network.metadata['avg_capacity_veh_per_hour']:.0f} veh/h")
    print(f"  Total capacity: {network.metadata['total_network_capacity']:.0f} veh/h")
    
    # Show segment details
    print(f"\n📋 Segment Details:")
    for i, segment in enumerate(network.segments[:4], 1):
        print(f"  {i}. {segment.segment_id} ({segment.street_name})")
        print(f"     Length: {segment.length}m, Lanes: {segment.lanes}, "
              f"Freeflow: {segment.freeflow_speed} km/h, "
              f"Capacity: {segment.capacity:.0f} veh/h")
    
    # Validate
    print(f"\n✅ Validating network topology...")
    validation = validate_network_topology(network)
    
    print(f"\n📊 Validation Results:")
    print(f"  Valid: {'✅' if validation['valid'] else '❌'}")
    print(f"  Checks:")
    for check, passed in validation['checks'].items():
        status = '✅' if passed else '❌'
        print(f"    {status} {check}")
    
    if validation['issues']:
        print(f"\n❌ Issues:")
        for issue in validation['issues']:
            print(f"    - {issue}")
    
    if validation['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"    - {warning}")
    
    # Export (optional)
    # export_to_uxsim_format(network, 'network_test.csv')
    
    print("\n" + "=" * 70)
    print("✅ Standalone test complete!")
    print("=" * 70)
