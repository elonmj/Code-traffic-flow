#!/usr/bin/env python3
"""
📊 CREATE ENRICHED CORRIDOR DATA
=================================

Fetches REAL data from OpenStreetMap and creates a comprehensive CSV
with all corridor information:
- Road segments with real coordinates
- Traffic signals locations
- Road attributes (lanes, speed limits, etc.)

This data can then be used by other parts of the system.
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

# Victoria Island bounding box
BBOX = {'south': 6.42, 'west': 3.40, 'north': 6.46, 'east': 3.46}


def fetch_roads():
    """Fetch road network from OSM."""
    query = """
    [out:json][timeout:90];
    (
      way["highway"]["name"~"Ahmadu Bello|Akin Adesola|Adeola Odeku|Saka Tinubu"]
        (6.42,3.40,6.46,3.46);
    );
    out body;
    >;
    out skel qt;
    """
    
    # List of Overpass API mirrors
    servers = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ]
    
    print("📡 Fetching road network from OpenStreetMap...")
    
    for server in servers:
        try:
            print(f"   Trying: {server.split('//')[1].split('/')[0]}...")
            response = requests.post(
                server,
                data={'data': query},
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            print(f"   ✅ Received {len(data.get('elements', []))} elements")
            return data
        except Exception as e:
            print(f"   ⚠️ Failed: {str(e)[:50]}...")
            continue
    
    raise Exception("All Overpass servers failed")


def fetch_traffic_signals():
    """Fetch traffic signals in the area."""
    query = """
    [out:json][timeout:60];
    (
      node["highway"="traffic_signals"](6.42,3.40,6.46,3.46);
    );
    out body;
    """
    
    # List of Overpass API mirrors
    servers = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ]
    
    print("🚦 Fetching traffic signals from OpenStreetMap...")
    
    for server in servers:
        try:
            print(f"   Trying: {server.split('//')[1].split('/')[0]}...")
            response = requests.post(
                server,
                data={'data': query},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            signals = []
            for elem in data.get('elements', []):
                if elem['type'] == 'node':
                    signals.append({
                        'osm_id': elem['id'],
                        'lat': elem['lat'],
                        'lon': elem['lon'],
                        'type': 'traffic_signals'
                    })
            
            print(f"   ✅ Found {len(signals)} traffic signals")
            return signals
        except Exception as e:
            print(f"   ⚠️ Failed: {str(e)[:50]}...")
            continue
    
    raise Exception("All Overpass servers failed")


def parse_roads(data):
    """Parse OSM data into nodes and ways."""
    nodes = {}
    ways = []
    
    for elem in data.get('elements', []):
        if elem['type'] == 'node':
            nodes[elem['id']] = {
                'lat': elem['lat'],
                'lon': elem['lon']
            }
        elif elem['type'] == 'way':
            tags = elem.get('tags', {})
            ways.append({
                'osm_way_id': elem['id'],
                'name': tags.get('name', ''),
                'highway': tags.get('highway', ''),
                'lanes': tags.get('lanes', ''),
                'maxspeed': tags.get('maxspeed', ''),
                'oneway': tags.get('oneway', ''),
                'surface': tags.get('surface', ''),
                'lit': tags.get('lit', ''),
                'nodes': elem.get('nodes', [])
            })
    
    return nodes, ways


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def find_nearest_signal(lat, lon, signals, threshold_m=50):
    """Find if there's a traffic signal within threshold meters."""
    for sig in signals:
        dist = haversine_distance(lat, lon, sig['lat'], sig['lon'])
        if dist <= threshold_m:
            return sig['osm_id'], dist
    return None, None


def create_segments_dataframe(nodes, ways, signals):
    """Create a DataFrame with all road segments."""
    segments = []
    segment_id = 0
    
    for way in ways:
        road_name = way['name']
        way_nodes = way['nodes']
        
        for i in range(len(way_nodes) - 1):
            node_from = way_nodes[i]
            node_to = way_nodes[i + 1]
            
            if node_from in nodes and node_to in nodes:
                n1 = nodes[node_from]
                n2 = nodes[node_to]
                
                # Calculate segment length
                length = haversine_distance(n1['lat'], n1['lon'], n2['lat'], n2['lon'])
                
                # Check for traffic signals at start/end
                sig_from, dist_from = find_nearest_signal(n1['lat'], n1['lon'], signals)
                sig_to, dist_to = find_nearest_signal(n2['lat'], n2['lon'], signals)
                
                # Determine position in the way
                is_entry = (i == 0)
                is_exit = (i == len(way_nodes) - 2)
                
                segments.append({
                    'segment_id': segment_id,
                    'osm_way_id': way['osm_way_id'],
                    'road_name': road_name,
                    'highway_type': way['highway'],
                    'lanes': way['lanes'] if way['lanes'] else '2',
                    'maxspeed_kmh': way['maxspeed'].replace(' km/h', '').replace('kmh', '') if way['maxspeed'] else '50',
                    'oneway': way['oneway'] if way['oneway'] else 'no',
                    'surface': way['surface'] if way['surface'] else 'asphalt',
                    'lit': way['lit'] if way['lit'] else 'yes',
                    
                    # Start node
                    'node_from_osm_id': node_from,
                    'lat_from': n1['lat'],
                    'lon_from': n1['lon'],
                    
                    # End node
                    'node_to_osm_id': node_to,
                    'lat_to': n2['lat'],
                    'lon_to': n2['lon'],
                    
                    # Center point
                    'lat_center': (n1['lat'] + n2['lat']) / 2,
                    'lon_center': (n1['lon'] + n2['lon']) / 2,
                    
                    # Geometry
                    'length_m': round(length, 2),
                    'bearing_deg': calculate_bearing(n1['lat'], n1['lon'], n2['lat'], n2['lon']),
                    
                    # Traffic signals
                    'has_signal_from': 1 if sig_from else 0,
                    'signal_from_osm_id': sig_from if sig_from else '',
                    'has_signal_to': 1 if sig_to else 0,
                    'signal_to_osm_id': sig_to if sig_to else '',
                    
                    # Entry/Exit
                    'is_way_entry': 1 if is_entry else 0,
                    'is_way_exit': 1 if is_exit else 0,
                    
                    # Segment position in way
                    'segment_index_in_way': i,
                    'total_segments_in_way': len(way_nodes) - 1
                })
                
                segment_id += 1
    
    return pd.DataFrame(segments)


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return round((bearing + 360) % 360, 1)


def create_signals_dataframe(signals):
    """Create a DataFrame with all traffic signals."""
    df = pd.DataFrame(signals)
    df = df.rename(columns={'osm_id': 'signal_osm_id'})
    return df


def create_nodes_dataframe(nodes, signals):
    """Create a DataFrame with all intersection nodes."""
    rows = []
    
    for osm_id, coords in nodes.items():
        sig_id, sig_dist = find_nearest_signal(coords['lat'], coords['lon'], signals, threshold_m=30)
        
        rows.append({
            'node_osm_id': osm_id,
            'lat': coords['lat'],
            'lon': coords['lon'],
            'has_signal': 1 if sig_id else 0,
            'signal_osm_id': sig_id if sig_id else ''
        })
    
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("📊 CREATING ENRICHED CORRIDOR DATA")
    print("   Victoria Island, Lagos - From OpenStreetMap")
    print("=" * 70)
    
    # Fetch data from OSM
    road_data = fetch_roads()
    signals = fetch_traffic_signals()
    
    # Parse roads
    print("\n🔧 Parsing road data...")
    nodes, ways = parse_roads(road_data)
    print(f"   Nodes: {len(nodes)}, Ways: {len(ways)}")
    
    # Create DataFrames
    print("\n📝 Creating data tables...")
    
    # Segments
    df_segments = create_segments_dataframe(nodes, ways, signals)
    print(f"   Segments: {len(df_segments)}")
    
    # Traffic signals
    df_signals = create_signals_dataframe(signals)
    print(f"   Traffic signals: {len(df_signals)}")
    
    # Nodes/intersections
    df_nodes = create_nodes_dataframe(nodes, signals)
    print(f"   Nodes: {len(df_nodes)}")
    
    # Save to CSV
    output_dir = Path(__file__).parent
    
    # Main segments file
    segments_path = output_dir / 'corridor_segments_enriched.csv'
    df_segments.to_csv(segments_path, index=False, encoding='utf-8')
    print(f"\n✅ Saved: {segments_path}")
    
    # Traffic signals file
    signals_path = output_dir / 'corridor_traffic_signals.csv'
    df_signals.to_csv(signals_path, index=False, encoding='utf-8')
    print(f"✅ Saved: {signals_path}")
    
    # Nodes file
    nodes_path = output_dir / 'corridor_nodes.csv'
    df_nodes.to_csv(nodes_path, index=False, encoding='utf-8')
    print(f"✅ Saved: {nodes_path}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("📊 SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\n🛣️  ROADS:")
    for road_name in df_segments['road_name'].unique():
        road_df = df_segments[df_segments['road_name'] == road_name]
        total_length = road_df['length_m'].sum()
        n_segments = len(road_df)
        print(f"   • {road_name}: {n_segments} segments, {total_length:.0f}m total")
    
    total_length_km = df_segments['length_m'].sum() / 1000
    print(f"\n   TOTAL: {len(df_segments)} segments, {total_length_km:.2f} km")
    
    print(f"\n🚦 TRAFFIC SIGNALS:")
    print(f"   Total signals in area: {len(signals)}")
    signals_on_segments = df_segments['has_signal_from'].sum() + df_segments['has_signal_to'].sum()
    print(f"   Signals connected to segments: {int(signals_on_segments)}")
    
    print(f"\n📍 GEOGRAPHIC BOUNDS:")
    print(f"   Latitude:  {df_segments['lat_from'].min():.6f} to {df_segments['lat_from'].max():.6f}")
    print(f"   Longitude: {df_segments['lon_from'].min():.6f} to {df_segments['lon_from'].max():.6f}")
    
    # Sample data
    print("\n📋 SAMPLE DATA (first 5 segments):")
    print(df_segments[['segment_id', 'road_name', 'lat_from', 'lon_from', 'lat_to', 'lon_to', 'length_m']].head().to_string())
    
    # Also save a JSON with all data for easy loading
    all_data = {
        'metadata': {
            'source': 'OpenStreetMap',
            'bbox': BBOX,
            'fetch_date': pd.Timestamp.now().isoformat(),
            'total_segments': len(df_segments),
            'total_signals': len(df_signals),
            'total_nodes': len(df_nodes),
            'total_length_km': round(total_length_km, 2)
        },
        'roads': list(df_segments['road_name'].unique()),
        'segments_summary': df_segments.groupby('road_name').agg({
            'segment_id': 'count',
            'length_m': 'sum'
        }).to_dict()
    }
    
    json_path = output_dir / 'corridor_metadata.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved: {json_path}")
    
    print("\n" + "=" * 70)
    print("✅ ENRICHED DATA CREATION COMPLETE!")
    print("=" * 70)
    
    return df_segments, df_signals, df_nodes


if __name__ == '__main__':
    main()
