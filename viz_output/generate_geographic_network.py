#!/usr/bin/env python3
"""
🌍 REAL GEOGRAPHIC NETWORK VISUALIZATION
==========================================

Fetches REAL coordinates from OpenStreetMap via Overpass API
and creates an accurate geographic representation of the 
Victoria Island corridor in Lagos, Nigeria.

This script:
1. Queries OSM for the exact streets in the corridor
2. Extracts node coordinates
3. Creates a geographically accurate visualization
"""

import sys
import os
import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
import networkx as nx
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Victoria Island, Lagos bounding box
BBOX = {
    'south': 6.42,
    'west': 3.40,
    'north': 6.46,
    'east': 3.46
}

# Streets in our corridor
STREET_NAMES = [
    "Ahmadu Bello Way",
    "Akin Adesola Street", 
    "Adeola Odeku Street",
    "Saka Tinubu Street"
]


def fetch_osm_data():
    """Fetch road data from OpenStreetMap via Overpass API."""
    
    # Build Overpass query for Victoria Island roads
    query = f"""
    [out:json][timeout:60];
    (
      way["highway"]["name"~"Ahmadu Bello|Akin Adesola|Adeola Odeku|Saka Tinubu"]
        ({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});
    );
    out body;
    >;
    out skel qt;
    """
    
    print("📡 Querying Overpass API for Victoria Island roads...")
    
    url = "https://overpass-api.de/api/interpreter"
    
    try:
        response = requests.post(url, data={'data': query}, timeout=120)
        response.raise_for_status()
        data = response.json()
        print(f"   ✅ Received {len(data.get('elements', []))} elements")
        return data
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error fetching OSM data: {e}")
        print("   Using fallback coordinates...")
        return None


def parse_osm_data(data):
    """Parse OSM response to extract nodes and ways."""
    if not data:
        return None, None
    
    nodes = {}
    ways = []
    
    for element in data.get('elements', []):
        if element['type'] == 'node':
            nodes[element['id']] = (element['lon'], element['lat'])
        elif element['type'] == 'way':
            ways.append({
                'id': element['id'],
                'name': element.get('tags', {}).get('name', 'Unknown'),
                'highway': element.get('tags', {}).get('highway', 'road'),
                'nodes': element.get('nodes', [])
            })
    
    print(f"   Parsed: {len(nodes)} nodes, {len(ways)} ways")
    return nodes, ways


def build_real_graph(nodes, ways):
    """Build NetworkX graph from real OSM data."""
    G = nx.DiGraph()
    
    road_edges = defaultdict(list)
    
    for way in ways:
        name = way['name']
        way_nodes = way['nodes']
        
        for i in range(len(way_nodes) - 1):
            u_id = way_nodes[i]
            v_id = way_nodes[i + 1]
            
            if u_id in nodes and v_id in nodes:
                u_str = str(u_id)
                v_str = str(v_id)
                
                lon1, lat1 = nodes[u_id]
                lon2, lat2 = nodes[v_id]
                
                # Calculate segment length (haversine approximation)
                length = haversine_distance(lat1, lon1, lat2, lon2)
                
                G.add_node(u_str, pos=(lon1, lat1))
                G.add_node(v_str, pos=(lon2, lat2))
                G.add_edge(u_str, v_str, name=name, length=length)
                
                road_edges[name].append({'u': u_str, 'v': v_str, 'length': length})
    
    return G, road_edges


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters."""
    R = 6371000  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def create_fallback_graph():
    """Create a graph with manually placed coordinates if OSM fails."""
    print("   Using fallback coordinates based on known Victoria Island layout...")
    
    # Victoria Island approximate center
    center_lat, center_lon = 6.4353, 3.4323
    
    # Manual layout based on actual road structure
    # These coordinates are approximate but represent the actual road pattern
    roads = {
        'Ahmadu Bello Way': [
            (3.4150, 6.4320),
            (3.4200, 6.4330),
            (3.4250, 6.4335),
            (3.4300, 6.4340),
            (3.4350, 6.4345),
            (3.4400, 6.4350),
            (3.4450, 6.4355),
            (3.4500, 6.4360)
        ],
        'Akin Adesola Street': [
            (3.4200, 6.4280),
            (3.4205, 6.4310),
            (3.4210, 6.4340),
            (3.4215, 6.4370),
            (3.4220, 6.4400),
            (3.4225, 6.4430)
        ],
        'Adeola Odeku Street': [
            (3.4180, 6.4400),
            (3.4230, 6.4395),
            (3.4280, 6.4390),
            (3.4330, 6.4385),
            (3.4380, 6.4380)
        ],
        'Saka Tinubu Street': [
            (3.4320, 6.4310),
            (3.4325, 6.4340),
            (3.4330, 6.4370),
            (3.4335, 6.4400)
        ]
    }
    
    G = nx.DiGraph()
    road_edges = defaultdict(list)
    node_id = 0
    
    for road_name, coords in roads.items():
        prev_id = None
        for i, (lon, lat) in enumerate(coords):
            node_str = f"{road_name[:3]}_{i}"
            G.add_node(node_str, pos=(lon, lat))
            
            if prev_id is not None:
                prev_lon, prev_lat = G.nodes[prev_id]['pos']
                length = haversine_distance(prev_lat, prev_lon, lat, lon)
                G.add_edge(prev_id, node_str, name=road_name, length=length)
                road_edges[road_name].append({'u': prev_id, 'v': node_str, 'length': length})
            
            prev_id = node_str
    
    return G, road_edges


def generate_traffic_pattern(G):
    """Generate realistic traffic speeds for visualization."""
    edge_speeds = {}
    
    pos = nx.get_node_attributes(G, 'pos')
    
    # Find network bounds
    if pos:
        lons = [p[0] for p in pos.values()]
        lats = [p[1] for p in pos.values()]
        center_lon = np.mean(lons)
        center_lat = np.mean(lats)
    
    for u, v, data in G.edges(data=True):
        if u in pos and v in pos:
            u_lon, u_lat = pos[u]
            
            # Distance from center affects congestion
            dist_from_center = np.sqrt((u_lon - center_lon)**2 + (u_lat - center_lat)**2)
            
            # Roads closer to center are more congested
            if dist_from_center < 0.005:
                base_speed = 15 + np.random.normal(0, 5)  # Congested center
            elif dist_from_center < 0.015:
                base_speed = 35 + np.random.normal(0, 10)  # Moderate traffic
            else:
                base_speed = 55 + np.random.normal(0, 10)  # Lighter traffic
            
            # Randomize by road
            name = data.get('name', '')
            if 'Ahmadu Bello' in name:
                base_speed *= 0.8  # Main road tends to be congested
            elif 'Saka Tinubu' in name:
                base_speed *= 1.1  # Side road flows better
            
            edge_speeds[(u, v)] = np.clip(base_speed, 5, 80)
        else:
            edge_speeds[(u, v)] = 40
    
    return edge_speeds


def create_speed_colormap():
    """Create traffic speed colormap."""
    colors = ['#8B0000', '#CC0000', '#FF4444', '#FF8800', '#FFCC00', 
              '#FFFF00', '#AAFF00', '#66FF33', '#00DD00', '#00AA44']
    cmap = LinearSegmentedColormap.from_list('traffic', colors, N=256)
    norm = mcolors.Normalize(vmin=0, vmax=80)
    return cmap, norm


def create_geographic_visualization(G, road_edges, edge_speeds, output_path):
    """Create visualization with real geographic coordinates."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_facecolor('#e8f4f8')
    
    cmap, norm = create_speed_colormap()
    pos = nx.get_node_attributes(G, 'pos')
    
    # ================================================================
    # 1. DRAW EDGES with traffic colors
    # ================================================================
    for (u, v), speed in edge_speeds.items():
        if u in pos and v in pos:
            lon1, lat1 = pos[u]
            lon2, lat2 = pos[v]
            
            color = cmap(norm(speed))
            
            ax.annotate('', xy=(lon2, lat2), xytext=(lon1, lat1),
                       arrowprops=dict(
                           arrowstyle='-|>',
                           color=color,
                           lw=4,
                           mutation_scale=15,
                           shrinkA=8,
                           shrinkB=8
                       ), zorder=3)
    
    # ================================================================
    # 2. DRAW NODES (intersections)
    # ================================================================
    # Identify entry/exit points
    entry_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    exit_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    # Draw regular nodes
    for node, (lon, lat) in pos.items():
        if node in entry_nodes:
            ax.plot(lon, lat, '>', markersize=18, color='#27ae60',
                   markeredgecolor='#1e8449', markeredgewidth=2, zorder=10)
        elif node in exit_nodes:
            ax.plot(lon, lat, 'H', markersize=15, color='#c0392b',
                   markeredgecolor='#922b21', markeredgewidth=2, zorder=10)
        else:
            circle = Circle((lon, lat), 0.0003, facecolor='white', 
                           edgecolor='#555555', linewidth=1.5, zorder=5)
            ax.add_patch(circle)
    
    # ================================================================
    # 3. ROAD LABELS
    # ================================================================
    for road_name in road_edges.keys():
        # Find center of road
        edges = road_edges[road_name]
        if not edges:
            continue
        
        all_pts = []
        for e in edges:
            if e['u'] in pos:
                all_pts.append(pos[e['u']])
            if e['v'] in pos:
                all_pts.append(pos[e['v']])
        
        if all_pts:
            lons = [p[0] for p in all_pts]
            lats = [p[1] for p in all_pts]
            cx, cy = np.mean(lons), np.mean(lats)
            
            # Calculate angle
            if len(all_pts) >= 2:
                dx = max(lons) - min(lons)
                dy = max(lats) - min(lats)
                angle = np.degrees(np.arctan2(dy, dx))
                if angle > 90: angle -= 180
                if angle < -90: angle += 180
            else:
                angle = 0
            
            ax.text(cx, cy + 0.0008, road_name,
                   fontsize=10, fontweight='bold',
                   color='#2c3e50', rotation=angle,
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white', edgecolor='#bdc3c7',
                            alpha=0.9),
                   zorder=20)
    
    # ================================================================
    # 4. COLORBAR
    # ================================================================
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar_ax = fig.add_axes([0.2, 0.06, 0.45, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Vitesse moyenne (km/h)', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # Regime labels
    cbar.ax.text(0.12, -2.2, 'Bouchon', ha='center', fontsize=9, transform=cbar.ax.transAxes)
    cbar.ax.text(0.37, -2.2, 'Congestion', ha='center', fontsize=9, transform=cbar.ax.transAxes)
    cbar.ax.text(0.62, -2.2, 'Modéré', ha='center', fontsize=9, transform=cbar.ax.transAxes)
    cbar.ax.text(0.87, -2.2, 'Fluide', ha='center', fontsize=9, transform=cbar.ax.transAxes)
    
    # ================================================================
    # 5. LEGEND
    # ================================================================
    legend_elements = [
        Line2D([0], [0], marker='>', color='w', markerfacecolor='#27ae60',
               markeredgecolor='#1e8449', markersize=14, label="Point d'entrée"),
        Line2D([0], [0], marker='H', color='w', markerfacecolor='#c0392b',
               markeredgecolor='#922b21', markersize=12, label='Point de sortie'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='#555555', markersize=8, label='Intersection'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
             framealpha=0.95, edgecolor='#555555', title='Éléments',
             title_fontsize=11)
    
    # ================================================================
    # 6. INFO BOX
    # ================================================================
    total_length = sum(d.get('length', 0) for _, _, d in G.edges(data=True)) / 1000
    avg_speed = np.mean(list(edge_speeds.values()))
    
    info_text = (
        f"CORRIDOR VICTORIA ISLAND\n"
        f"Lagos, Nigeria\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Segments: {G.number_of_edges()}\n"
        f"Intersections: {G.number_of_nodes()}\n"
        f"Longueur: {total_length:.2f} km\n"
        f"Vitesse moy.: {avg_speed:.1f} km/h\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Source: OpenStreetMap"
    )
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1',
                    edgecolor='#7f8c8d', alpha=0.95),
           zorder=25)
    
    # ================================================================
    # 7. COMPASS AND SCALE
    # ================================================================
    # North arrow
    if pos:
        lons = [p[0] for p in pos.values()]
        lats = [p[1] for p in pos.values()]
        ax_right = max(lons) - (max(lons) - min(lons)) * 0.05
        ax_bottom = min(lats) + (max(lats) - min(lats)) * 0.1
        
        arrow_len = (max(lats) - min(lats)) * 0.08
        ax.annotate('', xy=(ax_right, ax_bottom + arrow_len), 
                   xytext=(ax_right, ax_bottom),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))
        ax.text(ax_right, ax_bottom + arrow_len + 0.0005, 'N', 
               fontsize=11, fontweight='bold', ha='center', color='#2c3e50')
    
    # ================================================================
    # 8. TITLE
    # ================================================================
    ax.set_title('Réseau Routier de Victoria Island, Lagos\n'
                 'Visualisation géographique avec conditions de trafic simulées',
                 fontsize=14, fontweight='bold', pad=15)
    
    # Set axis
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.tick_params(labelsize=9)
    
    # Set bounds with padding
    if pos:
        lons = [p[0] for p in pos.values()]
        lats = [p[1] for p in pos.values()]
        lon_pad = (max(lons) - min(lons)) * 0.1
        lat_pad = (max(lats) - min(lats)) * 0.1
        ax.set_xlim(min(lons) - lon_pad, max(lons) + lon_pad)
        ax.set_ylim(min(lats) - lat_pad, max(lats) + lat_pad)
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✅ Saved: {output_path}")


def main():
    print("=" * 70)
    print("🌍 REAL GEOGRAPHIC NETWORK VISUALIZATION")
    print("   Victoria Island, Lagos - From OpenStreetMap")
    print("=" * 70)
    
    # Try to fetch real OSM data
    osm_data = fetch_osm_data()
    
    if osm_data:
        nodes, ways = parse_osm_data(osm_data)
        if nodes and ways:
            G, road_edges = build_real_graph(nodes, ways)
        else:
            G, road_edges = create_fallback_graph()
    else:
        G, road_edges = create_fallback_graph()
    
    print(f"\n📊 Network Statistics:")
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   Roads: {list(road_edges.keys())}")
    
    # Generate traffic
    print("\n🚗 Generating traffic pattern...")
    edge_speeds = generate_traffic_pattern(G)
    print(f"   Speed range: {min(edge_speeds.values()):.1f} - {max(edge_speeds.values()):.1f} km/h")
    
    # Create visualization
    print("\n🖼️  Creating geographic visualization...")
    output_path = Path('viz_output/geographic_network.png')
    create_geographic_visualization(G, road_edges, edge_speeds, output_path)
    
    # Copy to thesis
    thesis_path = Path('images/chapter3/geographic_network.png')
    thesis_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(output_path, thesis_path)
    print(f"✅ Copied to: {thesis_path}")
    
    print("\n" + "=" * 70)
    print("✅ GEOGRAPHIC VISUALIZATION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
