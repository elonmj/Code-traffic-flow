#!/usr/bin/env python3
"""
🎯 FINAL THESIS NETWORK VISUALIZATION
======================================

The DEFINITIVE visualization for the thesis combining:
1. REAL geographic coordinates from OpenStreetMap
2. Infrastructure elements (traffic lights, entry/exit points)
3. Professional publication-quality rendering

This creates a geographically accurate representation of the
Victoria Island corridor in Lagos, Nigeria.
"""

import sys
import os
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox
import networkx as nx
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Victoria Island bounding box
BBOX = {'south': 6.42, 'west': 3.40, 'north': 6.46, 'east': 3.46}


def fetch_osm_roads():
    """Fetch Victoria Island road network from OSM."""
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
    
    print("📡 Fetching road data from OpenStreetMap...")
    try:
        response = requests.post("https://overpass-api.de/api/interpreter", 
                                data={'data': query}, timeout=120)
        response.raise_for_status()
        data = response.json()
        print(f"   ✅ Received {len(data.get('elements', []))} elements")
        return data
    except:
        print("   ⚠️ OSM fetch failed, using fallback")
        return None


def fetch_traffic_signals():
    """Fetch traffic signals in the area."""
    query = f"""
    [out:json][timeout:30];
    (
      node["highway"="traffic_signals"]
        ({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});
    );
    out body;
    """
    
    print("🚦 Fetching traffic signals...")
    try:
        response = requests.post("https://overpass-api.de/api/interpreter",
                                data={'data': query}, timeout=60)
        response.raise_for_status()
        data = response.json()
        signals = [(e['lon'], e['lat']) for e in data.get('elements', []) 
                   if e['type'] == 'node']
        print(f"   ✅ Found {len(signals)} traffic signals")
        return signals
    except:
        print("   ⚠️ Traffic signals fetch failed")
        return []


def build_network(osm_data):
    """Build NetworkX graph from OSM data."""
    if not osm_data:
        return create_fallback()
    
    nodes = {}
    ways = []
    
    for elem in osm_data.get('elements', []):
        if elem['type'] == 'node':
            nodes[elem['id']] = (elem['lon'], elem['lat'])
        elif elem['type'] == 'way':
            ways.append({
                'id': elem['id'],
                'name': elem.get('tags', {}).get('name', 'Unknown'),
                'nodes': elem.get('nodes', [])
            })
    
    G = nx.DiGraph()
    road_edges = defaultdict(list)
    
    for way in ways:
        name = way['name']
        for i in range(len(way['nodes']) - 1):
            u, v = way['nodes'][i], way['nodes'][i+1]
            if u in nodes and v in nodes:
                lon1, lat1 = nodes[u]
                lon2, lat2 = nodes[v]
                length = haversine(lat1, lon1, lat2, lon2)
                
                G.add_node(str(u), pos=(lon1, lat1))
                G.add_node(str(v), pos=(lon2, lat2))
                G.add_edge(str(u), str(v), name=name, length=length)
                road_edges[name].append({'u': str(u), 'v': str(v)})
    
    return G, road_edges


def create_fallback():
    """Fallback if OSM fails."""
    roads = {
        'Ahmadu Bello Way': [(3.415, 6.432), (3.42, 6.433), (3.43, 6.434), (3.44, 6.435), (3.45, 6.436)],
        'Akin Adesola Street': [(3.42, 6.428), (3.421, 6.434), (3.422, 6.44), (3.423, 6.446)],
        'Adeola Odeku Street': [(3.418, 6.44), (3.428, 6.439), (3.438, 6.438)],
        'Saka Tinubu Street': [(3.432, 6.431), (3.433, 6.437), (3.434, 6.443)]
    }
    G = nx.DiGraph()
    road_edges = defaultdict(list)
    for name, coords in roads.items():
        for i in range(len(coords)-1):
            u, v = f"{name[:3]}_{i}", f"{name[:3]}_{i+1}"
            G.add_node(u, pos=coords[i])
            G.add_node(v, pos=coords[i+1])
            G.add_edge(u, v, name=name)
            road_edges[name].append({'u': u, 'v': v})
    return G, road_edges


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in meters."""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def find_signal_nodes(G, signals, threshold=0.0005):
    """Find network nodes near traffic signals."""
    pos = nx.get_node_attributes(G, 'pos')
    signal_nodes = set()
    
    for sig_lon, sig_lat in signals:
        for node, (lon, lat) in pos.items():
            dist = np.sqrt((lon - sig_lon)**2 + (lat - sig_lat)**2)
            if dist < threshold:
                signal_nodes.add(node)
                break
    
    return list(signal_nodes)


def generate_traffic(G):
    """Generate realistic rush hour traffic."""
    pos = nx.get_node_attributes(G, 'pos')
    lons = [p[0] for p in pos.values()]
    lats = [p[1] for p in pos.values()]
    center = (np.mean(lons), np.mean(lats))
    
    speeds = {}
    for u, v, data in G.edges(data=True):
        if u in pos:
            lon, lat = pos[u]
            dist = np.sqrt((lon - center[0])**2 + (lat - center[1])**2)
            
            name = data.get('name', '')
            if 'Ahmadu Bello' in name:
                base = 25 + np.random.normal(0, 8)  # Main artery - congested
            elif 'Akin Adesola' in name:
                base = 35 + np.random.normal(0, 10)
            elif 'Adeola Odeku' in name:
                base = 45 + np.random.normal(0, 10)
            else:
                base = 40 + np.random.normal(0, 12)
            
            # Variation by position
            if dist < 0.004:
                base *= 0.7  # Center more congested
            elif dist > 0.01:
                base *= 1.2  # Periphery flows better
            
            speeds[(u, v)] = np.clip(base, 5, 80)
        else:
            speeds[(u, v)] = 40
    
    return speeds


def create_colormap():
    """Traffic speed colormap."""
    colors = ['#8B0000', '#CC0000', '#FF4444', '#FF8800', '#FFCC00',
              '#FFFF00', '#AAFF00', '#66FF33', '#00DD00', '#00AA44']
    cmap = LinearSegmentedColormap.from_list('traffic', colors, N=256)
    norm = mcolors.Normalize(vmin=0, vmax=80)
    return cmap, norm


def create_thesis_figure(G, road_edges, speeds, signals, signal_nodes, output_path):
    """Create publication-quality thesis figure."""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('#f5f9fc')
    
    pos = nx.get_node_attributes(G, 'pos')
    cmap, norm = create_colormap()
    
    entry_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    exit_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    # ================================================================
    # 1. DRAW EDGES (traffic)
    # ================================================================
    for (u, v), speed in speeds.items():
        if u in pos and v in pos:
            lon1, lat1 = pos[u]
            lon2, lat2 = pos[v]
            color = cmap(norm(speed))
            
            ax.annotate('', xy=(lon2, lat2), xytext=(lon1, lat1),
                       arrowprops=dict(arrowstyle='-|>', color=color,
                                      lw=3.5, mutation_scale=12,
                                      shrinkA=6, shrinkB=6), zorder=3)
    
    # ================================================================
    # 2. REGULAR INTERSECTIONS
    # ================================================================
    regular = [n for n in G.nodes() 
               if n not in entry_nodes and n not in exit_nodes and n not in signal_nodes]
    
    for node in regular:
        if node in pos:
            lon, lat = pos[node]
            circle = Circle((lon, lat), 0.00025, facecolor='white',
                           edgecolor='#555555', linewidth=1.2, zorder=5)
            ax.add_patch(circle)
    
    # ================================================================
    # 3. TRAFFIC SIGNALS
    # ================================================================
    for node in signal_nodes:
        if node in pos:
            lon, lat = pos[node]
            # Traffic light box
            rect = FancyBboxPatch((lon-0.0003, lat-0.0004), 0.0006, 0.0008,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#2c3e50', edgecolor='#1a252f',
                                  linewidth=1.5, zorder=10)
            ax.add_patch(rect)
            # Lights
            ax.plot(lon, lat+0.0002, 'o', markersize=5, color='#e74c3c', zorder=11)
            ax.plot(lon, lat, 'o', markersize=5, color='#f39c12', zorder=11)
            ax.plot(lon, lat-0.0002, 'o', markersize=5, color='#2ecc71', zorder=11)
    
    # Also mark OSM signals not near nodes
    for sig_lon, sig_lat in signals:
        # Check if already marked
        is_near_node = any(
            np.sqrt((sig_lon - pos.get(n, (0,0))[0])**2 + (sig_lat - pos.get(n, (0,0))[1])**2) < 0.0005
            for n in signal_nodes
        )
        if not is_near_node:
            rect = FancyBboxPatch((sig_lon-0.0003, sig_lat-0.0004), 0.0006, 0.0008,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#34495e', edgecolor='#2c3e50',
                                  linewidth=1, alpha=0.7, zorder=8)
            ax.add_patch(rect)
    
    # ================================================================
    # 4. ENTRY POINTS
    # ================================================================
    for node in entry_nodes[:8]:  # Limit to first 8
        if node in pos:
            lon, lat = pos[node]
            ax.plot(lon, lat, '>', markersize=16, color='#27ae60',
                   markeredgecolor='#1e8449', markeredgewidth=2, zorder=10)
    
    # ================================================================
    # 5. EXIT POINTS
    # ================================================================
    for node in exit_nodes[:8]:  # Limit to first 8
        if node in pos:
            lon, lat = pos[node]
            ax.plot(lon, lat, 'H', markersize=14, color='#c0392b',
                   markeredgecolor='#922b21', markeredgewidth=2, zorder=10)
    
    # ================================================================
    # 6. ROAD LABELS
    # ================================================================
    labeled = set()
    for road_name in road_edges.keys():
        if road_name in labeled:
            continue
        
        edges = road_edges[road_name]
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
                sorted_pts = sorted(all_pts, key=lambda p: p[0])
                dx = sorted_pts[-1][0] - sorted_pts[0][0]
                dy = sorted_pts[-1][1] - sorted_pts[0][1]
                angle = np.degrees(np.arctan2(dy, dx))
                if angle > 90: angle -= 180
                if angle < -90: angle += 180
            else:
                angle = 0
            
            ax.text(cx, cy + 0.001, road_name,
                   fontsize=11, fontweight='bold', fontstyle='italic',
                   color='#1a1a1a', rotation=angle,
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.25',
                            facecolor='white', edgecolor='#aaaaaa',
                            alpha=0.95),
                   zorder=25)
            labeled.add(road_name)
    
    # ================================================================
    # 7. COLORBAR
    # ================================================================
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar_ax = fig.add_axes([0.18, 0.06, 0.5, 0.022])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Vitesse moyenne (km/h)', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # Regime labels with colored backgrounds
    for pos_x, label, color in [(0.1, 'Bouchon', '#8B0000'), (0.35, 'Congestion', '#FF8800'),
                                 (0.65, 'Modéré', '#CCFF00'), (0.9, 'Fluide', '#00AA44')]:
        cbar.ax.text(pos_x, -2.0, label, ha='center', fontsize=9,
                    fontweight='bold', transform=cbar.ax.transAxes)
    
    # ================================================================
    # 8. LEGEND
    # ================================================================
    legend_elements = [
        Line2D([0], [0], marker='>', color='w', markerfacecolor='#27ae60',
               markeredgecolor='#1e8449', markersize=14, label=f"Point d'entrée ({len(entry_nodes)})"),
        Line2D([0], [0], marker='H', color='w', markerfacecolor='#c0392b',
               markeredgecolor='#922b21', markersize=12, label=f'Point de sortie ({len(exit_nodes)})'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2c3e50',
               markeredgecolor='#1a252f', markersize=12, label=f'Feu de signalisation ({len(signals)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='#555555', markersize=8, label='Intersection'),
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                      framealpha=0.95, edgecolor='#555555', fancybox=True,
                      title='Éléments du réseau', title_fontsize=11)
    legend.get_frame().set_linewidth(1.5)
    
    # ================================================================
    # 9. STATISTICS BOX
    # ================================================================
    total_length = sum(speeds.values()) / len(speeds) if speeds else 0  # avg for display
    
    lons = [p[0] for p in pos.values()]
    lats = [p[1] for p in pos.values()]
    
    stats_text = (
        f"CORRIDOR VICTORIA ISLAND\n"
        f"Lagos, Nigeria\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Segments routiers: {G.number_of_edges()}\n"
        f"Intersections: {G.number_of_nodes()}\n"
        f"Feux de signalisation: {len(signals)}\n"
        f"Points d'entrée: {len(entry_nodes)}\n"
        f"Points de sortie: {len(exit_nodes)}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Vitesse moyenne: {np.mean(list(speeds.values())):.1f} km/h\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Source: OpenStreetMap"
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1',
                    edgecolor='#7f8c8d', linewidth=1.5, alpha=0.95),
           zorder=30)
    
    # ================================================================
    # 10. COMPASS ROSE
    # ================================================================
    compass_lon = max(lons) - (max(lons) - min(lons)) * 0.08
    compass_lat = min(lats) + (max(lats) - min(lats)) * 0.12
    arrow_len = (max(lats) - min(lats)) * 0.06
    
    ax.annotate('', xy=(compass_lon, compass_lat + arrow_len),
               xytext=(compass_lon, compass_lat),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#2c3e50'))
    ax.text(compass_lon, compass_lat + arrow_len + 0.0008, 'N',
           fontsize=13, fontweight='bold', ha='center', color='#2c3e50')
    
    # ================================================================
    # 11. TITLE
    # ================================================================
    ax.set_title('Réseau Routier du Corridor Victoria Island, Lagos\n'
                 'Simulation de trafic heure de pointe avec infrastructure de contrôle',
                 fontsize=14, fontweight='bold', pad=20)
    
    # ================================================================
    # 12. AXES
    # ================================================================
    ax.set_xlabel('Longitude (°E)', fontsize=11)
    ax.set_ylabel('Latitude (°N)', fontsize=11)
    ax.tick_params(labelsize=9)
    
    lon_pad = (max(lons) - min(lons)) * 0.1
    lat_pad = (max(lats) - min(lats)) * 0.15
    ax.set_xlim(min(lons) - lon_pad, max(lons) + lon_pad)
    ax.set_ylim(min(lats) - lat_pad, max(lats) + lat_pad)
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3, color='#888888')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✅ Saved: {output_path}")


def main():
    print("=" * 70)
    print("🎯 FINAL THESIS NETWORK VISUALIZATION")
    print("   Publication-quality figure with real OSM data")
    print("=" * 70)
    
    # Fetch data
    osm_data = fetch_osm_roads()
    signals = fetch_traffic_signals()
    
    # Build network
    print("\n📊 Building network...")
    G, road_edges = build_network(osm_data)
    print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"   Roads: {list(road_edges.keys())}")
    
    # Find signal nodes
    signal_nodes = find_signal_nodes(G, signals)
    print(f"   Signals near nodes: {len(signal_nodes)}")
    
    # Generate traffic
    print("\n🚗 Generating traffic pattern...")
    speeds = generate_traffic(G)
    print(f"   Speed range: {min(speeds.values()):.1f} - {max(speeds.values()):.1f} km/h")
    
    # Create visualization
    print("\n🖼️  Creating thesis figure...")
    output_path = Path('viz_output/thesis_network_final.png')
    create_thesis_figure(G, road_edges, speeds, signals, signal_nodes, output_path)
    
    # Copy to thesis locations
    for dest in ['images/chapter3/thesis_network_final.png',
                 'results/thesis_figures/thesis_network_final.png']:
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(output_path, dest_path)
        print(f"✅ Copied to: {dest_path}")
    
    print("\n" + "=" * 70)
    print("✅ THESIS VISUALIZATION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
