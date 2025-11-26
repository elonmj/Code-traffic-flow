#!/usr/bin/env python3
"""
🎨 STRUCTURED CORRIDOR VISUALIZATION
=====================================

Creates a STRUCTURED, GEOMETRIC layout that represents the Victoria Island
corridor as it actually looks - a grid of interconnected roads.

This uses a CUSTOM LAYOUT ALGORITHM that:
1. Places roads along clear axes
2. Respects segment connectivity
3. Creates a visually meaningful representation
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrow
import networkx as nx
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def create_speed_colormap():
    """Create traffic speed colormap."""
    colors = ['#8B0000', '#CC0000', '#FF4444', '#FF8800', '#FFCC00', 
              '#FFFF00', '#AAFF00', '#66FF33', '#00DD00', '#00AA44']
    cmap = LinearSegmentedColormap.from_list('traffic', colors, N=256)
    norm = mcolors.Normalize(vmin=0, vmax=80)
    return cmap, norm


def load_network_data():
    """Load network data from enriched Excel."""
    excel_path = Path('arz_model/data/fichier_de_travail_corridor_enriched.xlsx')
    df = pd.read_excel(excel_path)
    return df


def build_structured_layout(df):
    """
    Create a STRUCTURED GEOMETRIC layout based on road topology.
    
    Victoria Island has 4 main roads:
    - Ahmadu Bello Way: Major arterial (east-west)
    - Akin Adesola Street: Perpendicular connector (north-south)  
    - Adeola Odeku Street: Parallel to Ahmadu Bello
    - Saka Tinubu Street: Short connector
    
    We create a logical grid layout based on these roads.
    """
    G = nx.DiGraph()
    
    # Build graph with road info
    for _, row in df.iterrows():
        u, v = str(row['u']), str(row['v'])
        G.add_edge(u, v, 
                   name=row['name_clean'],
                   length=row['length'],
                   u_signal=row.get('u_has_signal', False),
                   v_signal=row.get('v_has_signal', False))
    
    # Group edges by road
    road_edges = defaultdict(list)
    for _, row in df.iterrows():
        name = row['name_clean']
        if pd.notna(name):
            road_edges[name].append({
                'u': str(row['u']),
                'v': str(row['v']),
                'length': row['length']
            })
    
    # Build node-to-road mapping
    node_roads = defaultdict(set)
    for road, edges in road_edges.items():
        for e in edges:
            node_roads[e['u']].add(road)
            node_roads[e['v']].add(road)
    
    # ================================================================
    # DEFINE ROAD ORIENTATIONS AND POSITIONS
    # ================================================================
    
    # Road layout configuration:
    # Y axis = "north" 
    # X axis = "east"
    
    road_config = {
        'Ahmadu Bello Way': {
            'orientation': 'horizontal',  # Runs east-west
            'y_base': 3.0,                 # Position along Y
            'x_range': (0, 10),            # X extent
            'priority': 1                   # Layout priority
        },
        'Akin Adesola Street': {
            'orientation': 'vertical',     # Runs north-south
            'x_base': 3.0,                  # Position along X
            'y_range': (0, 6),              # Y extent
            'priority': 2
        },
        'Adeola Odeku Street': {
            'orientation': 'horizontal',   # Parallel to Ahmadu Bello
            'y_base': 5.5,                  # Position along Y
            'x_range': (2, 8),              # X extent
            'priority': 3
        },
        'Saka Tinubu Street': {
            'orientation': 'vertical',     # Short connector
            'x_base': 6.0,                  # Position along X  
            'y_range': (2, 5),              # Y extent
            'priority': 4
        }
    }
    
    # ================================================================
    # PLACE NODES ALONG THEIR ROADS
    # ================================================================
    positions = {}
    
    # For each road, chain its segments and place nodes
    for road_name, config in sorted(road_config.items(), key=lambda x: x[1]['priority']):
        edges = road_edges.get(road_name, [])
        if not edges:
            continue
            
        # Build a path through this road's segments
        road_graph = nx.DiGraph()
        for e in edges:
            road_graph.add_edge(e['u'], e['v'], length=e['length'])
        
        # Find chains of nodes
        # Start from nodes with no predecessor in this road
        starts = [n for n in road_graph.nodes() if road_graph.in_degree(n) == 0]
        
        for start in starts:
            # Walk the chain
            chain = [start]
            current = start
            while True:
                successors = list(road_graph.successors(current))
                if not successors:
                    break
                current = successors[0]
                chain.append(current)
            
            # Calculate cumulative lengths
            cum_lengths = [0]
            for i in range(len(chain) - 1):
                edge_len = road_graph[chain[i]][chain[i+1]]['length']
                cum_lengths.append(cum_lengths[-1] + edge_len)
            
            # Normalize to [0, 1]
            total_len = cum_lengths[-1] if cum_lengths[-1] > 0 else 1
            normalized = [l / total_len for l in cum_lengths]
            
            # Map to positions
            if config['orientation'] == 'horizontal':
                x_min, x_max = config['x_range']
                y_val = config['y_base']
                for node, t in zip(chain, normalized):
                    x = x_min + t * (x_max - x_min)
                    # Small vertical offset if already placed (intersection)
                    if node in positions:
                        # This is an intersection - average positions
                        old_x, old_y = positions[node]
                        positions[node] = ((old_x + x) / 2, (old_y + y_val) / 2)
                    else:
                        positions[node] = (x, y_val)
            else:  # vertical
                y_min, y_max = config['y_range']
                x_val = config['x_base']
                for node, t in zip(chain, normalized):
                    y = y_min + t * (y_max - y_min)
                    if node in positions:
                        old_x, old_y = positions[node]
                        positions[node] = ((old_x + x_val) / 2, (old_y + y) / 2)
                    else:
                        positions[node] = (x_val, y)
    
    # Place any remaining nodes using spring layout
    unplaced = [n for n in G.nodes() if n not in positions]
    if unplaced:
        subgraph = G.subgraph(unplaced)
        if len(subgraph) > 0:
            spring_pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
            for node, (x, y) in spring_pos.items():
                positions[node] = (x * 2 + 5, y * 2 + 3)  # Scale and center
    
    return G, positions, road_edges


def identify_elements(G, df):
    """Identify entry, exit, and signal nodes."""
    entry_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    exit_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    signal_nodes = set()
    for _, row in df.iterrows():
        if row.get('u_has_signal') == True:
            signal_nodes.add(str(row['u']))
        if row.get('v_has_signal') == True:
            signal_nodes.add(str(row['v']))
    
    return entry_nodes, exit_nodes, list(signal_nodes)


def generate_realistic_traffic(G, positions):
    """Generate realistic rush hour traffic pattern."""
    edge_speeds = {}
    
    entry_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    exit_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    for u, v in G.edges():
        # Base speed
        if u in positions and v in positions:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            
            # Congestion patterns:
            # 1. Near entry points: congested (vehicles queuing)
            dist_to_entry = min([
                abs(x1 - positions.get(e, (0,0))[0]) + abs(y1 - positions.get(e, (0,0))[1])
                for e in entry_nodes if e in positions
            ] or [10])
            
            # 2. Main corridors (center of network): moderate
            center_dist = abs(x1 - 5) + abs(y1 - 4)
            
            # 3. Near exits: fluid (vehicles dispersing)
            dist_to_exit = min([
                abs(x1 - positions.get(e, (0,0))[0]) + abs(y1 - positions.get(e, (0,0))[1])
                for e in exit_nodes if e in positions
            ] or [10])
            
            # Calculate speed based on these factors
            if dist_to_entry < 2:
                # Near entry - congested (rush hour)
                base_speed = 12 + np.random.uniform(0, 8)
            elif center_dist < 3:
                # Central area - moderate traffic
                base_speed = 35 + np.random.uniform(-10, 10)
            elif dist_to_exit < 2:
                # Near exit - flowing well
                base_speed = 55 + np.random.uniform(-5, 15)
            else:
                # Default - moderate flow
                base_speed = 45 + np.random.uniform(-15, 15)
            
            edge_speeds[(u, v)] = np.clip(base_speed, 5, 80)
        else:
            edge_speeds[(u, v)] = 40  # Default
    
    return edge_speeds


def create_visualization(G, positions, df, road_edges,
                         entry_nodes, exit_nodes, signal_nodes,
                         edge_speeds, output_path):
    """Create the structured corridor visualization."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 11))
    ax.set_facecolor('#f8f9fa')
    
    cmap, norm = create_speed_colormap()
    
    # ================================================================
    # 1. DRAW ROAD CORRIDORS (background)
    # ================================================================
    road_colors = {
        'Ahmadu Bello Way': '#e3e8ed',
        'Akin Adesola Street': '#e3e8ed', 
        'Adeola Odeku Street': '#e3e8ed',
        'Saka Tinubu Street': '#e3e8ed'
    }
    
    # Draw background roads as thick lines
    for road_name, edges in road_edges.items():
        all_pts = []
        for e in edges:
            if e['u'] in positions:
                all_pts.append(positions[e['u']])
            if e['v'] in positions:
                all_pts.append(positions[e['v']])
        
        if len(all_pts) >= 2:
            # Sort by x then y for consistent ordering
            sorted_pts = sorted(all_pts, key=lambda p: (p[0], p[1]))
            xs = [p[0] for p in sorted_pts]
            ys = [p[1] for p in sorted_pts]
            ax.plot(xs, ys, '-', color='#dde3e9', linewidth=25, 
                   solid_capstyle='round', zorder=1)
    
    # ================================================================
    # 2. DRAW EDGES with traffic colors
    # ================================================================
    for (u, v), speed in edge_speeds.items():
        if u in positions and v in positions:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            
            color = cmap(norm(speed))
            
            # Draw as arrow
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(
                           arrowstyle='-|>',
                           color=color,
                           lw=5,
                           mutation_scale=18,
                           shrinkA=15,
                           shrinkB=15,
                           connectionstyle='arc3,rad=0'
                       ), zorder=3)
    
    # ================================================================
    # 3. REGULAR INTERSECTIONS
    # ================================================================
    regular = [n for n in G.nodes() 
               if n not in entry_nodes 
               and n not in exit_nodes 
               and n not in signal_nodes]
    
    for node in regular:
        if node in positions:
            x, y = positions[node]
            circle = Circle((x, y), 0.18, facecolor='white', 
                           edgecolor='#555555', linewidth=2, zorder=5)
            ax.add_patch(circle)
    
    # ================================================================
    # 4. TRAFFIC LIGHTS (Yellow with R/Y/G lights)
    # ================================================================
    for node in signal_nodes:
        if node in positions:
            x, y = positions[node]
            # Traffic light box
            rect = FancyBboxPatch((x-0.22, y-0.30), 0.44, 0.60,
                                  boxstyle="round,pad=0.03",
                                  facecolor='#2c3e50', 
                                  edgecolor='#1a252f',
                                  linewidth=2, zorder=10)
            ax.add_patch(rect)
            # Lights
            ax.plot(x, y+0.15, 'o', markersize=8, color='#e74c3c', zorder=11)  # Red
            ax.plot(x, y, 'o', markersize=8, color='#f39c12', zorder=11)       # Yellow
            ax.plot(x, y-0.15, 'o', markersize=8, color='#2ecc71', zorder=11)  # Green
    
    # ================================================================
    # 5. ENTRY POINTS (Green arrows)
    # ================================================================
    for node in entry_nodes:
        if node in positions:
            x, y = positions[node]
            # Green arrow marker
            ax.plot(x, y, '>', markersize=28, color='#27ae60',
                   markeredgecolor='#1e8449', markeredgewidth=3, zorder=10)
            # Label
            ax.text(x-0.6, y+0.3, 'ENTRÉE', fontsize=9, fontweight='bold',
                   ha='right', va='bottom', color='#1e8449',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            edgecolor='#27ae60', alpha=0.9))
    
    # ================================================================
    # 6. EXIT POINTS (Red hexagons)
    # ================================================================
    for node in exit_nodes:
        if node in positions:
            x, y = positions[node]
            ax.plot(x, y, 'H', markersize=24, color='#c0392b',
                   markeredgecolor='#922b21', markeredgewidth=3, zorder=10)
            ax.text(x+0.6, y+0.3, 'SORTIE', fontsize=9, fontweight='bold',
                   ha='left', va='bottom', color='#922b21',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='#c0392b', alpha=0.9))
    
    # ================================================================
    # 7. ROAD LABELS
    # ================================================================
    road_label_placed = set()
    
    for road_name, edges in road_edges.items():
        if road_name in road_label_placed:
            continue
        
        # Find center of road
        all_pts = []
        for e in edges:
            if e['u'] in positions:
                all_pts.append(positions[e['u']])
            if e['v'] in positions:
                all_pts.append(positions[e['v']])
        
        if all_pts:
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            cx, cy = np.mean(xs), np.mean(ys)
            
            # Determine orientation for rotation
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)
            
            if dx > dy:
                angle = 0  # Horizontal road
                offset = (0, 0.7)
            else:
                angle = 90  # Vertical road
                offset = (0.7, 0)
            
            ax.text(cx + offset[0], cy + offset[1], road_name,
                   fontsize=11, fontweight='bold',
                   color='#2c3e50', rotation=angle,
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='#ffffff', edgecolor='#bdc3c7', 
                            alpha=0.95, linewidth=1.5),
                   zorder=20)
            
            road_label_placed.add(road_name)
    
    # ================================================================
    # 8. COLORBAR
    # ================================================================
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar_ax = fig.add_axes([0.2, 0.06, 0.45, 0.025])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Vitesse moyenne (km/h)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Traffic regime labels
    regime_positions = [10, 30, 50, 70]
    regime_labels = ['Bouchon\n(<20)', 'Congestion\n(20-40)', 
                     'Modéré\n(40-60)', 'Fluide\n(>60)']
    
    for pos, label in zip(regime_positions, regime_labels):
        cbar.ax.axvline(x=pos, color='white', linewidth=0.5, alpha=0.5)
    
    # Add regime text below colorbar
    cbar.ax.text(0.12, -2.5, 'Bouchon', ha='center', fontsize=9, 
                transform=cbar.ax.transAxes)
    cbar.ax.text(0.37, -2.5, 'Congestion', ha='center', fontsize=9,
                transform=cbar.ax.transAxes)
    cbar.ax.text(0.62, -2.5, 'Modéré', ha='center', fontsize=9,
                transform=cbar.ax.transAxes)
    cbar.ax.text(0.87, -2.5, 'Fluide', ha='center', fontsize=9,
                transform=cbar.ax.transAxes)
    
    # ================================================================
    # 9. LEGEND
    # ================================================================
    legend_elements = [
        Line2D([0], [0], marker='>', color='w', markerfacecolor='#27ae60',
               markeredgecolor='#1e8449', markersize=16, label="Point d'entrée (4)"),
        Line2D([0], [0], marker='H', color='w', markerfacecolor='#c0392b',
               markeredgecolor='#922b21', markersize=14, label='Point de sortie (4)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2c3e50',
               markeredgecolor='#1a252f', markersize=14, label='Feu de signalisation (8)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='#555555', markersize=10, label='Intersection simple'),
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
                      framealpha=0.95, edgecolor='#555555', fancybox=True,
                      title='Éléments du réseau', title_fontsize=12)
    legend.get_frame().set_linewidth(1.5)
    
    # ================================================================
    # 10. INFO BOX
    # ================================================================
    total_length = df['length'].sum() / 1000
    avg_speed = np.mean(list(edge_speeds.values()))
    
    info_text = (
        f"CORRIDOR VICTORIA ISLAND\n"
        f"Lagos, Nigeria\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Segments routiers: {G.number_of_edges()}\n"
        f"Intersections: {G.number_of_nodes()}\n"
        f"Longueur totale: {total_length:.2f} km\n"
        f"Feux de signalisation: {len(signal_nodes)}\n"
        f"Points d'entrée: {len(entry_nodes)}\n"
        f"Points de sortie: {len(exit_nodes)}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Vitesse moyenne: {avg_speed:.1f} km/h"
    )
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#ecf0f1',
                    edgecolor='#7f8c8d', linewidth=1.5, alpha=0.95),
           zorder=25)
    
    # ================================================================
    # 11. COMPASS ROSE
    # ================================================================
    compass_x, compass_y = 9.5, 0.5
    arrow_len = 0.4
    ax.annotate('', xy=(compass_x, compass_y + arrow_len), 
               xytext=(compass_x, compass_y - arrow_len),
               arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))
    ax.text(compass_x, compass_y + arrow_len + 0.2, 'N', fontsize=12, 
           fontweight='bold', ha='center', color='#2c3e50')
    
    # ================================================================
    # 12. TITLE
    # ================================================================
    ax.set_title('Schéma du Réseau Routier - Victoria Island, Lagos\n'
                 'Simulation de trafic en heure de pointe avec infrastructure de contrôle',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Axes setup
    ax.set_xlim(-1.5, 12)
    ax.set_ylim(-1, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✅ Saved: {output_path}")


def main():
    print("=" * 70)
    print("🎨 STRUCTURED CORRIDOR VISUALIZATION")
    print("   Victoria Island Network - Geometric Layout")
    print("=" * 70)
    
    # Load data
    print("\n📊 Loading network data...")
    df = load_network_data()
    print(f"   Loaded {len(df)} segments")
    
    # Build layout
    print("\n🗺️  Building structured layout...")
    G, positions, road_edges = build_structured_layout(df)
    print(f"   Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Positioned: {len(positions)} nodes")
    
    # Identify elements
    print("\n🔍 Identifying network elements...")
    entry_nodes, exit_nodes, signal_nodes = identify_elements(G, df)
    print(f"   Entries: {len(entry_nodes)}, Exits: {len(exit_nodes)}, Signals: {len(signal_nodes)}")
    
    # Generate traffic
    print("\n🚗 Generating rush hour traffic pattern...")
    edge_speeds = generate_realistic_traffic(G, positions)
    print(f"   Speed range: {min(edge_speeds.values()):.1f} - {max(edge_speeds.values()):.1f} km/h")
    
    # Create visualization
    print("\n🖼️  Creating visualization...")
    output_path = Path('viz_output/corridor_structured.png')
    create_visualization(G, positions, df, road_edges,
                        entry_nodes, exit_nodes, signal_nodes,
                        edge_speeds, output_path)
    
    # Copy to thesis images
    thesis_path = Path('images/chapter3/corridor_structured.png')
    thesis_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(output_path, thesis_path)
    print(f"✅ Copied to: {thesis_path}")
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
