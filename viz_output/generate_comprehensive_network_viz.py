#!/usr/bin/env python3
"""
🎨 Comprehensive Network Visualization for Thesis
==================================================

Creates a single, information-rich figure showing:
- Traffic flow with speed-based coloring
- Traffic lights (🚦)
- Entry/Exit points (▶ / ◀)
- Road names (discrete, non-repeating labels)
- Network topology

All elements are balanced for readability.
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
import networkx as nx
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def create_speed_colormap(v_min: float = 0.0, v_max: float = 80.0):
    """Create continuous colormap for speed visualization."""
    colors = [
        '#8B0000', '#CC0000', '#FF0000', '#FF3300', '#FF6600',
        '#FF9900', '#FFCC00', '#FFFF00', '#CCFF00', '#99FF00',
        '#66FF00', '#33FF00', '#00FF00', '#00FF44', '#00FF88'
    ]
    cmap = LinearSegmentedColormap.from_list('speed', colors, N=256)
    norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
    return cmap, norm


def load_enriched_data():
    """Load the enriched network data."""
    excel_path = Path('arz_model/data/fichier_de_travail_corridor_enriched.xlsx')
    df = pd.read_excel(excel_path)
    return df


def build_graph_with_geo(df):
    """Build NetworkX graph with geographic positions."""
    G = nx.DiGraph()
    
    # Add edges with attributes
    for _, row in df.iterrows():
        u, v = str(row['u']), str(row['v'])
        G.add_edge(u, v, 
                   name=row['name_clean'],
                   highway=row['highway'],
                   length=row['length'])
    
    # Compute positions from lat/lon
    positions = {}
    for _, row in df.iterrows():
        u, v = str(row['u']), str(row['v'])
        if u not in positions and pd.notna(row['u_lat']) and pd.notna(row['u_lon']):
            positions[u] = (row['u_lon'], row['u_lat'])
        if v not in positions and pd.notna(row['v_lat']) and pd.notna(row['v_lon']):
            positions[v] = (row['v_lon'], row['v_lat'])
    
    # Fill missing positions with spring layout for those nodes
    missing = [n for n in G.nodes() if n not in positions]
    if missing:
        # Use spring layout for missing nodes, anchored to known positions
        subgraph_pos = nx.spring_layout(G, pos=positions, fixed=list(positions.keys()), k=0.5)
        positions.update({n: subgraph_pos[n] for n in missing})
    
    return G, positions


def identify_network_elements(G, df):
    """Identify entry points, exit points, and traffic lights."""
    
    # Entry points: nodes with in-degree = 0
    entry_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    
    # Exit points: nodes with out-degree = 0
    exit_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    # Traffic lights: from enriched data
    signal_nodes = set()
    for _, row in df.iterrows():
        if row.get('u_has_signal') == True:
            signal_nodes.add(str(row['u']))
        if row.get('v_has_signal') == True:
            signal_nodes.add(str(row['v']))
    
    # Junctions: nodes with degree > 2
    junction_nodes = [n for n in G.nodes() 
                      if G.in_degree(n) + G.out_degree(n) > 2 
                      and n not in entry_nodes 
                      and n not in exit_nodes]
    
    return entry_nodes, exit_nodes, list(signal_nodes), junction_nodes


def compute_road_label_positions(G, positions, df):
    """
    Compute ONE label position per road name (not per segment).
    Places label at the middle of the road's extent.
    """
    # Group edges by road name
    road_edges = defaultdict(list)
    for _, row in df.iterrows():
        name = row['name_clean']
        if pd.notna(name):
            u, v = str(row['u']), str(row['v'])
            road_edges[name].append((u, v))
    
    road_labels = {}
    for road_name, edges in road_edges.items():
        # Collect all node positions for this road
        all_points = []
        for u, v in edges:
            if u in positions:
                all_points.append(positions[u])
            if v in positions:
                all_points.append(positions[v])
        
        if all_points:
            # Find the center of the road
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            
            # Use the midpoint along the road extent
            center_x = (min(xs) + max(xs)) / 2
            center_y = (min(ys) + max(ys)) / 2
            
            # Calculate road angle for text rotation
            if len(all_points) >= 2:
                # Sort points to find general direction
                sorted_by_x = sorted(all_points, key=lambda p: p[0])
                dx = sorted_by_x[-1][0] - sorted_by_x[0][0]
                dy = sorted_by_x[-1][1] - sorted_by_x[0][1]
                angle = np.degrees(np.arctan2(dy, dx))
                # Keep angle readable (not upside down)
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180
            else:
                angle = 0
            
            road_labels[road_name] = {
                'pos': (center_x, center_y),
                'angle': angle
            }
    
    return road_labels


def generate_traffic_scenario(G, positions):
    """Generate realistic traffic scenario with spatial variation."""
    
    # Calculate node centrality for congestion pattern
    entry_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    
    # Distance from entry nodes determines congestion
    edge_speeds = {}
    
    for u, v in G.edges():
        # Base speed with some randomness
        base_speed = 55 + np.random.normal(0, 10)
        
        # Check proximity to entry nodes (more congestion near entry)
        min_dist_to_entry = float('inf')
        for entry in entry_nodes:
            try:
                dist = nx.shortest_path_length(G, entry, u)
                min_dist_to_entry = min(min_dist_to_entry, dist)
            except nx.NetworkXNoPath:
                pass
        
        # Congestion factor based on distance from entry
        if min_dist_to_entry < 2:
            # Near entry: congested (red/orange)
            speed = 15 + np.random.normal(0, 5)
        elif min_dist_to_entry < 4:
            # Moderate distance: moderate congestion (yellow/orange)
            speed = 35 + np.random.normal(0, 8)
        else:
            # Far from entry: free flow (green)
            speed = 60 + np.random.normal(0, 10)
        
        speed = np.clip(speed, 5, 80)
        edge_speeds[(u, v)] = speed
    
    return edge_speeds


def create_comprehensive_figure(G, positions, df, edge_speeds, 
                                 entry_nodes, exit_nodes, signal_nodes, 
                                 road_labels, output_path):
    """Create the comprehensive visualization."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    
    # Create colormap
    cmap, norm = create_speed_colormap()
    
    # ============================================================
    # 1. DRAW EDGES (Traffic flow with speed-based colors)
    # ============================================================
    for (u, v), speed in edge_speeds.items():
        if u in positions and v in positions:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            
            color = cmap(norm(speed))
            
            # Draw edge as arrow
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(
                           arrowstyle='-|>',
                           color=color,
                           lw=3.5,
                           mutation_scale=12,
                           shrinkA=8,
                           shrinkB=8
                       ))
    
    # ============================================================
    # 2. DRAW NODES (Different styles for different types)
    # ============================================================
    
    # Regular nodes (small, discrete)
    regular_nodes = [n for n in G.nodes() 
                     if n not in entry_nodes 
                     and n not in exit_nodes 
                     and n not in signal_nodes]
    
    for node in regular_nodes:
        if node in positions:
            x, y = positions[node]
            ax.plot(x, y, 'o', markersize=6, color='white', 
                   markeredgecolor='#333333', markeredgewidth=1.5, zorder=5)
    
    # ============================================================
    # 3. DRAW TRAFFIC LIGHTS (🚦 symbol)
    # ============================================================
    for node in signal_nodes:
        if node in positions:
            x, y = positions[node]
            # Draw traffic light as colored circle with border
            ax.plot(x, y, 's', markersize=14, color='#FFD700', 
                   markeredgecolor='#333333', markeredgewidth=2, zorder=10)
            # Add small red/green dots inside
            ax.plot(x, y + 0.00008, 'o', markersize=4, color='#FF0000', zorder=11)
            ax.plot(x, y - 0.00008, 'o', markersize=4, color='#00FF00', zorder=11)
    
    # ============================================================
    # 4. DRAW ENTRY POINTS (Green arrow pointing IN)
    # ============================================================
    for node in entry_nodes:
        if node in positions:
            x, y = positions[node]
            # Entry marker: green triangle pointing right
            ax.plot(x, y, '>', markersize=18, color='#00AA00', 
                   markeredgecolor='#005500', markeredgewidth=2, zorder=10)
    
    # ============================================================
    # 5. DRAW EXIT POINTS (Red square)
    # ============================================================
    for node in exit_nodes:
        if node in positions:
            x, y = positions[node]
            # Exit marker: red octagon-like
            ax.plot(x, y, 'H', markersize=16, color='#CC0000', 
                   markeredgecolor='#660000', markeredgewidth=2, zorder=10)
    
    # ============================================================
    # 6. ADD ROAD NAMES (Discrete, one per road)
    # ============================================================
    for road_name, label_info in road_labels.items():
        x, y = label_info['pos']
        angle = label_info['angle']
        
        # Shorten very long names
        display_name = road_name
        if len(road_name) > 20:
            display_name = road_name[:18] + '...'
        
        # Add text with subtle background
        ax.text(x, y, display_name,
               fontsize=8,
               fontweight='bold',
               color='#1a1a1a',
               rotation=angle,
               ha='center',
               va='center',
               bbox=dict(boxstyle='round,pad=0.2', 
                        facecolor='white', 
                        edgecolor='none',
                        alpha=0.75),
               zorder=15)
    
    # ============================================================
    # 7. ADD COLORBAR (Speed legend)
    # ============================================================
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar_ax = fig.add_axes([0.15, 0.06, 0.5, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Vitesse moyenne (km/h)', fontsize=11, fontweight='bold')
    
    # Add regime labels below colorbar
    cbar.ax.text(0.06, -2.0, 'Bouchon', ha='center', va='top', fontsize=9, transform=cbar.ax.transAxes)
    cbar.ax.text(0.30, -2.0, 'Congestion', ha='center', va='top', fontsize=9, transform=cbar.ax.transAxes)
    cbar.ax.text(0.55, -2.0, 'Modéré', ha='center', va='top', fontsize=9, transform=cbar.ax.transAxes)
    cbar.ax.text(0.80, -2.0, 'Fluide', ha='center', va='top', fontsize=9, transform=cbar.ax.transAxes)
    cbar.ax.text(0.95, -2.0, 'Libre', ha='center', va='top', fontsize=9, transform=cbar.ax.transAxes)
    
    # ============================================================
    # 8. ADD LEGEND (Network elements)
    # ============================================================
    legend_elements = [
        Line2D([0], [0], marker='>', color='w', markerfacecolor='#00AA00', 
               markeredgecolor='#005500', markersize=14, label='Point d\'entrée'),
        Line2D([0], [0], marker='H', color='w', markerfacecolor='#CC0000', 
               markeredgecolor='#660000', markersize=12, label='Point de sortie'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFD700', 
               markeredgecolor='#333333', markersize=12, label='Feu de signalisation'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
               markeredgecolor='#333333', markersize=8, label='Intersection'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
             framealpha=0.95, edgecolor='#333333', fancybox=True)
    
    # ============================================================
    # 9. ADD TITLE AND STATISTICS BOX
    # ============================================================
    ax.set_title('Réseau Routier de Victoria Island - Simulation du Trafic\n'
                 'avec Feux de Signalisation et Points d\'Accès',
                fontsize=14, fontweight='bold', pad=20)
    
    # Statistics box (without emojis for compatibility)
    stats_text = (
        f"STATISTIQUES DU RESEAU\n"
        f"----------------------\n"
        f"Segments routiers: {G.number_of_edges()}\n"
        f"Intersections: {G.number_of_nodes()}\n"
        f"Points d'entree: {len(entry_nodes)}\n"
        f"Points de sortie: {len(exit_nodes)}\n"
        f"Feux de signalisation: {len(signal_nodes)}\n"
        f"Routes nommees: {len(road_labels)}"
    )
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', 
                    edgecolor='#666666', alpha=0.92))
    
    # Remove axes
    ax.axis('off')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✅ Figure saved to: {output_path}")


def main():
    print("=" * 70)
    print("🎨 COMPREHENSIVE NETWORK VISUALIZATION")
    print("   Traffic + Signals + Entry/Exit + Road Names")
    print("=" * 70)
    
    # Load data
    print("\n📊 Loading enriched network data...")
    df = load_enriched_data()
    print(f"   ✓ Loaded {len(df)} segments")
    
    # Build graph with geographic positions
    print("\n🗺️  Building network graph with geo-positions...")
    G, positions = build_graph_with_geo(df)
    print(f"   ✓ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Identify network elements
    print("\n🔍 Identifying network elements...")
    entry_nodes, exit_nodes, signal_nodes, junction_nodes = identify_network_elements(G, df)
    print(f"   ✓ Entry points: {len(entry_nodes)}")
    print(f"   ✓ Exit points: {len(exit_nodes)}")
    print(f"   ✓ Traffic lights: {len(signal_nodes)}")
    print(f"   ✓ Junctions: {len(junction_nodes)}")
    
    # Compute road label positions (ONE per road)
    print("\n🏷️  Computing road label positions (one per road)...")
    road_labels = compute_road_label_positions(G, positions, df)
    print(f"   ✓ Road labels: {len(road_labels)}")
    for name in road_labels:
        print(f"      - {name}")
    
    # Generate traffic scenario
    print("\n🚗 Generating traffic scenario...")
    edge_speeds = generate_traffic_scenario(G, positions)
    
    # Create visualization
    print("\n🖼️  Creating comprehensive visualization...")
    output_path = Path('viz_output/comprehensive_network_simulation.png')
    
    create_comprehensive_figure(
        G, positions, df, edge_speeds,
        entry_nodes, exit_nodes, signal_nodes,
        road_labels, output_path
    )
    
    # Copy to thesis images
    thesis_path = Path('images/chapter3/comprehensive_network.png')
    thesis_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(output_path, thesis_path)
    print(f"✅ Copied to: {thesis_path}")
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
