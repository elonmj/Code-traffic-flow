#!/usr/bin/env python3
"""
🎯 THESIS NETWORK VISUALIZATION - FINAL VERSION
================================================

Uses the FINAL CLEAN CSV files:
- corridor_final_segments.csv
- corridor_final_signals.csv  
- corridor_final_nodes.csv

No calculations needed - just read and plot!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / 'arz_model' / 'data'
OUTPUT_DIR = Path(__file__).parent


def load_data():
    """Load final clean CSV files."""
    print("Loading final corridor data...")
    
    segments = pd.read_csv(DATA_DIR / 'corridor_final_segments.csv')
    signals = pd.read_csv(DATA_DIR / 'corridor_final_signals.csv')
    nodes = pd.read_csv(DATA_DIR / 'corridor_final_nodes.csv')
    
    print(f"   Segments: {len(segments)}")
    print(f"   Signals ON corridor: {len(signals)}")
    print(f"   Nodes: {len(nodes)}")
    
    return segments, signals, nodes


def generate_traffic_speeds(segments):
    """Generate realistic traffic speeds."""
    np.random.seed(42)
    
    speeds = []
    for _, seg in segments.iterrows():
        road = seg['road_name']
        
        if 'Ahmadu Bello' in road:
            base = 25
        elif 'Akin Adesola' in road:
            base = 35
        elif 'Adeola Odeku' in road:
            base = 45
        else:
            base = 40
        
        variation = np.random.normal(0, 8)
        if seg['has_signal_from'] or seg['has_signal_to']:
            base *= 0.7
        
        speeds.append(np.clip(base + variation, 5, 80))
    
    return np.array(speeds)


def create_colormap():
    """Traffic speed colormap."""
    colors = ['#8B0000', '#CC0000', '#FF4444', '#FF8800', '#FFCC00',
              '#FFFF00', '#AAFF00', '#66FF33', '#00DD00', '#00AA44']
    cmap = LinearSegmentedColormap.from_list('traffic', colors, N=256)
    norm = mcolors.Normalize(vmin=0, vmax=80)
    return cmap, norm


def create_figure(segments, signals, nodes, speeds, output_path):
    """Create the final visualization."""
    
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_facecolor('#f8f9fa')
    
    cmap, norm = create_colormap()
    
    # Road widths
    road_widths = {
        'Ahmadu Bello Way': 3.0,
        'Akin Adesola Street': 2.5,
        'Adeola Odeku Street': 2.2,
        'Saka Tinubu Street': 2.0
    }
    
    # ================================================================
    # 1. DRAW SEGMENTS
    # ================================================================
    print("   Drawing segments...")
    
    # Gray outline first
    for _, seg in segments.iterrows():
        lw = road_widths.get(seg['road_name'], 2.0)
        ax.plot([seg['lon_from'], seg['lon_to']], 
                [seg['lat_from'], seg['lat_to']], 
                color='#333333', linewidth=lw + 1.0, 
                solid_capstyle='round', zorder=2)
    
    # Colored roads
    for idx, seg in segments.iterrows():
        lw = road_widths.get(seg['road_name'], 2.0)
        color = cmap(norm(speeds[idx]))
        ax.plot([seg['lon_from'], seg['lon_to']], 
                [seg['lat_from'], seg['lat_to']], 
                color=color, linewidth=lw, 
                solid_capstyle='round', zorder=3)
    
    # ================================================================
    # 2. DRAW SIGNALS (from clean signals CSV)
    # ================================================================
    print(f"   Drawing {len(signals)} traffic signals...")
    
    for _, sig in signals.iterrows():
        ax.plot(sig['lon'], sig['lat'], 's', markersize=10, color='#1a252f',
               markeredgecolor='white', markeredgewidth=1.5, zorder=12)
        ax.plot(sig['lon'], sig['lat'] + 0.0002, 'o', markersize=4, 
               color='#e74c3c', zorder=13)
    
    # ================================================================
    # 3. DRAW ENTRY/EXIT POINTS (from nodes CSV)
    # ================================================================
    print("   Drawing entry/exit points...")
    
    entries = nodes[nodes['point_type'] == 'entry']
    exits = nodes[nodes['point_type'] == 'exit']
    
    for _, node in entries.iterrows():
        ax.plot(node['lon'], node['lat'], '>', markersize=16, color='#27ae60',
               markeredgecolor='white', markeredgewidth=2.5, zorder=15)
    
    for _, node in exits.iterrows():
        ax.plot(node['lon'], node['lat'], 's', markersize=12, color='#c0392b',
               markeredgecolor='white', markeredgewidth=2.5, zorder=15)
    
    # ================================================================
    # 4. DRAW INTERSECTIONS (non-signal nodes)
    # ================================================================
    print("   Drawing intersections...")
    
    intersections = nodes[(nodes['is_signal'] == 0) & (nodes['point_type'] == 'intersection')]
    
    for _, node in intersections.iterrows():
        ax.plot(node['lon'], node['lat'], 'o', markersize=5, color='white',
               markeredgecolor='#555555', markeredgewidth=1, zorder=8)
    
    # ================================================================
    # 5. ROAD LABELS
    # ================================================================
    print("   Adding labels...")
    
    for road_name in segments['road_name'].unique():
        road_df = segments[segments['road_name'] == road_name]
        
        cx = road_df['lon_center'].mean()
        cy = road_df['lat_center'].mean()
        
        first = road_df.iloc[0]
        last = road_df.iloc[-1]
        dx = last['lon_to'] - first['lon_from']
        dy = last['lat_to'] - first['lat_from']
        angle = np.degrees(np.arctan2(dy, dx))
        if angle > 90: angle -= 180
        if angle < -90: angle += 180
        
        ax.text(cx, cy + 0.0012, road_name,
               fontsize=10, fontweight='bold', fontstyle='italic',
               color='#111111', rotation=angle,
               ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white', edgecolor='#888888',
                        alpha=0.95, linewidth=1.2),
               zorder=20)
    
    # ================================================================
    # 6. COLORBAR
    # ================================================================
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
    cbar.set_label('Vitesse (km/h)', fontsize=11, fontweight='bold')
    
    # ================================================================
    # 7. LEGEND
    # ================================================================
    legend_elements = [
        Line2D([0], [0], marker='>', color='w', label="Point d'entree",
               markerfacecolor='#27ae60', markeredgecolor='white',
               markersize=12, markeredgewidth=2),
        Line2D([0], [0], marker='s', color='w', label='Point de sortie',
               markerfacecolor='#c0392b', markeredgecolor='white',
               markersize=10, markeredgewidth=2),
        Line2D([0], [0], marker='s', color='w', label='Feu de signalisation',
               markerfacecolor='#1a252f', markeredgecolor='white',
               markersize=9, markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', label='Intersection',
               markerfacecolor='white', markeredgecolor='#555555',
               markersize=7, markeredgewidth=1),
        Line2D([0], [0], color='#FF8800', linewidth=4, label='Segment routier'),
    ]
    
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
             framealpha=0.95, edgecolor='#888', fancybox=True)
    
    # ================================================================
    # 8. TITLES
    # ================================================================
    ax.set_title('Reseau Routier du Corridor Victoria Island\nLagos, Nigeria',
                fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Longitude (°E)', fontsize=11)
    ax.set_ylabel('Latitude (°N)', fontsize=11)
    
    ax.grid(True, linestyle=':', alpha=0.4, color='#cccccc')
    
    # Auto-fit
    ax.autoscale_view()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    margin = 0.002
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_aspect('equal', adjustable='box')
    
    # Stats box
    total_km = segments['length_m'].sum() / 1000
    stats = (f"Routes: {segments['road_name'].nunique()}\n"
             f"Segments: {len(segments)}\n"
             f"Feux: {len(signals)}\n"
             f"Longueur: {total_km:.1f} km")
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9,
           va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved: {output_path}")
    
    return fig


def main():
    print("=" * 60)
    print("THESIS NETWORK - FINAL VERSION")
    print("=" * 60)
    
    segments, signals, nodes = load_data()
    speeds = generate_traffic_speeds(segments)
    
    print("\nCreating visualization...")
    output_path = OUTPUT_DIR / 'thesis_network_final.png'
    fig = create_figure(segments, signals, nodes, speeds, output_path)
    plt.close(fig)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
