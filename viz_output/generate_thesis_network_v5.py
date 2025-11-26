#!/usr/bin/env python3
"""
🎯 THESIS NETWORK VISUALIZATION V5
===================================

CLEAN version showing ONLY infrastructure on our corridor:
- Traffic signals only where they're connected to our segments
- Signals placed exactly on segment nodes (not raw OSM coords)
- Clear corridor structure without external noise
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


def load_enriched_data():
    """Load corridor data."""
    print("Loading corridor data...")
    
    segments_path = DATA_DIR / 'corridor_segments_enriched.csv'
    df_segments = pd.read_csv(segments_path)
    
    print(f"   Segments: {len(df_segments)}")
    
    # Count signals ON our segments (not all OSM signals in the area)
    signals_on_from = df_segments[df_segments['has_signal_from'] == 1]
    signals_on_to = df_segments[df_segments['has_signal_to'] == 1]
    
    # Get unique signal node positions from our segments
    signal_nodes = set()
    for _, seg in signals_on_from.iterrows():
        signal_nodes.add((seg['lon_from'], seg['lat_from']))
    for _, seg in signals_on_to.iterrows():
        signal_nodes.add((seg['lon_to'], seg['lat_to']))
    
    print(f"   Signals ON corridor: {len(signal_nodes)}")
    
    return df_segments, list(signal_nodes)


def generate_traffic_speeds(df_segments):
    """Generate realistic traffic speeds."""
    np.random.seed(42)
    
    speeds = []
    for _, seg in df_segments.iterrows():
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


def create_thesis_figure(df_segments, signal_nodes, speeds, output_path):
    """Create clean visualization with ONLY corridor infrastructure."""
    
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_facecolor('#f8f9fa')
    
    cmap, norm = create_colormap()
    
    # Road width by importance
    road_widths = {
        'Ahmadu Bello Way': 3.0,
        'Akin Adesola Street': 2.5,
        'Adeola Odeku Street': 2.2,
        'Saka Tinubu Street': 2.0
    }
    
    # ================================================================
    # 1. DRAW ROAD SEGMENTS
    # ================================================================
    print("   Drawing road segments...")
    
    # First: gray outline
    for _, seg in df_segments.iterrows():
        lon1, lat1 = seg['lon_from'], seg['lat_from']
        lon2, lat2 = seg['lon_to'], seg['lat_to']
        road_name = seg['road_name']
        lw = road_widths.get(road_name, 2.0)
        
        ax.plot([lon1, lon2], [lat1, lat2], 
                color='#333333', linewidth=lw + 1.0, 
                solid_capstyle='round', zorder=2)
    
    # Second: colored roads
    for idx, seg in df_segments.iterrows():
        lon1, lat1 = seg['lon_from'], seg['lat_from']
        lon2, lat2 = seg['lon_to'], seg['lat_to']
        speed = speeds[idx]
        color = cmap(norm(speed))
        road_name = seg['road_name']
        lw = road_widths.get(road_name, 2.0)
        
        ax.plot([lon1, lon2], [lat1, lat2], 
                color=color, linewidth=lw, 
                solid_capstyle='round', zorder=3)
    
    # ================================================================
    # 2. TRAFFIC SIGNALS - ONLY ON OUR SEGMENTS
    # ================================================================
    print(f"   Drawing {len(signal_nodes)} traffic signals ON corridor...")
    
    for lon, lat in signal_nodes:
        # Traffic light - placed exactly on our node
        ax.plot(lon, lat, 's', markersize=10, color='#1a252f',
               markeredgecolor='white', markeredgewidth=1.5, zorder=12)
        # Red light indicator
        ax.plot(lon, lat + 0.0002, 'o', markersize=4, color='#e74c3c', zorder=13)
    
    # ================================================================
    # 3. ENTRY/EXIT POINTS
    # ================================================================
    print("   Drawing entry/exit points...")
    
    entries = df_segments[df_segments['is_way_entry'] == 1][['lon_from', 'lat_from', 'road_name']].drop_duplicates()
    exits = df_segments[df_segments['is_way_exit'] == 1][['lon_to', 'lat_to', 'road_name']].drop_duplicates()
    
    for _, row in entries.iterrows():
        lon, lat = row['lon_from'], row['lat_from']
        ax.plot(lon, lat, '>', markersize=16, color='#27ae60',
               markeredgecolor='white', markeredgewidth=2.5, zorder=15)
    
    for _, row in exits.iterrows():
        lon, lat = row['lon_to'], row['lat_to']
        ax.plot(lon, lat, 's', markersize=12, color='#c0392b',
               markeredgecolor='white', markeredgewidth=2.5, zorder=15)
    
    # ================================================================
    # 4. INTERSECTIONS (only non-signal nodes)
    # ================================================================
    print("   Drawing intersections...")
    
    nodes = set()
    for _, seg in df_segments.iterrows():
        nodes.add((seg['lon_from'], seg['lat_from']))
        nodes.add((seg['lon_to'], seg['lat_to']))
    
    signal_set = set(signal_nodes)
    
    for lon, lat in nodes:
        # Skip if it's a signal node
        is_signal = any(
            abs(lon - sx) < 0.0001 and abs(lat - sy) < 0.0001 
            for sx, sy in signal_set
        )
        if not is_signal:
            ax.plot(lon, lat, 'o', markersize=5, color='white',
                   markeredgecolor='#555555', markeredgewidth=1, zorder=8)
    
    # ================================================================
    # 5. ROAD LABELS
    # ================================================================
    print("   Adding road labels...")
    
    for road_name in df_segments['road_name'].unique():
        road_df = df_segments[df_segments['road_name'] == road_name]
        
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
    cbar.ax.tick_params(labelsize=9)
    
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
    # 8. TITLES AND STYLING
    # ================================================================
    ax.set_title('Reseau Routier du Corridor Victoria Island\nLagos, Nigeria',
                fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Longitude (°E)', fontsize=11)
    ax.set_ylabel('Latitude (°N)', fontsize=11)
    
    ax.grid(True, linestyle=':', alpha=0.4, color='#cccccc')
    ax.tick_params(axis='both', labelsize=9)
    
    # Auto-fit with margin
    ax.autoscale_view()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    margin = 0.002
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_aspect('equal', adjustable='box')
    
    # Stats box - with CORRECTED signal count
    total_km = df_segments['length_m'].sum() / 1000
    n_roads = df_segments['road_name'].nunique()
    stats = (f"Routes: {n_roads}\n"
             f"Segments: {len(df_segments)}\n"
             f"Feux sur corridor: {len(signal_nodes)}\n"
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
    print("THESIS NETWORK V5 - Corridor-Only Infrastructure")
    print("=" * 60)
    
    df_segments, signal_nodes = load_enriched_data()
    speeds = generate_traffic_speeds(df_segments)
    
    print("\nCreating visualization...")
    output_path = OUTPUT_DIR / 'thesis_network_v5.png'
    fig = create_thesis_figure(df_segments, signal_nodes, speeds, output_path)
    plt.close(fig)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
