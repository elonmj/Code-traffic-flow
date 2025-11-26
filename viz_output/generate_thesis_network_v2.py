#!/usr/bin/env python3
"""
🎯 THESIS NETWORK VISUALIZATION V2
===================================

Uses the enriched CSV data to create a beautiful visualization
with clearly visible road segments.

Key improvements:
- Uses pre-fetched CSV data (no API calls)
- Draws segments as thick colored lines
- Better traffic light and marker visibility
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / 'arz_model' / 'data'
OUTPUT_DIR = Path(__file__).parent


def load_enriched_data():
    """Load pre-fetched corridor data."""
    print("📂 Loading enriched corridor data...")
    
    segments_path = DATA_DIR / 'corridor_segments_enriched.csv'
    signals_path = DATA_DIR / 'corridor_traffic_signals.csv'
    nodes_path = DATA_DIR / 'corridor_nodes.csv'
    
    df_segments = pd.read_csv(segments_path)
    df_signals = pd.read_csv(signals_path)
    df_nodes = pd.read_csv(nodes_path)
    
    print(f"   ✅ Segments: {len(df_segments)}")
    print(f"   ✅ Traffic signals: {len(df_signals)}")
    print(f"   ✅ Nodes: {len(df_nodes)}")
    
    return df_segments, df_signals, df_nodes


def generate_traffic_speeds(df_segments):
    """Generate realistic traffic speeds for visualization."""
    np.random.seed(42)
    
    speeds = []
    for _, seg in df_segments.iterrows():
        road = seg['road_name']
        
        # Base speeds by road type
        if 'Ahmadu Bello' in road:
            base = 25  # Main artery - most congested
        elif 'Akin Adesola' in road:
            base = 35
        elif 'Adeola Odeku' in road:
            base = 45
        else:
            base = 40
        
        # Add variation
        variation = np.random.normal(0, 8)
        
        # Traffic lights slow things down
        if seg['has_signal_from'] or seg['has_signal_to']:
            base *= 0.7
        
        speed = np.clip(base + variation, 5, 80)
        speeds.append(speed)
    
    return speeds


def create_colormap():
    """Create traffic speed colormap (red=slow, green=fast)."""
    colors = ['#8B0000', '#CC0000', '#FF4444', '#FF8800', '#FFCC00',
              '#FFFF00', '#AAFF00', '#66FF33', '#00DD00', '#00AA44']
    cmap = LinearSegmentedColormap.from_list('traffic', colors, N=256)
    norm = mcolors.Normalize(vmin=0, vmax=80)
    return cmap, norm


def create_thesis_figure(df_segments, df_signals, df_nodes, speeds, output_path):
    """Create the thesis visualization with visible segments."""
    
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_facecolor('#f5f9fc')
    
    cmap, norm = create_colormap()
    
    # ================================================================
    # 1. DRAW ROAD SEGMENTS (as thick colored lines)
    # ================================================================
    print("   📍 Drawing road segments...")
    
    # Group by road for different base widths
    road_widths = {
        'Ahmadu Bello Way': 5.0,
        'Akin Adesola Street': 4.0,
        'Adeola Odeku Street': 3.5,
        'Saka Tinubu Street': 3.0
    }
    
    for idx, seg in df_segments.iterrows():
        lon1, lat1 = seg['lon_from'], seg['lat_from']
        lon2, lat2 = seg['lon_to'], seg['lat_to']
        speed = speeds[idx]
        color = cmap(norm(speed))
        
        road_name = seg['road_name']
        lw = road_widths.get(road_name, 3.0)
        
        # Draw the segment as a thick line
        ax.plot([lon1, lon2], [lat1, lat2], 
                color=color, linewidth=lw, solid_capstyle='round', 
                zorder=3, alpha=0.95)
        
        # Add direction arrow at segment midpoint (every 5th segment)
        if idx % 5 == 0:
            mid_lon = (lon1 + lon2) / 2
            mid_lat = (lat1 + lat2) / 2
            dx = (lon2 - lon1) * 0.3
            dy = (lat2 - lat1) * 0.3
            ax.annotate('', 
                       xy=(mid_lon + dx/2, mid_lat + dy/2),
                       xytext=(mid_lon - dx/2, mid_lat - dy/2),
                       arrowprops=dict(arrowstyle='->', color='white',
                                      lw=1.5, mutation_scale=10),
                       zorder=4)
    
    # ================================================================
    # 2. DRAW INTERSECTIONS
    # ================================================================
    print("   🔵 Drawing intersections...")
    
    # Get unique intersection coordinates
    intersections = set()
    for _, seg in df_segments.iterrows():
        intersections.add((seg['lon_from'], seg['lat_from']))
        intersections.add((seg['lon_to'], seg['lat_to']))
    
    # Get nodes with signals
    signal_locs = set()
    for _, sig in df_signals.iterrows():
        signal_locs.add((round(sig['lon'], 5), round(sig['lat'], 5)))
    
    for lon, lat in intersections:
        # Check if this is near a traffic signal
        is_signal = any(
            np.sqrt((lon - s[0])**2 + (lat - s[1])**2) < 0.0005
            for s in signal_locs
        )
        
        if not is_signal:
            circle = Circle((lon, lat), 0.00035, 
                           facecolor='white', edgecolor='#555555', 
                           linewidth=1.5, zorder=7)
            ax.add_patch(circle)
    
    # ================================================================
    # 3. TRAFFIC SIGNALS (more prominent)
    # ================================================================
    print("   🚦 Drawing traffic signals...")
    
    for _, sig in df_signals.iterrows():
        lon, lat = sig['lon'], sig['lat']
        
        # Traffic light post
        rect = FancyBboxPatch(
            (lon - 0.0004, lat - 0.0006), 0.0008, 0.0012,
            boxstyle="round,pad=0.02",
            facecolor='#2c3e50', edgecolor='#1a252f',
            linewidth=2, zorder=12
        )
        ax.add_patch(rect)
        
        # Lights (red, yellow, green)
        ax.plot(lon, lat + 0.0003, 'o', markersize=6, color='#e74c3c', 
               markeredgecolor='#c0392b', markeredgewidth=0.5, zorder=13)
        ax.plot(lon, lat, 'o', markersize=6, color='#f39c12',
               markeredgecolor='#d68910', markeredgewidth=0.5, zorder=13)
        ax.plot(lon, lat - 0.0003, 'o', markersize=6, color='#2ecc71',
               markeredgecolor='#27ae60', markeredgewidth=0.5, zorder=13)
    
    # ================================================================
    # 4. ENTRY/EXIT POINTS
    # ================================================================
    print("   🚗 Drawing entry/exit points...")
    
    # Find entry points (way entries)
    entries = df_segments[df_segments['is_way_entry'] == 1][['lon_from', 'lat_from', 'road_name']].drop_duplicates()
    exits = df_segments[df_segments['is_way_exit'] == 1][['lon_to', 'lat_to', 'road_name']].drop_duplicates()
    
    for _, row in entries.head(6).iterrows():
        lon, lat = row['lon_from'], row['lat_from']
        ax.plot(lon, lat, '>', markersize=20, color='#27ae60',
               markeredgecolor='#1e8449', markeredgewidth=2.5, zorder=15)
        ax.text(lon - 0.001, lat, 'IN', fontsize=8, fontweight='bold',
               color='#1e8449', ha='right', va='center', zorder=16)
    
    for _, row in exits.head(6).iterrows():
        lon, lat = row['lon_to'], row['lat_to']
        ax.plot(lon, lat, 'H', markersize=18, color='#c0392b',
               markeredgecolor='#922b21', markeredgewidth=2.5, zorder=15)
        ax.text(lon + 0.001, lat, 'OUT', fontsize=8, fontweight='bold',
               color='#922b21', ha='left', va='center', zorder=16)
    
    # ================================================================
    # 5. ROAD LABELS
    # ================================================================
    print("   🏷️  Adding road labels...")
    
    for road_name in df_segments['road_name'].unique():
        road_df = df_segments[df_segments['road_name'] == road_name]
        
        # Get center of road
        cx = road_df['lon_center'].mean()
        cy = road_df['lat_center'].mean()
        
        # Calculate road angle
        if len(road_df) > 1:
            start_idx = road_df.index[0]
            end_idx = road_df.index[-1]
            dx = road_df.loc[end_idx, 'lon_to'] - road_df.loc[start_idx, 'lon_from']
            dy = road_df.loc[end_idx, 'lat_to'] - road_df.loc[start_idx, 'lat_from']
            angle = np.degrees(np.arctan2(dy, dx))
            if angle > 90: angle -= 180
            if angle < -90: angle += 180
        else:
            angle = 0
        
        # Place label
        ax.text(cx, cy + 0.0015, road_name,
               fontsize=12, fontweight='bold', fontstyle='italic',
               color='#1a1a1a', rotation=angle,
               ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white', edgecolor='#888888',
                        alpha=0.95, linewidth=1.5),
               zorder=25)
    
    # ================================================================
    # 6. COLORBAR
    # ================================================================
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=25, pad=0.02)
    cbar.set_label('Vitesse du trafic (km/h)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Add speed annotations
    cbar.ax.text(1.3, 0.1, 'Congestion', fontsize=10, transform=cbar.ax.transAxes, va='center', color='#8B0000')
    cbar.ax.text(1.3, 0.9, 'Fluide', fontsize=10, transform=cbar.ax.transAxes, va='center', color='#00AA44')
    
    # ================================================================
    # 7. LEGEND
    # ================================================================
    legend_elements = [
        Line2D([0], [0], marker='>', color='w', label='Point d\'entrée',
               markerfacecolor='#27ae60', markeredgecolor='#1e8449',
               markersize=14, markeredgewidth=2),
        Line2D([0], [0], marker='H', color='w', label='Point de sortie',
               markerfacecolor='#c0392b', markeredgecolor='#922b21',
               markersize=12, markeredgewidth=2),
        Line2D([0], [0], marker='s', color='w', label='Feu de signalisation',
               markerfacecolor='#2c3e50', markeredgecolor='#1a252f',
               markersize=12, markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', label='Intersection',
               markerfacecolor='white', markeredgecolor='#555555',
               markersize=10, markeredgewidth=1.5),
        Line2D([0], [0], color='#FF8800', linewidth=4, label='Segment routier'),
    ]
    
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11,
             framealpha=0.95, edgecolor='#aaaaaa', fancybox=True,
             title='Légende', title_fontsize=12)
    
    # ================================================================
    # 8. TITLES AND LABELS
    # ================================================================
    ax.set_title('Réseau Routier du Corridor Victoria Island\nLagos, Nigeria',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude (°E)', fontsize=13)
    ax.set_ylabel('Latitude (°N)', fontsize=13)
    
    # Grid and styling
    ax.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax.tick_params(axis='both', labelsize=11)
    
    # Set axis limits with padding
    lon_min = df_segments['lon_from'].min() - 0.002
    lon_max = df_segments['lon_to'].max() + 0.002
    lat_min = df_segments['lat_from'].min() - 0.002
    lat_max = df_segments['lat_to'].max() + 0.002
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    
    # Ensure equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Statistics annotation
    total_length = df_segments['length_m'].sum() / 1000
    n_signals = len(df_signals)
    n_segments = len(df_segments)
    
    stats_text = (f"Statistiques du reseau\n"
                  f"----------------------\n"
                  f"Segments: {n_segments}\n"
                  f"Feux: {n_signals}\n"
                  f"Longueur: {total_length:.2f} km\n"
                  f"Source: OpenStreetMap")
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5',
                    facecolor='white', edgecolor='#888888',
                    alpha=0.95))
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"\n✅ Figure saved: {output_path}")
    
    return fig, ax


def main():
    print("=" * 70)
    print("🎯 THESIS NETWORK VISUALIZATION V2")
    print("   Using enriched CSV data")
    print("=" * 70)
    
    # Load data
    df_segments, df_signals, df_nodes = load_enriched_data()
    
    # Generate traffic speeds
    print("\n🚗 Generating traffic speeds...")
    speeds = generate_traffic_speeds(df_segments)
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    output_path = OUTPUT_DIR / 'thesis_network_v2.png'
    fig, ax = create_thesis_figure(df_segments, df_signals, df_nodes, speeds, output_path)
    
    plt.close(fig)
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
