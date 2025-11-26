#!/usr/bin/env python3
"""
🎯 THESIS NETWORK VISUALIZATION V3
===================================

Fixed version with CLEARLY VISIBLE road segments.
Uses LineCollection for better rendering of road networks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
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
    
    df_segments = pd.read_csv(segments_path)
    df_signals = pd.read_csv(signals_path)
    
    print(f"   ✅ Segments: {len(df_segments)}")
    print(f"   ✅ Traffic signals: {len(df_signals)}")
    
    return df_segments, df_signals


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
    
    return np.array(speeds)


def create_colormap():
    """Create traffic speed colormap (red=slow, green=fast)."""
    colors = ['#8B0000', '#CC0000', '#FF4444', '#FF8800', '#FFCC00',
              '#FFFF00', '#AAFF00', '#66FF33', '#00DD00', '#00AA44']
    cmap = LinearSegmentedColormap.from_list('traffic', colors, N=256)
    norm = mcolors.Normalize(vmin=0, vmax=80)
    return cmap, norm


def create_thesis_figure(df_segments, df_signals, speeds, output_path):
    """Create the thesis visualization with VISIBLE segments."""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('#f0f4f8')
    
    cmap, norm = create_colormap()
    
    # ================================================================
    # 1. DRAW ROAD SEGMENTS USING LINE COLLECTION (THICK LINES)
    # ================================================================
    print("   📍 Drawing road segments with LineCollection...")
    
    # Build line segments
    segments_lines = []
    colors_list = []
    
    for idx, seg in df_segments.iterrows():
        lon1, lat1 = seg['lon_from'], seg['lat_from']
        lon2, lat2 = seg['lon_to'], seg['lat_to']
        
        segments_lines.append([(lon1, lat1), (lon2, lat2)])
        colors_list.append(cmap(norm(speeds[idx])))
    
    # Create LineCollection with VERY THICK lines
    lc = LineCollection(segments_lines, colors=colors_list, 
                        linewidths=8,  # VERY THICK
                        capstyle='round',
                        zorder=2)
    ax.add_collection(lc)
    
    # Add a darker outline for better visibility
    lc_outline = LineCollection(segments_lines, colors='#333333', 
                                linewidths=10,  # Slightly thicker for outline
                                capstyle='round',
                                zorder=1)
    ax.add_collection(lc_outline)
    
    # ================================================================
    # 2. DRAW DIRECTION ARROWS ON ROADS
    # ================================================================
    print("   ➡️  Adding direction arrows...")
    
    # Group segments by road and add arrows
    for road_name in df_segments['road_name'].unique():
        road_df = df_segments[df_segments['road_name'] == road_name]
        
        # Pick every 10th segment for an arrow
        for i in range(0, len(road_df), 10):
            seg = road_df.iloc[i]
            lon1, lat1 = seg['lon_from'], seg['lat_from']
            lon2, lat2 = seg['lon_to'], seg['lat_to']
            
            # Arrow at midpoint
            mid_lon = (lon1 + lon2) / 2
            mid_lat = (lat1 + lat2) / 2
            
            # Direction
            dx = lon2 - lon1
            dy = lat2 - lat1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx /= length
                dy /= length
                
                # Draw small arrow
                arrow_size = 0.0008
                ax.annotate('', 
                           xy=(mid_lon + dx*arrow_size, mid_lat + dy*arrow_size),
                           xytext=(mid_lon - dx*arrow_size, mid_lat - dy*arrow_size),
                           arrowprops=dict(arrowstyle='-|>', color='white',
                                          lw=2, mutation_scale=15),
                           zorder=5)
    
    # ================================================================
    # 3. TRAFFIC SIGNALS (prominent markers)
    # ================================================================
    print("   🚦 Drawing traffic signals...")
    
    for _, sig in df_signals.iterrows():
        lon, lat = sig['lon'], sig['lat']
        
        # Traffic light post (dark rectangle)
        rect = FancyBboxPatch(
            (lon - 0.0005, lat - 0.0007), 0.001, 0.0014,
            boxstyle="round,pad=0.02",
            facecolor='#2c3e50', edgecolor='#1a252f',
            linewidth=2, zorder=15
        )
        ax.add_patch(rect)
        
        # Lights (red, yellow, green)
        ax.plot(lon, lat + 0.0004, 'o', markersize=7, color='#e74c3c', 
               markeredgecolor='white', markeredgewidth=0.5, zorder=16)
        ax.plot(lon, lat, 'o', markersize=7, color='#f39c12',
               markeredgecolor='white', markeredgewidth=0.5, zorder=16)
        ax.plot(lon, lat - 0.0004, 'o', markersize=7, color='#2ecc71',
               markeredgecolor='white', markeredgewidth=0.5, zorder=16)
    
    # ================================================================
    # 4. ENTRY/EXIT POINTS
    # ================================================================
    print("   🚗 Drawing entry/exit points...")
    
    # Find entry points (way entries)
    entries = df_segments[df_segments['is_way_entry'] == 1][['lon_from', 'lat_from', 'road_name']].drop_duplicates()
    exits = df_segments[df_segments['is_way_exit'] == 1][['lon_to', 'lat_to', 'road_name']].drop_duplicates()
    
    for _, row in entries.head(8).iterrows():
        lon, lat = row['lon_from'], row['lat_from']
        ax.plot(lon, lat, '>', markersize=22, color='#27ae60',
               markeredgecolor='white', markeredgewidth=3, zorder=20)
    
    for _, row in exits.head(8).iterrows():
        lon, lat = row['lon_to'], row['lat_to']
        ax.plot(lon, lat, 's', markersize=16, color='#c0392b',
               markeredgecolor='white', markeredgewidth=3, zorder=20)
    
    # ================================================================
    # 5. INTERSECTIONS (small circles at junctions)
    # ================================================================
    print("   🔵 Drawing intersections...")
    
    # Get unique intersection coordinates
    intersections = set()
    for _, seg in df_segments.iterrows():
        intersections.add((seg['lon_from'], seg['lat_from']))
        intersections.add((seg['lon_to'], seg['lat_to']))
    
    # Get signal locations to exclude
    signal_locs = set()
    for _, sig in df_signals.iterrows():
        signal_locs.add((round(sig['lon'], 4), round(sig['lat'], 4)))
    
    for lon, lat in intersections:
        # Check if this is NOT near a traffic signal
        is_signal = any(
            np.sqrt((lon - s[0])**2 + (lat - s[1])**2) < 0.001
            for s in signal_locs
        )
        
        if not is_signal:
            circle = Circle((lon, lat), 0.0004, 
                           facecolor='white', edgecolor='#555555', 
                           linewidth=1.5, zorder=10)
            ax.add_patch(circle)
    
    # ================================================================
    # 6. ROAD LABELS
    # ================================================================
    print("   🏷️  Adding road labels...")
    
    label_positions = {
        'Ahmadu Bello Way': {'offset': (0, 0.003), 'rotation': 30},
        'Akin Adesola Street': {'offset': (0.002, 0), 'rotation': 80},
        'Adeola Odeku Street': {'offset': (0, 0.002), 'rotation': -5},
        'Saka Tinubu Street': {'offset': (0, 0.002), 'rotation': 60}
    }
    
    for road_name in df_segments['road_name'].unique():
        road_df = df_segments[df_segments['road_name'] == road_name]
        
        # Get center of road
        cx = road_df['lon_center'].mean()
        cy = road_df['lat_center'].mean()
        
        # Get custom offset and rotation
        config = label_positions.get(road_name, {'offset': (0, 0.002), 'rotation': 0})
        cx += config['offset'][0]
        cy += config['offset'][1]
        rotation = config['rotation']
        
        # Place label with nice styling
        ax.text(cx, cy, road_name,
               fontsize=11, fontweight='bold', fontstyle='italic',
               color='#1a1a1a', rotation=rotation,
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.4',
                        facecolor='white', edgecolor='#666666',
                        alpha=0.95, linewidth=1.5),
               zorder=30)
    
    # ================================================================
    # 7. COLORBAR
    # ================================================================
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=25, pad=0.02)
    cbar.set_label('Vitesse du trafic (km/h)', fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Speed annotations on colorbar
    cbar.ax.text(1.4, 0.08, 'Congestion', fontsize=10, transform=cbar.ax.transAxes, 
                va='center', color='#8B0000', fontweight='bold')
    cbar.ax.text(1.4, 0.92, 'Fluide', fontsize=10, transform=cbar.ax.transAxes, 
                va='center', color='#00AA44', fontweight='bold')
    
    # ================================================================
    # 8. LEGEND
    # ================================================================
    legend_elements = [
        Line2D([0], [0], marker='>', color='w', label='Point d\'entree',
               markerfacecolor='#27ae60', markeredgecolor='white',
               markersize=14, markeredgewidth=2),
        Line2D([0], [0], marker='s', color='w', label='Point de sortie',
               markerfacecolor='#c0392b', markeredgecolor='white',
               markersize=12, markeredgewidth=2),
        Line2D([0], [0], marker='s', color='w', label='Feu de signalisation',
               markerfacecolor='#2c3e50', markeredgecolor='#1a252f',
               markersize=12, markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', label='Intersection',
               markerfacecolor='white', markeredgecolor='#555555',
               markersize=10, markeredgewidth=1.5),
        Line2D([0], [0], color='#FF8800', linewidth=6, label='Segment routier'),
    ]
    
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10,
             framealpha=0.95, edgecolor='#888888', fancybox=True,
             title='Legende', title_fontsize=11)
    
    # ================================================================
    # 9. TITLES AND LABELS
    # ================================================================
    ax.set_title('Reseau Routier du Corridor Victoria Island\nLagos, Nigeria',
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.3, color='#aaaaaa')
    ax.tick_params(axis='both', labelsize=10)
    
    # Auto-scale to data
    ax.autoscale_view()
    
    # Add small margin
    lon_min, lon_max = ax.get_xlim()
    lat_min, lat_max = ax.get_ylim()
    margin = 0.003
    ax.set_xlim(lon_min - margin, lon_max + margin)
    ax.set_ylim(lat_min - margin, lat_max + margin)
    
    # Equal aspect for geographic data
    ax.set_aspect('equal', adjustable='box')
    
    # Statistics box
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
           transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5',
                    facecolor='white', edgecolor='#888888',
                    alpha=0.95))
    
    plt.tight_layout()
    
    # Save high quality
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"\n✅ Figure saved: {output_path}")
    
    return fig, ax


def main():
    print("=" * 70)
    print("🎯 THESIS NETWORK VISUALIZATION V3")
    print("   With VISIBLE road segments")
    print("=" * 70)
    
    # Load data
    df_segments, df_signals = load_enriched_data()
    
    # Generate traffic speeds
    print("\n🚗 Generating traffic speeds...")
    speeds = generate_traffic_speeds(df_segments)
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    output_path = OUTPUT_DIR / 'thesis_network_v3.png'
    fig, ax = create_thesis_figure(df_segments, df_signals, speeds, output_path)
    
    plt.close(fig)
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
