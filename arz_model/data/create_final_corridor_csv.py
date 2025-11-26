#!/usr/bin/env python3
"""
📊 CREATE FINAL CORRIDOR CSV
=============================

Creates a CLEAN, READY-TO-USE CSV with all corridor data:
- Only segments on our 4 main roads
- Only traffic signals that are ON our segments
- Pre-computed fields for visualization
- No external noise from OSM bounding box

This CSV is the SINGLE SOURCE OF TRUTH for all visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent


def main():
    print("=" * 60)
    print("CREATING FINAL CORRIDOR CSV")
    print("=" * 60)
    
    # Load raw enriched data
    df = pd.read_csv(DATA_DIR / 'corridor_segments_enriched.csv')
    print(f"\nLoaded {len(df)} segments")
    
    # ================================================================
    # 1. IDENTIFY SIGNALS ON OUR SEGMENTS
    # ================================================================
    
    # Collect all signal node positions from our segments
    signal_nodes_from = df[df['has_signal_from'] == 1][['lon_from', 'lat_from', 'node_from_osm_id']].copy()
    signal_nodes_from.columns = ['lon', 'lat', 'node_osm_id']
    
    signal_nodes_to = df[df['has_signal_to'] == 1][['lon_to', 'lat_to', 'node_to_osm_id']].copy()
    signal_nodes_to.columns = ['lon', 'lat', 'node_osm_id']
    
    # Combine and deduplicate
    all_signals = pd.concat([signal_nodes_from, signal_nodes_to]).drop_duplicates(subset=['node_osm_id'])
    all_signals['is_signal'] = 1
    
    print(f"Traffic signals ON corridor: {len(all_signals)}")
    
    # ================================================================
    # 2. IDENTIFY ENTRY/EXIT POINTS
    # ================================================================
    
    entries = df[df['is_way_entry'] == 1][['lon_from', 'lat_from', 'road_name', 'node_from_osm_id']].copy()
    entries.columns = ['lon', 'lat', 'road_name', 'node_osm_id']
    entries['point_type'] = 'entry'
    
    exits = df[df['is_way_exit'] == 1][['lon_to', 'lat_to', 'road_name', 'node_to_osm_id']].copy()
    exits.columns = ['lon', 'lat', 'road_name', 'node_osm_id']
    exits['point_type'] = 'exit'
    
    access_points = pd.concat([entries, exits]).drop_duplicates(subset=['node_osm_id'])
    
    print(f"Entry points: {len(entries)}")
    print(f"Exit points: {len(exits)}")
    
    # ================================================================
    # 3. COLLECT ALL UNIQUE NODES
    # ================================================================
    
    nodes_from = df[['lon_from', 'lat_from', 'node_from_osm_id', 'road_name']].copy()
    nodes_from.columns = ['lon', 'lat', 'node_osm_id', 'road_name']
    
    nodes_to = df[['lon_to', 'lat_to', 'node_to_osm_id', 'road_name']].copy()
    nodes_to.columns = ['lon', 'lat', 'node_osm_id', 'road_name']
    
    all_nodes = pd.concat([nodes_from, nodes_to]).drop_duplicates(subset=['node_osm_id'])
    
    # Merge signal info
    all_nodes = all_nodes.merge(
        all_signals[['node_osm_id', 'is_signal']], 
        on='node_osm_id', 
        how='left'
    )
    all_nodes['is_signal'] = all_nodes['is_signal'].fillna(0).astype(int)
    
    # Merge access point info
    all_nodes = all_nodes.merge(
        access_points[['node_osm_id', 'point_type']], 
        on='node_osm_id', 
        how='left'
    )
    all_nodes['point_type'] = all_nodes['point_type'].fillna('intersection')
    
    print(f"Total unique nodes: {len(all_nodes)}")
    
    # ================================================================
    # 4. CREATE CLEAN SEGMENTS TABLE
    # ================================================================
    
    segments_clean = df[[
        'segment_id', 'road_name', 'highway_type',
        'lon_from', 'lat_from', 'lon_to', 'lat_to',
        'lon_center', 'lat_center',
        'length_m', 'bearing_deg',
        'lanes', 'maxspeed_kmh', 'oneway',
        'has_signal_from', 'has_signal_to',
        'is_way_entry', 'is_way_exit'
    ]].copy()
    
    # Add road importance for line width
    road_importance = {
        'Ahmadu Bello Way': 1,
        'Akin Adesola Street': 2,
        'Adeola Odeku Street': 3,
        'Saka Tinubu Street': 4
    }
    segments_clean['road_importance'] = segments_clean['road_name'].map(road_importance)
    
    # ================================================================
    # 5. CREATE SIGNALS TABLE (ONLY ON CORRIDOR)
    # ================================================================
    
    signals_clean = all_signals.copy()
    signals_clean = signals_clean.reset_index(drop=True)
    signals_clean['signal_id'] = range(1, len(signals_clean) + 1)
    signals_clean = signals_clean[['signal_id', 'node_osm_id', 'lon', 'lat']]
    
    # ================================================================
    # 6. CREATE NODES TABLE
    # ================================================================
    
    nodes_clean = all_nodes.copy()
    nodes_clean = nodes_clean.reset_index(drop=True)
    nodes_clean['node_id'] = range(1, len(nodes_clean) + 1)
    nodes_clean = nodes_clean[['node_id', 'node_osm_id', 'lon', 'lat', 'road_name', 'is_signal', 'point_type']]
    
    # ================================================================
    # 7. SAVE ALL TABLES
    # ================================================================
    
    # Main segments file
    segments_path = DATA_DIR / 'corridor_final_segments.csv'
    segments_clean.to_csv(segments_path, index=False)
    print(f"\n✅ Saved: {segments_path.name} ({len(segments_clean)} segments)")
    
    # Signals file (ONLY on corridor)
    signals_path = DATA_DIR / 'corridor_final_signals.csv'
    signals_clean.to_csv(signals_path, index=False)
    print(f"✅ Saved: {signals_path.name} ({len(signals_clean)} signals)")
    
    # Nodes file
    nodes_path = DATA_DIR / 'corridor_final_nodes.csv'
    nodes_clean.to_csv(nodes_path, index=False)
    print(f"✅ Saved: {nodes_path.name} ({len(nodes_clean)} nodes)")
    
    # ================================================================
    # 8. SUMMARY STATISTICS
    # ================================================================
    
    print("\n" + "=" * 60)
    print("FINAL CORRIDOR DATA SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 SEGMENTS:")
    for road in segments_clean['road_name'].unique():
        road_df = segments_clean[segments_clean['road_name'] == road]
        length_km = road_df['length_m'].sum() / 1000
        print(f"   {road}: {len(road_df)} segments, {length_km:.2f} km")
    
    total_km = segments_clean['length_m'].sum() / 1000
    print(f"   TOTAL: {len(segments_clean)} segments, {total_km:.2f} km")
    
    print(f"\n🚦 TRAFFIC SIGNALS ON CORRIDOR: {len(signals_clean)}")
    
    print(f"\n📍 NODES:")
    print(f"   Total: {len(nodes_clean)}")
    print(f"   With signals: {nodes_clean['is_signal'].sum()}")
    print(f"   Entry points: {len(nodes_clean[nodes_clean['point_type'] == 'entry'])}")
    print(f"   Exit points: {len(nodes_clean[nodes_clean['point_type'] == 'exit'])}")
    print(f"   Regular intersections: {len(nodes_clean[nodes_clean['point_type'] == 'intersection'])}")
    
    print(f"\n📐 GEOGRAPHIC BOUNDS:")
    print(f"   Longitude: {segments_clean['lon_from'].min():.6f} to {segments_clean['lon_to'].max():.6f}")
    print(f"   Latitude: {segments_clean['lat_from'].min():.6f} to {segments_clean['lat_to'].max():.6f}")
    
    # ================================================================
    # 9. SHOW SAMPLE DATA
    # ================================================================
    
    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)
    
    print("\n📋 SEGMENTS (first 3):")
    print(segments_clean[['segment_id', 'road_name', 'lon_from', 'lat_from', 'lon_to', 'lat_to', 'length_m']].head(3).to_string(index=False))
    
    print("\n🚦 SIGNALS (first 5):")
    print(signals_clean.head().to_string(index=False))
    
    print("\n📍 NODES with signals:")
    print(nodes_clean[nodes_clean['is_signal'] == 1].head().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("DONE! Use these CSV files for all visualizations.")
    print("=" * 60)


if __name__ == '__main__':
    main()
