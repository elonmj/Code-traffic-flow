#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse rapide d'un corridor généré.
"""

import pandas as pd
import sys

def analyze(csv_path: str):
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print("🚦 SEGMENTS AVEC FEU À LA FIN:")
    print("=" * 80)
    
    feux = df[df['has_signal_end'] == 1]
    for _, row in feux.iterrows():
        road = row['road_name'][:35].ljust(35)
        print(f"  Segment {row['segment_id']:3d}: {road} | {row['length_m']:6.1f}m | fin: ({row['v_lat']:.5f}, {row['v_lon']:.5f})")
    
    print(f"\n  → Total: {len(feux)} segments contrôlables par le RL")
    
    print("\n" + "=" * 80)
    print("📊 STATISTIQUES PAR ROUTE:")
    print("=" * 80)
    
    stats = df.groupby('road_name').agg({
        'segment_id': 'count',
        'length_m': 'sum',
        'has_signal_end': 'sum'
    }).rename(columns={
        'segment_id': 'segments', 
        'length_m': 'longueur_m', 
        'has_signal_end': 'feux'
    }).sort_values('segments', ascending=False)
    
    print(stats.head(15).to_string())
    
    print("\n" + "=" * 80)
    print("🗺️ COUVERTURE GÉOGRAPHIQUE:")
    print("=" * 80)
    print(f"  Latitude:  {df['u_lat'].min():.5f} → {df['u_lat'].max():.5f}")
    print(f"  Longitude: {df['u_lon'].min():.5f} → {df['u_lon'].max():.5f}")
    print(f"  Longueur totale: {df['length_m'].sum() / 1000:.2f} km")
    print(f"  Segments: {len(df)}")
    

if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'corridor_cotonou_vedoko_triangle_segments.csv'
    analyze(csv_path)
