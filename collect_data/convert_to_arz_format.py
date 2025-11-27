#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 CONVERTISSEUR DE FORMAT: corridor_generator → ARZ Model
============================================================

Ce script convertit le CSV généré par corridor_generator.py vers le format
attendu par le modèle ARZ (fichier_de_travail_corridor_utf8.csv).

FORMAT SOURCE (corridor_generator):
    segment_id, osm_way_id, road_name, highway_type, lanes, maxspeed_kmh,
    u_osm_id, u_lat, u_lon, v_osm_id, v_lat, v_lon, length_m, has_signal_end, ...

FORMAT CIBLE (ARZ model):
    u, v, name_clean, highway, length, oneway, lanes_manual, Rx_manual, maxspeed_manual_kmh

Auteur: Thesis RL Traffic Control
Date: 2025-11
"""

import pandas as pd
import sys
from pathlib import Path


def convert_to_arz_format(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Convertit un CSV de corridor_generator vers le format ARZ.
    
    Args:
        input_csv: Chemin vers le CSV source (corridor_*_segments.csv)
        output_csv: Chemin de sortie (optionnel, sinon génère automatiquement)
        
    Returns:
        DataFrame au format ARZ
    """
    print(f"📂 Chargement de {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"   {len(df)} segments chargés")
    
    # Conversion des colonnes
    arz_df = pd.DataFrame({
        'u': df['u_osm_id'],
        'v': df['v_osm_id'],
        'name_clean': df['road_name'],
        'highway': df['highway_type'],
        'length': df['length_m'],
        'oneway': df['oneway'].apply(lambda x: x == 'yes' if isinstance(x, str) else True),
        'lanes_manual': df['lanes'].apply(lambda x: x if x > 0 else ''),
        'Rx_manual': '',  # À remplir manuellement si nécessaire
        'maxspeed_manual_kmh': df['maxspeed_kmh'].apply(lambda x: x if x > 0 else ''),
        
        # Colonnes supplémentaires pour le RL
        'has_signal_end': df['has_signal_end'],  # ← CRITIQUE pour le RL
        
        # Coordonnées (utile pour la visualisation)
        'u_lat': df['u_lat'],
        'u_lon': df['u_lon'],
        'v_lat': df['v_lat'],
        'v_lon': df['v_lon'],
    })
    
    # Nettoyage
    arz_df['lanes_manual'] = arz_df['lanes_manual'].replace({0: '', '0': ''})
    
    # Générer le nom de sortie si non spécifié
    if output_csv is None:
        input_path = Path(input_csv)
        # Extraire le nom du corridor
        corridor_name = input_path.stem.replace('corridor_', '').replace('_segments', '')
        output_csv = input_path.parent / f"arz_topology_{corridor_name}.csv"
    
    # Sauvegarder
    arz_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ Sauvegardé: {output_csv}")
    
    # Statistiques
    print(f"\n📊 RÉSUMÉ:")
    print(f"   Segments: {len(arz_df)}")
    print(f"   Routes uniques: {arz_df['name_clean'].nunique()}")
    print(f"   Longueur totale: {arz_df['length'].sum() / 1000:.2f} km")
    print(f"   Segments avec feu: {arz_df['has_signal_end'].sum()}")
    
    print(f"\n📍 Routes:")
    for road in arz_df['name_clean'].unique()[:10]:
        road_df = arz_df[arz_df['name_clean'] == road]
        print(f"   • {road}: {len(road_df)} segments, {road_df['length'].sum():.0f}m")
    
    return arz_df


def main():
    if len(sys.argv) < 2:
        print("""
🔄 CONVERTISSEUR corridor_generator → ARZ

Usage:
    python convert_to_arz_format.py <corridor_segments.csv> [output.csv]
    
Exemples:
    python convert_to_arz_format.py corridor_cotonou_vedoko_triangle_segments.csv
    python convert_to_arz_format.py corridor_cotonou_vedoko_triangle_segments.csv arz_vedoko.csv
        """)
        return
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_to_arz_format(input_csv, output_csv)
    print("\n✅ Conversion terminée!")


if __name__ == '__main__':
    main()
