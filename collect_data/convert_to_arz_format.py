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
import networkx as nx
import sys
from pathlib import Path


def filter_largest_connected_component(df: pd.DataFrame, keep_signal_components: bool = True) -> pd.DataFrame:
    """
    Filtre le DataFrame pour ne garder que les composants connexes importants.
    
    Stratégie:
    - Si keep_signal_components=True: garde tous les composants contenant des feux
      (car ce sont les routes importantes pour le contrôle RL)
    - Sinon: garde uniquement le plus grand composant connexe
    
    Args:
        df: DataFrame avec colonnes 'u', 'v', et optionnellement 'has_signal_end'
        keep_signal_components: Si True, garde tous les composants avec feux
        
    Returns:
        DataFrame filtré
    """
    # Construire le graphe (non-dirigé pour trouver les composantes)
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(str(row['u']), str(row['v']))
    
    # Trouver les composantes connexes
    components = list(nx.connected_components(G))
    n_components = len(components)
    
    if n_components == 1:
        print(f"   ✅ Réseau entièrement connecté (1 composante)")
        return df
    
    # Identifier les nœuds signalisés
    signal_nodes = set()
    if 'has_signal_end' in df.columns:
        signal_nodes = set(df[df['has_signal_end'] == 1]['v'].astype(str).unique())
    
    # Déterminer quels composants garder
    if keep_signal_components and signal_nodes:
        # Garder tous les composants qui contiennent au moins un feu
        components_to_keep = []
        for comp in components:
            has_signal = any(n in signal_nodes for n in comp)
            if has_signal:
                components_to_keep.append(comp)
        
        if not components_to_keep:
            # Fallback: si aucun composant n'a de feux, garder le plus grand
            components_to_keep = [max(components, key=len)]
            print(f"   ⚠️  Aucun feu détecté - fallback sur le plus grand composant")
        
        kept_nodes = set()
        for comp in components_to_keep:
            kept_nodes.update(str(n) for n in comp)
        
        print(f"   🚦 {len(components_to_keep)} composants avec feux de signalisation conservés")
    else:
        # Mode classique: garder uniquement le plus grand composant
        largest_component = max(components, key=len)
        kept_nodes = set(str(n) for n in largest_component)
        components_to_keep = [largest_component]
    
    # Statistiques avant filtrage
    total_nodes = G.number_of_nodes()
    total_edges = len(df)
    
    # Filtrer les segments qui appartiennent aux composants gardés
    mask = df.apply(lambda row: str(row['u']) in kept_nodes and str(row['v']) in kept_nodes, axis=1)
    df_filtered = df[mask].copy()
    
    # Statistiques après filtrage
    filtered_nodes = len(kept_nodes)
    filtered_edges = len(df_filtered)
    removed_edges = total_edges - filtered_edges
    removed_nodes = total_nodes - filtered_nodes
    n_removed_components = n_components - len(components_to_keep)
    
    print(f"   ⚠️  {n_components} composantes détectées")
    print(f"   📉 Supprimé: {removed_edges} segments ({removed_nodes} nœuds) de {n_removed_components} composants sans feux")
    print(f"   ✅ Conservé: {filtered_edges} segments ({filtered_nodes} nœuds) dans {len(components_to_keep)} composant(s)")
    
    return df_filtered


from typing import Optional


def convert_to_arz_format(input_csv: str, output_csv: Optional[str] = None, keep_largest_only: bool = True) -> pd.DataFrame:
    """
    Convertit un CSV de corridor_generator vers le format ARZ.
    
    Args:
        input_csv: Chemin vers le CSV source (corridor_*_segments.csv)
        output_csv: Chemin de sortie (optionnel, sinon génère automatiquement)
        keep_largest_only: Si True, ne garde que le plus grand composant connexe (recommandé)
        
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
    
    # ========== FILTRAGE DU PLUS GRAND COMPOSANT CONNEXE ==========
    if keep_largest_only:
        print("\n🔗 Analyse de la connectivité du réseau...")
        arz_df = filter_largest_connected_component(arz_df)
    
    # Générer le nom de sortie si non spécifié
    if output_csv is None:
        input_path = Path(input_csv)
        # Extraire le nom du corridor
        corridor_name = input_path.stem.replace('corridor_', '').replace('_segments', '')
        output_csv = str(input_path.parent / f"arz_topology_{corridor_name}.csv")
    
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
