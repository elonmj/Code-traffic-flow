#!/usr/bin/env python3
"""
OSM Metadata Enrichment Tool for Road Network Corridors
========================================================

Enrichit un fichier Excel de segments routiers (u, v, ...) avec des métadonnées
OpenStreetMap pour chaque nœud : feux de signalisation, types de jonction,
phases de signalisation, croisements, etc.

Usage:
    python enrich_osm_metadata.py <input_file.xlsx> [output_file.xlsx]

Arguments:
    input_file.xlsx   : Fichier Excel avec colonnes 'u' et 'v' (OSM node IDs)
    output_file.xlsx  : Fichier de sortie (par défaut: input_enriched.xlsx)

Format d'entrée requis:
    - Colonnes obligatoires: 'u', 'v' (node IDs entiers ou négatifs)
    - Colonnes optionnelles: name_clean, highway, length, oneway, etc.

Format de sortie:
    - Colonnes originales +
    - Pour chaque nœud (u/v):
        * {prefix}_has_signal        : bool - Présence de feu tricolore
        * {prefix}_junction_type     : str  - Type de jonction (si défini)
        * {prefix}_signal_tags       : str  - Toutes les tags traffic_signals:*
        * {prefix}_signal_phases     : str  - Nombre/config de phases
        * {prefix}_signal_controller : str  - Type de contrôleur
        * {prefix}_crossing_type     : str  - Type de passage piéton
        * {prefix}_traffic_calming   : str  - Dispositifs d'apaisement
        * {prefix}_highway_tag       : str  - Tag highway du nœud
        * {prefix}_lat               : float - Latitude
        * {prefix}_lon               : float - Longitude

Auteur: Alibi Traffic Analysis Team
Date: 2025-11-18
"""

import sys
import argparse
from pathlib import Path
import time
import pandas as pd
import requests


def fetch_osm_nodes(node_ids, chunk_size=50, timeout=120, verbose=True):
    """
    Récupère les métadonnées OSM pour une liste de node IDs via Overpass API.
    
    Args:
        node_ids: Liste de node IDs OSM (entiers positifs)
        chunk_size: Taille des batches pour éviter timeouts
        timeout: Timeout pour chaque requête HTTP
        verbose: Afficher progression
    
    Returns:
        dict: {node_id: {'lat': float, 'lon': float, 'tags': dict}}
    """
    url = "https://overpass-api.de/api/interpreter"
    node_data = {}
    
    # Filtrer les IDs positifs seulement
    valid_ids = sorted([n for n in node_ids if n > 0])
    
    if verbose:
        print(f"📡 Récupération des métadonnées pour {len(valid_ids)} nœuds OSM...")
    
    for i in range(0, len(valid_ids), chunk_size):
        chunk = valid_ids[i:i+chunk_size]
        ids_str = ",".join(str(n) for n in chunk)
        query = f"[out:json][timeout:{timeout}];node(id:{ids_str});out body;"
        
        try:
            resp = requests.get(url, params={'data': query}, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            
            for element in data.get('elements', []):
                node_data[element['id']] = {
                    'lat': element.get('lat'),
                    'lon': element.get('lon'),
                    'tags': element.get('tags', {})
                }
            
            if verbose and (i // chunk_size + 1) % 5 == 0:
                print(f"  ✓ {min(i + chunk_size, len(valid_ids))}/{len(valid_ids)} nœuds traités")
            
            # Pause courtoisie pour Overpass API
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  Erreur requête batch {i//chunk_size + 1}: {e}")
            continue
    
    if verbose:
        missing = set(valid_ids) - set(node_data.keys())
        if missing:
            print(f"  ⚠️  {len(missing)} nœuds non trouvés dans OSM (supprimés ou invisibles)")
        print(f"✅ Métadonnées récupérées pour {len(node_data)} nœuds\n")
    
    return node_data


def extract_node_metadata(node_id, node_data):
    """
    Extrait les métadonnées structurées pour un nœud OSM.
    
    Args:
        node_id: ID du nœud (peut être négatif = virtuel)
        node_data: dict retourné par fetch_osm_nodes
    
    Returns:
        tuple: (has_signal, junction_type, signal_tags, signal_phases,
                signal_controller, crossing_type, traffic_calming,
                highway_tag, lat, lon)
    """
    # Nœuds virtuels (négatifs) ou absents
    if node_id <= 0 or int(node_id) not in node_data:
        return (False, None, None, None, None, None, None, None, None, None)
    
    info = node_data[int(node_id)]
    tags = info.get('tags', {})
    
    # Détection feu tricolore
    has_signal = (
        tags.get('highway') == 'traffic_signals' or
        bool(tags.get('traffic_signals')) or
        any(k.startswith('traffic_signals:') for k in tags)
    )
    
    # Type de jonction
    junction = tags.get('junction')
    
    # Tags de signalisation (format clé=valeur)
    signal_tags_list = [
        f"{k}={v}" for k, v in tags.items()
        if k.startswith('traffic_signals')
    ]
    signal_tags = "; ".join(signal_tags_list) if signal_tags_list else None
    
    # Phases de signalisation
    signal_phases = (
        tags.get('traffic_signals:phases') or
        tags.get('traffic_signals:multiphase')
    )
    
    # Type de contrôleur
    signal_controller = (
        tags.get('traffic_signals:direction') or
        tags.get('traffic_signals:control')
    )
    
    # Type de passage piéton
    crossing = tags.get('crossing')
    
    # Dispositifs d'apaisement
    traffic_calming = tags.get('traffic_calming')
    
    # Tag highway du nœud
    highway_tag = tags.get('highway')
    
    return (
        has_signal,
        junction,
        signal_tags,
        signal_phases,
        signal_controller,
        crossing,
        traffic_calming,
        highway_tag,
        info.get('lat'),
        info.get('lon')
    )


def enrich_corridor_data(input_path, output_path=None, verbose=True):
    """
    Pipeline complet d'enrichissement d'un fichier corridor.
    
    Args:
        input_path: Chemin vers fichier Excel d'entrée
        output_path: Chemin vers fichier Excel de sortie (auto si None)
        verbose: Afficher progression
    
    Returns:
        Path: Chemin du fichier enrichi
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_enriched{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🚀 ENRICHISSEMENT OSM METADATA")
        print(f"{'='*70}")
        print(f"📂 Fichier d'entrée : {input_path}")
        print(f"📂 Fichier de sortie: {output_path}\n")
    
    # Chargement données
    if verbose:
        print("📖 Chargement du fichier corridor...")
    
    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        print(f"❌ Erreur lecture fichier: {e}")
        sys.exit(1)
    
    # Validation colonnes
    required_cols = ['u', 'v']
    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"❌ Colonnes manquantes: {missing}")
        print(f"   Colonnes présentes: {df.columns.tolist()}")
        sys.exit(1)
    
    if verbose:
        print(f"✅ {len(df)} segments chargés")
        print(f"   Colonnes: {df.columns.tolist()}\n")
    
    # Extraction node IDs uniques
    raw_nodes = set(df['u'].dropna().astype(int)).union(set(df['v'].dropna().astype(int)))
    node_ids = sorted([n for n in raw_nodes if n > 0])
    
    if verbose:
        print(f"🔍 Nœuds uniques détectés: {len(raw_nodes)} total, {len(node_ids)} positifs (OSM)")
        negative_count = len(raw_nodes) - len(node_ids)
        if negative_count > 0:
            print(f"   ℹ️  {negative_count} nœuds virtuels (ID < 0) seront marqués sans métadonnées\n")
    
    # Récupération métadonnées OSM
    node_data = fetch_osm_nodes(node_ids, verbose=verbose)
    
    # Extraction métadonnées pour chaque nœud
    if verbose:
        print("🔧 Extraction des métadonnées structurées...")
    
    u_metadata = df['u'].apply(lambda x: extract_node_metadata(int(x), node_data))
    v_metadata = df['v'].apply(lambda x: extract_node_metadata(int(x), node_data))
    
    # Unpacking en colonnes
    u_cols = list(zip(*u_metadata))
    v_cols = list(zip(*v_metadata))
    
    col_names = [
        'has_signal', 'junction_type', 'signal_tags', 'signal_phases',
        'signal_controller', 'crossing_type', 'traffic_calming',
        'highway_tag', 'lat', 'lon'
    ]
    
    for i, name in enumerate(col_names):
        df[f'u_{name}'] = u_cols[i]
        df[f'v_{name}'] = v_cols[i]
    
    if verbose:
        print(f"✅ {len(col_names)*2} nouvelles colonnes ajoutées\n")
    
    # Statistiques
    if verbose:
        u_signals = df['u_has_signal'].sum()
        v_signals = df['v_has_signal'].sum()
        unique_signal_nodes = len(
            set(df.loc[df['u_has_signal'], 'u']).union(
                set(df.loc[df['v_has_signal'], 'v'])
            )
        )
        
        print(f"📊 Statistiques:")
        print(f"   • Segments avec feux à l'origine (u): {u_signals}")
        print(f"   • Segments avec feux à la destination (v): {v_signals}")
        print(f"   • Nœuds uniques avec feux: {unique_signal_nodes}")
        
        # Junctions
        u_junctions = df['u_junction_type'].dropna().nunique()
        v_junctions = df['v_junction_type'].dropna().nunique()
        if u_junctions + v_junctions > 0:
            print(f"   • Types de jonctions trouvés: {u_junctions + v_junctions}")
        
        # Phases
        u_phases = df['u_signal_phases'].dropna().count()
        v_phases = df['v_signal_phases'].dropna().count()
        if u_phases + v_phases > 0:
            print(f"   • Nœuds avec info phases: {u_phases + v_phases}")
        print()
    
    # Sauvegarde
    if verbose:
        print(f"💾 Sauvegarde du fichier enrichi...")
    
    try:
        df.to_excel(output_path, index=False)
        if verbose:
            print(f"✅ Fichier sauvegardé: {output_path}")
            print(f"   Taille: {output_path.stat().st_size / 1024:.1f} KB\n")
    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")
        sys.exit(1)
    
    if verbose:
        print(f"{'='*70}")
        print(f"✅ ENRICHISSEMENT TERMINÉ AVEC SUCCÈS")
        print(f"{'='*70}\n")
    
    return output_path


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Enrichit un fichier corridor avec métadonnées OSM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    python enrich_osm_metadata.py input/corridor.xlsx
    python enrich_osm_metadata.py input/data.xlsx output/enriched.xlsx
    python enrich_osm_metadata.py corridor.xlsx --quiet

Format d'entrée requis:
    • Colonnes 'u' et 'v' avec OSM node IDs (entiers)
    • Format Excel (.xlsx)
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help="Fichier Excel d'entrée (avec colonnes u, v)"
    )
    
    parser.add_argument(
        'output_file',
        type=str,
        nargs='?',
        default=None,
        help="Fichier Excel de sortie (optionnel, auto-nommé si absent)"
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Mode silencieux (pas de logs)"
    )
    
    args = parser.parse_args()
    
    # Validation existence fichier
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Fichier introuvable: {input_path}")
        sys.exit(1)
    
    # Exécution
    try:
        enrich_corridor_data(
            input_path,
            args.output_file,
            verbose=not args.quiet
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption utilisateur - arrêt du programme")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
