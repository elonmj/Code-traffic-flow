#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌍 GÉNÉRATEUR DE GÉOGRAPHIE DE CORRIDOR (REFERENCE ARCHITECTURE)
================================================================
Ce script extrait la topologie réelle d'un corridor pour la simulation ARZ/RL.

ARCHITECTURE MODULAIRE :
- Changez uniquement le bloc CONFIG pour cibler une nouvelle ville
- Le code génère automatiquement les CSV compatibles avec le modèle ARZ

POURQUOI CE CORRIDOR EST ADAPTÉ AU RL (vs Fixed-Time) :
=========================================================
Un corridor où le RL peut battre Fixed-Time doit avoir :

1. ASYMÉTRIE GÉOMÉTRIQUE : Plusieurs branches avec des flux différents
   → Vêdoko : Triangle à 3 branches (Stade, Toyota, Godomey)
   
2. CONFLITS DE MOUVEMENTS : Tournants à gauche/droite qui interagissent
   → Vêdoko : Carrefour central avec flux Nord-Sud ET Est-Ouest
   
3. VARIABILITÉ SPATIALE : Le goulot change de position selon l'heure
   → Vêdoko : Matin=flux entrant ville, Soir=flux sortant
   
4. DISTANCES IRRÉGULIÈRES : Onde verte difficile à synchroniser
   → Vêdoko : 3 carrefours à distances inégales (~500m, ~800m, ~600m)

Auteur: Thesis RL Traffic Control
Date: 2025-11
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from math import radians, cos, sin, asin, sqrt, atan2, degrees
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# =============================================================================
# 1. CONFIGURATIONS DES CORRIDORS (AJOUTEZ VOS VILLES ICI)
# =============================================================================

CORRIDORS = {
    # =========================================================================
    # COTONOU - VÊDOKO : "Le Triangle de la Mort"
    # =========================================================================
    "cotonou_vedoko": {
        "name": "Cotonou_Vedoko_Triangle",
        "description": "Triangle Vêdoko - Stade Amitié - Étoile Rouge (Cotonou, Bénin)",
        
        # Zone géographique (Bounding Box) - Élargie pour capturer le corridor complet
        "bbox": {
            'south': 6.365, 
            'west': 2.375, 
            'north': 6.395, 
            'east': 2.410
        },
        
        # Types de routes à extraire (OSM highway tags)
        "highway_types": ["trunk", "primary", "secondary"],
        
        # Filtre optionnel par nom de rue (regex)
        "road_filter": None,  # None = toutes les routes du type ci-dessus
        
        # ⚠️ FORÇAGE DES FEUX (CRITIQUE POUR L'AFRIQUE)
        # Coordonnées vérifiées via OpenStreetMap - utiliser les nœuds junction existants
        # OSM node 4133659794 = Carrefour Védoko (alt: Carrefour Toyota)
        "force_signals": [
            # Carrefour Védoko - Coordonnées exactes d'OSM node 4133659794
            {"name": "Carrefour Vêdoko (Toyota)", "lat": 6.3770938, "lon": 2.3898028},
            # Carrefour Stade Amitié - sur Avenue du Renouveau (intersection estimée)
            {"name": "Carrefour Stade Amitié",    "lat": 6.3862210, "lon": 2.3842480},
            # Carrefour Rue 150 / RNIE1 (vers Godomey)
            {"name": "Carrefour Rue 150",         "lat": 6.3753800, "lon": 2.4050420},
        ],
        
        # Rayon de matching pour les feux forcés (mètres) - augmenté car OSM peut varier
        "signal_match_radius": 100.0,
        
        # Valeurs par défaut si OSM ne les a pas
        "defaults": {
            "lanes": 2,
            "maxspeed": 50,  # km/h
            "surface": "asphalt"
        },
        
        # Pourquoi ce corridor est bon pour RL
        "rl_suitability": {
            "asymmetric_branches": True,    # Triangle = 3 branches asymétriques
            "conflicting_movements": True,  # Tournants à gauche au carrefour central
            "variable_bottleneck": True,    # Goulot change selon direction dominante
            "irregular_spacing": True,      # Distances inégales entre carrefours
            "score": "EXCELLENT"            # RL devrait battre Fixed-Time
        }
    },
    
    # =========================================================================
    # LAGOS - VICTORIA ISLAND (Configuration originale)
    # =========================================================================
    "lagos_victoria_island": {
        "name": "Lagos_Victoria_Island",
        "description": "Corridor Ahmadu Bello - Adeola Odeku (Victoria Island, Lagos)",
        
        "bbox": {
            'south': 6.42, 
            'west': 3.40, 
            'north': 6.46, 
            'east': 3.46
        },
        
        "highway_types": ["trunk", "primary", "secondary"],
        
        "road_filter": "Ahmadu Bello|Akin Adesola|Adeola Odeku|Saka Tinubu",
        
        "force_signals": [],  # Lagos a généralement de bonnes données OSM
        
        "signal_match_radius": 50.0,
        
        "defaults": {
            "lanes": 2,
            "maxspeed": 50,
            "surface": "asphalt"
        },
        
        "rl_suitability": {
            "asymmetric_branches": False,   # Corridor quasi-linéaire
            "conflicting_movements": False, # Peu de tournants modélisés
            "variable_bottleneck": False,   # Goulot stable
            "irregular_spacing": False,     # Distances régulières
            "score": "MODERATE"             # Fixed-Time peut être optimal
        }
    },
    
    # =========================================================================
    # TEMPLATE : Copiez ce bloc pour ajouter une nouvelle ville
    # =========================================================================
    "_template": {
        "name": "City_Corridor_Name",
        "description": "Description du corridor",
        
        "bbox": {
            'south': 0.0, 
            'west': 0.0, 
            'north': 0.0, 
            'east': 0.0
        },
        
        "highway_types": ["trunk", "primary", "secondary"],
        "road_filter": None,
        
        "force_signals": [
            # {"name": "Carrefour X", "lat": 0.0, "lon": 0.0},
        ],
        
        "signal_match_radius": 50.0,
        
        "defaults": {
            "lanes": 2,
            "maxspeed": 50,
            "surface": "asphalt"
        },
        
        "rl_suitability": {
            "asymmetric_branches": False,
            "conflicting_movements": False,
            "variable_bottleneck": False,
            "irregular_spacing": False,
            "score": "UNKNOWN"
        }
    }
}


# =============================================================================
# 2. FONCTIONS UTILITAIRES GÉOGRAPHIQUES
# =============================================================================

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcule la distance en mètres entre deux points GPS (formule de Haversine).
    
    Args:
        lat1, lon1: Coordonnées du point 1
        lat2, lon2: Coordonnées du point 2
        
    Returns:
        Distance en mètres
    """
    R = 6371000  # Rayon de la Terre en mètres
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcule l'angle de direction (azimut) entre deux points (0-360°).
    
    Args:
        lat1, lon1: Coordonnées du point de départ
        lat2, lon2: Coordonnées du point d'arrivée
        
    Returns:
        Bearing en degrés (0 = Nord, 90 = Est, 180 = Sud, 270 = Ouest)
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = degrees(atan2(x, y))
    return (bearing + 360) % 360


# =============================================================================
# 3. MOTEUR D'EXTRACTION OVERPASS API
# =============================================================================

class OverpassExtractor:
    """
    Extracteur de données OpenStreetMap via l'API Overpass.
    Robuste avec fallback sur plusieurs serveurs.
    """
    
    SERVERS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter",
    ]
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
    
    def fetch(self, query: str) -> Dict[str, Any]:
        """
        Exécute une requête Overpass avec fallback automatique.
        
        Args:
            query: Requête Overpass QL
            
        Returns:
            Données JSON de la réponse
            
        Raises:
            Exception: Si tous les serveurs échouent
        """
        for server in self.SERVERS:
            try:
                print(f"   📡 Tentative: {server.split('//')[1].split('/')[0]}...")
                response = requests.post(
                    server,
                    data={'data': query},
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                print(f"   ✅ Succès: {len(data.get('elements', []))} éléments reçus")
                return data
            except requests.exceptions.Timeout:
                print(f"   ⏱️ Timeout sur {server}")
            except requests.exceptions.RequestException as e:
                print(f"   ⚠️ Erreur: {str(e)[:50]}")
            except json.JSONDecodeError:
                print(f"   ⚠️ Réponse invalide (non-JSON)")
        
        raise Exception("Tous les serveurs Overpass ont échoué")
    
    def fetch_roads_and_signals(self, config: Dict) -> Dict[str, Any]:
        """
        Récupère les routes et feux de signalisation pour un corridor.
        
        Args:
            config: Configuration du corridor (bbox, highway_types, etc.)
            
        Returns:
            Données OSM brutes
        """
        bbox = config['bbox']
        bbox_str = f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}"
        
        # Construction de la requête
        highway_filter = "|".join(config['highway_types'])
        
        # Si un filtre de nom est spécifié
        name_filter = ""
        if config.get('road_filter'):
            name_filter = f'["name"~"{config["road_filter"]}"]'
        
        query = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["highway"~"{highway_filter}"]{name_filter}({bbox_str});
          node["highway"="traffic_signals"]({bbox_str});
        );
        out body;
        >;
        out skel qt;
        """
        
        return self.fetch(query)


# =============================================================================
# 4. PROCESSEUR DE RÉSEAU
# =============================================================================

class NetworkProcessor:
    """
    Traite les données OSM brutes pour créer un réseau de corridor.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.nodes: Dict[int, Dict] = {}
        self.ways: List[Dict] = []
        self.signals_osm: set = set()
        self.signals_forced: set = set()
    
    def parse_osm_data(self, data: Dict) -> None:
        """Parse les données OSM brutes."""
        for elem in data.get('elements', []):
            if elem['type'] == 'node':
                self.nodes[elem['id']] = {
                    'lat': elem['lat'],
                    'lon': elem['lon'],
                    'tags': elem.get('tags', {})
                }
                # Détecter les feux tagués dans OSM
                if elem.get('tags', {}).get('highway') == 'traffic_signals':
                    self.signals_osm.add(elem['id'])
            
            elif elem['type'] == 'way':
                self.ways.append(elem)
        
        print(f"   📊 Parsing: {len(self.nodes)} nœuds, {len(self.ways)} ways")
        print(f"   🚦 Feux OSM trouvés: {len(self.signals_osm)}")
    
    def apply_forced_signals(self) -> None:
        """Applique les feux forcés (vérité terrain)."""
        radius = self.config.get('signal_match_radius', 50.0)
        
        for forced in self.config.get('force_signals', []):
            # Trouver le nœud le plus proche
            best_node = None
            min_dist = float('inf')
            
            for node_id, node in self.nodes.items():
                dist = haversine(forced['lat'], forced['lon'], node['lat'], node['lon'])
                if dist < min_dist:
                    min_dist = dist
                    best_node = node_id
            
            if best_node and min_dist <= radius:
                if best_node not in self.signals_osm:
                    self.signals_forced.add(best_node)
                    print(f"   🔧 Feu FORCÉ: {forced['name']} (dist={min_dist:.1f}m)")
                else:
                    print(f"   ✅ Feu CONFIRMÉ: {forced['name']}")
            else:
                print(f"   ⚠️ IMPOSSIBLE de mapper: {forced['name']} (dist={min_dist:.1f}m > {radius}m)")
    
    @property
    def all_signals(self) -> set:
        """Tous les feux (OSM + forcés)."""
        return self.signals_osm | self.signals_forced
    
    def build_segments(self) -> pd.DataFrame:
        """
        Construit le DataFrame des segments du corridor.
        
        Returns:
            DataFrame avec tous les segments et leurs attributs
        """
        segments = []
        segment_id = 0
        defaults = self.config.get('defaults', {})
        
        for way in self.ways:
            way_nodes = way.get('nodes', [])
            tags = way.get('tags', {})
            
            road_name = tags.get('name', 'Unknown')
            highway_type = tags.get('highway', '')
            
            # Attributs avec fallback sur les defaults
            lanes = tags.get('lanes', str(defaults.get('lanes', 2)))
            maxspeed = tags.get('maxspeed', str(defaults.get('maxspeed', 50)))
            maxspeed = maxspeed.replace(' km/h', '').replace('kmh', '')
            surface = tags.get('surface', defaults.get('surface', 'asphalt'))
            oneway = tags.get('oneway', 'no')
            
            # Découper la way en segments [A→B], [B→C], ...
            for i in range(len(way_nodes) - 1):
                u_id = way_nodes[i]
                v_id = way_nodes[i + 1]
                
                if u_id not in self.nodes or v_id not in self.nodes:
                    continue
                
                u = self.nodes[u_id]
                v = self.nodes[v_id]
                
                # Calcul de la longueur
                length = haversine(u['lat'], u['lon'], v['lat'], v['lon'])
                
                # Ignorer les micro-segments (< 5m)
                if length < 5:
                    continue
                
                # Calcul du bearing
                bearing = calculate_bearing(u['lat'], u['lon'], v['lat'], v['lon'])
                
                # Détection des feux aux extrémités
                has_signal_start = 1 if u_id in self.all_signals else 0
                has_signal_end = 1 if v_id in self.all_signals else 0
                
                segments.append({
                    # Identifiants
                    'segment_id': segment_id,
                    'osm_way_id': way['id'],
                    
                    # Informations routières
                    'road_name': road_name,
                    'highway_type': highway_type,
                    'lanes': int(lanes) if lanes.isdigit() else 2,
                    'maxspeed_kmh': int(maxspeed) if maxspeed.isdigit() else 50,
                    'surface': surface,
                    'oneway': oneway,
                    
                    # Nœud de départ
                    'u_osm_id': u_id,
                    'u_lat': u['lat'],
                    'u_lon': u['lon'],
                    
                    # Nœud d'arrivée
                    'v_osm_id': v_id,
                    'v_lat': v['lat'],
                    'v_lon': v['lon'],
                    
                    # Géométrie
                    'length_m': round(length, 2),
                    'bearing_deg': round(bearing, 1),
                    
                    # Centre du segment
                    'center_lat': (u['lat'] + v['lat']) / 2,
                    'center_lon': (u['lon'] + v['lon']) / 2,
                    
                    # Feux de signalisation (CRITIQUE POUR RL)
                    'has_signal_start': has_signal_start,
                    'has_signal_end': has_signal_end,
                    
                    # Position dans la way
                    'segment_index': i,
                    'way_total_segments': len(way_nodes) - 1,
                    
                    # Géométrie WKT pour SIG
                    'geometry_wkt': f"LINESTRING({u['lon']} {u['lat']}, {v['lon']} {v['lat']})"
                })
                
                segment_id += 1
        
        return pd.DataFrame(segments)


# =============================================================================
# 5. GÉNÉRATEUR PRINCIPAL
# =============================================================================

class CorridorGenerator:
    """
    Générateur principal de corridors.
    Interface unifiée pour extraire et exporter les données.
    """
    
    def __init__(self, corridor_id: str):
        """
        Args:
            corridor_id: Clé du corridor dans CORRIDORS (ex: 'cotonou_vedoko')
        """
        if corridor_id not in CORRIDORS:
            available = [k for k in CORRIDORS.keys() if not k.startswith('_')]
            raise ValueError(f"Corridor inconnu: {corridor_id}. Disponibles: {available}")
        
        self.corridor_id = corridor_id
        self.config = CORRIDORS[corridor_id]
        self.df_segments: Optional[pd.DataFrame] = None
        self.metadata: Dict = {}
    
    def generate(self) -> pd.DataFrame:
        """
        Génère le corridor complet.
        
        Returns:
            DataFrame des segments
        """
        print("=" * 70)
        print(f"🌍 GÉNÉRATION DU CORRIDOR: {self.config['name']}")
        print(f"   {self.config['description']}")
        print("=" * 70)
        
        # 1. Extraction des données OSM
        print("\n📡 ÉTAPE 1: Extraction OpenStreetMap")
        extractor = OverpassExtractor(timeout=60)
        osm_data = extractor.fetch_roads_and_signals(self.config)
        
        # 2. Traitement du réseau
        print("\n🔧 ÉTAPE 2: Traitement du réseau")
        processor = NetworkProcessor(self.config)
        processor.parse_osm_data(osm_data)
        processor.apply_forced_signals()
        
        # 3. Construction des segments
        print("\n📊 ÉTAPE 3: Construction des segments")
        self.df_segments = processor.build_segments()
        
        # 4. Filtrage (segments > 10m)
        n_before = len(self.df_segments)
        self.df_segments = self.df_segments[self.df_segments['length_m'] >= 10].copy()
        self.df_segments = self.df_segments.reset_index(drop=True)
        self.df_segments['segment_id'] = range(len(self.df_segments))
        n_after = len(self.df_segments)
        print(f"   Filtrage: {n_before} → {n_after} segments (>= 10m)")
        
        # 5. Métadonnées
        self._build_metadata(processor)
        
        return self.df_segments
    
    def _build_metadata(self, processor: NetworkProcessor) -> None:
        """Construit les métadonnées du corridor."""
        df = self.df_segments
        
        self.metadata = {
            'corridor_id': self.corridor_id,
            'name': self.config['name'],
            'description': self.config['description'],
            'generated_at': datetime.now().isoformat(),
            
            'bbox': self.config['bbox'],
            
            'statistics': {
                'total_segments': len(df),
                'total_length_m': round(df['length_m'].sum(), 1),
                'total_length_km': round(df['length_m'].sum() / 1000, 2),
                'avg_segment_length_m': round(df['length_m'].mean(), 1),
                'signals_osm': len(processor.signals_osm),
                'signals_forced': len(processor.signals_forced),
                'signals_total': len(processor.all_signals),
                'segments_with_signal_end': int(df['has_signal_end'].sum()),
            },
            
            'roads': df['road_name'].value_counts().to_dict(),
            
            'rl_suitability': self.config.get('rl_suitability', {}),
            
            'config_used': {
                'highway_types': self.config['highway_types'],
                'road_filter': self.config.get('road_filter'),
                'force_signals': self.config.get('force_signals', []),
                'defaults': self.config.get('defaults', {})
            }
        }
    
    def save(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Sauvegarde le corridor en CSV et JSON.
        
        Args:
            output_dir: Répertoire de sortie (défaut: répertoire courant)
            
        Returns:
            Dict avec les chemins des fichiers créés
        """
        if self.df_segments is None:
            raise RuntimeError("Appelez generate() d'abord")
        
        output_dir = output_dir or Path(__file__).parent
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Nom de base
        base_name = f"corridor_{self.config['name'].lower()}"
        
        # CSV des segments
        csv_path = output_dir / f"{base_name}_segments.csv"
        self.df_segments.to_csv(csv_path, index=False, encoding='utf-8')
        
        # JSON des métadonnées
        json_path = output_dir / f"{base_name}_metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Fichiers sauvegardés:")
        print(f"   CSV:  {csv_path}")
        print(f"   JSON: {json_path}")
        
        return {'csv': csv_path, 'json': json_path}
    
    def print_summary(self) -> None:
        """Affiche un résumé du corridor généré."""
        if self.df_segments is None:
            print("⚠️ Aucun corridor généré. Appelez generate() d'abord.")
            return
        
        df = self.df_segments
        meta = self.metadata
        
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ DU CORRIDOR")
        print("=" * 70)
        
        print(f"\n🗺️  GÉOGRAPHIE:")
        print(f"   Nom: {meta['name']}")
        print(f"   Segments: {meta['statistics']['total_segments']}")
        print(f"   Longueur totale: {meta['statistics']['total_length_km']} km")
        print(f"   Longueur moyenne/segment: {meta['statistics']['avg_segment_length_m']} m")
        
        print(f"\n🚦 FEUX DE SIGNALISATION:")
        print(f"   Trouvés dans OSM: {meta['statistics']['signals_osm']}")
        print(f"   Ajoutés manuellement: {meta['statistics']['signals_forced']}")
        print(f"   Total: {meta['statistics']['signals_total']}")
        print(f"   Segments avec feu à la fin: {meta['statistics']['segments_with_signal_end']}")
        
        print(f"\n🛣️  ROUTES:")
        for road, count in meta['roads'].items():
            road_df = df[df['road_name'] == road]
            length = road_df['length_m'].sum()
            print(f"   • {road}: {count} segments, {length:.0f}m")
        
        print(f"\n🤖 ADÉQUATION POUR RL:")
        suit = meta.get('rl_suitability', {})
        print(f"   Branches asymétriques: {'✅' if suit.get('asymmetric_branches') else '❌'}")
        print(f"   Mouvements conflictuels: {'✅' if suit.get('conflicting_movements') else '❌'}")
        print(f"   Goulot variable: {'✅' if suit.get('variable_bottleneck') else '❌'}")
        print(f"   Espacement irrégulier: {'✅' if suit.get('irregular_spacing') else '❌'}")
        print(f"   → Score: {suit.get('score', 'N/A')}")
        
        print(f"\n📋 APERÇU DES DONNÉES (5 premiers segments):")
        cols = ['segment_id', 'road_name', 'length_m', 'has_signal_end', 'lanes']
        print(df[cols].head().to_string())
        
        print("\n" + "=" * 70)


# =============================================================================
# 6. POINT D'ENTRÉE
# =============================================================================

def list_corridors() -> None:
    """Affiche la liste des corridors disponibles."""
    print("\n🌍 CORRIDORS DISPONIBLES:")
    print("-" * 50)
    for key, config in CORRIDORS.items():
        if key.startswith('_'):
            continue
        score = config.get('rl_suitability', {}).get('score', 'N/A')
        print(f"  • {key}")
        print(f"    {config['description']}")
        print(f"    Score RL: {score}")
        print()


def main():
    """Point d'entrée principal."""
    import sys
    
    # Aide
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help', 'help']:
        print("""
🌍 GÉNÉRATEUR DE CORRIDOR - Usage:
    
    python corridor_generator.py <corridor_id>
    python corridor_generator.py list
    
Exemples:
    python corridor_generator.py cotonou_vedoko
    python corridor_generator.py lagos_victoria_island
    python corridor_generator.py list
        """)
        list_corridors()
        return
    
    # Lister les corridors
    if sys.argv[1] == 'list':
        list_corridors()
        return
    
    # Générer un corridor
    corridor_id = sys.argv[1]
    
    try:
        generator = CorridorGenerator(corridor_id)
        generator.generate()
        generator.save()
        generator.print_summary()
        
        print("\n✅ GÉNÉRATION TERMINÉE!")
        print("   Le fichier CSV peut être utilisé directement par le modèle ARZ.")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        raise


if __name__ == '__main__':
    main()
