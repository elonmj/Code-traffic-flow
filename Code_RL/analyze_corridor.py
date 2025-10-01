"""
Analyse des données du corridor Victoria Island Lagos
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import os

# Add src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

def analyze_corridor_data():
    """Analyser les données du corridor"""
    print("🗺️  ANALYSE DU CORRIDOR VICTORIA ISLAND LAGOS")
    print("=" * 60)
    
    # Charger les données
    df = pd.read_csv("data/fichier_de_travail_corridor.csv")
    
    print(f"📊 Données générales :")
    print(f"   • {len(df)} segments de route")
    print(f"   • {len(df['u'].unique())} nœuds uniques (origine)")
    print(f"   • {len(df['v'].unique())} nœuds uniques (destination)")
    
    # Analyser les rues principales
    streets = df['name_clean'].value_counts()
    print(f"\n🛣️  Rues principales :")
    for street, count in streets.head(10).items():
        print(f"   • {street}: {count} segments")
    
    # Analyser les types de routes
    highway_types = df['highway'].value_counts()
    print(f"\n🚦 Types de routes :")
    for highway, count in highway_types.items():
        avg_length = df[df['highway'] == highway]['length'].mean()
        print(f"   • {highway}: {count} segments (longueur moy: {avg_length:.1f}m)")
    
    # Sens unique vs bidirectionnel
    oneway_count = df['oneway'].value_counts()
    print(f"\n➡️  Direction :")
    print(f"   • Sens unique: {oneway_count.get(True, 0)} segments")
    print(f"   • Bidirectionnel: {oneway_count.get(False, 0)} segments")
    
    # Intersections importantes (nœuds avec plus de connexions)
    nodes_as_source = df['u'].value_counts()
    nodes_as_dest = df['v'].value_counts()
    all_nodes = defaultdict(int)
    
    for node, count in nodes_as_source.items():
        all_nodes[node] += count
    for node, count in nodes_as_dest.items():
        all_nodes[node] += count
    
    # Trouver les intersections majeures
    major_intersections = sorted(all_nodes.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n🏢 Intersections majeures (par nombre de connexions) :")
    for node, connections in major_intersections[:5]:
        # Trouver les rues qui se rencontrent à ce nœud
        streets_at_node = set()
        streets_at_node.update(df[df['u'] == node]['name_clean'].tolist())
        streets_at_node.update(df[df['v'] == node]['name_clean'].tolist())
        streets_at_node.discard('')  # Enlever les noms vides
        
        print(f"   • Nœud {node}: {connections} connexions")
        if streets_at_node:
            print(f"     Rues: {', '.join(list(streets_at_node)[:3])}")
    
    return df

def identify_key_intersections(df):
    """Identifier les intersections clés pour le contrôle de signaux"""
    print(f"\n🎯 IDENTIFICATION DES INTERSECTIONS CLÉS")
    print("=" * 50)
    
    # Intersections Akin Adesola x Adeola Odeku (rues principales)
    akin_nodes = set(df[df['name_clean'] == 'Akin Adesola Street']['u'].tolist() + 
                    df[df['name_clean'] == 'Akin Adesola Street']['v'].tolist())
    
    adeola_nodes = set(df[df['name_clean'] == 'Adeola Odeku Street']['u'].tolist() + 
                      df[df['name_clean'] == 'Adeola Odeku Street']['v'].tolist())
    
    # Trouver les intersections entre ces deux rues
    intersections = akin_nodes.intersection(adeola_nodes)
    
    print(f"🚦 Intersections Akin Adesola x Adeola Odeku :")
    for node in list(intersections)[:3]:
        print(f"   • Nœud {node}")
        
        # Segments connectés
        connected_segments = df[(df['u'] == node) | (df['v'] == node)]
        streets = connected_segments['name_clean'].unique()
        highways = connected_segments['highway'].unique()
        
        print(f"     Rues: {', '.join([s for s in streets if s])}")
        print(f"     Types: {', '.join(highways)}")
    
    # Intersection avec Ahmadu Bello Way (route principale)
    ahmadu_nodes = set(df[df['name_clean'] == 'Ahmadu Bello Way']['u'].tolist() + 
                      df[df['name_clean'] == 'Ahmadu Bello Way']['v'].tolist())
    
    major_intersection = akin_nodes.intersection(ahmadu_nodes)
    
    if major_intersection:
        print(f"\n🏛️  Intersections majeures Akin Adesola x Ahmadu Bello :")
        for node in list(major_intersection)[:2]:
            print(f"   • Nœud {node}")
    
    return list(intersections)[:2] if intersections else []

def generate_realistic_branches(df, key_intersections):
    """Générer des branches réalistes basées sur la topologie"""
    print(f"\n🌳 GÉNÉRATION DES BRANCHES DE TRAFIC")
    print("=" * 45)
    
    branches = []
    
    # Pour chaque intersection clé, créer les branches d'entrée/sortie
    for i, intersection_node in enumerate(key_intersections[:2]):  # Limiter à 2 intersections
        print(f"\n📍 Intersection {i+1} - Nœud {intersection_node}:")
        
        # Trouver tous les segments connectés
        incoming = df[df['v'] == intersection_node]  # Segments arrivant
        outgoing = df[df['u'] == intersection_node]  # Segments sortant
        
        # Créer des branches pour chaque direction
        directions = ['north', 'south', 'east', 'west']
        
        for j, (_, segment) in enumerate(incoming.iterrows()):
            if j < 4:  # Limiter à 4 branches d'entrée par intersection
                branch_id = f"intersection_{i+1}_{directions[j]}_in"
                branches.append({
                    'id': branch_id,
                    'type': 'incoming',
                    'street': segment['name_clean'],
                    'highway_type': segment['highway'],
                    'length': segment['length'],
                    'node': intersection_node,
                    'direction': directions[j]
                })
                print(f"   📥 {branch_id}: {segment['name_clean']} ({segment['highway']})")
        
        for j, (_, segment) in enumerate(outgoing.iterrows()):
            if j < 4:  # Limiter à 4 branches de sortie par intersection
                branch_id = f"intersection_{i+1}_{directions[j]}_out"
                branches.append({
                    'id': branch_id,
                    'type': 'outgoing', 
                    'street': segment['name_clean'],
                    'highway_type': segment['highway'],
                    'length': segment['length'],
                    'node': intersection_node,
                    'direction': directions[j]
                })
                print(f"   📤 {branch_id}: {segment['name_clean']} ({segment['highway']})")
    
    print(f"\n✅ Total: {len(branches)} branches créées")
    return branches

def update_network_config(branches):
    """Mettre à jour la configuration réseau avec les vraies données"""
    print(f"\n🔧 MISE À JOUR DE LA CONFIGURATION RÉSEAU")
    print("=" * 50)
    
    # Créer la nouvelle configuration
    network_config = {
        'network': {
            'name': 'Victoria Island Lagos - Real Data',
            'description': 'Réseau basé sur les données réelles du corridor Lagos',
            'location': {
                'city': 'Lagos',
                'country': 'Nigeria',
                'area': 'Victoria Island'
            },
            'branches': []
        }
    }
    
    # Ajouter les branches avec paramètres réalistes
    for branch in branches:
        # Paramètres basés sur le type de route
        if branch['highway_type'] == 'primary':
            lanes = 3
            capacity = 1800  # véh/h/voie
            speed_limit = 50  # km/h
        elif branch['highway_type'] == 'secondary':
            lanes = 2
            capacity = 1500
            speed_limit = 40
        else:  # tertiary
            lanes = 1
            capacity = 1200
            speed_limit = 30
        
        branch_config = {
            'id': branch['id'],
            'name': f"{branch['street']} - {branch['direction']} ({branch['type']})",
            'type': branch['type'],
            'length': round(branch['length'], 1),
            'lanes': lanes,
            'capacity_per_lane': capacity,
            'speed_limit_kmh': speed_limit,
            'street_name': branch['street'],
            'highway_type': branch['highway_type']
        }
        
        network_config['network']['branches'].append(branch_config)
        
        print(f"   ✓ {branch['id']}: {lanes} voies, {speed_limit}km/h")
    
    # Sauvegarder la nouvelle configuration
    import yaml
    with open('configs/network_real.yaml', 'w') as f:
        yaml.dump(network_config, f, default_flow_style=False, indent=2)
    
    print(f"\n💾 Configuration sauvegardée dans 'configs/network_real.yaml'")
    
    return network_config

def main():
    """Analyse principale et adaptation"""
    
    # Analyser les données
    df = analyze_corridor_data()
    
    # Identifier les intersections clés
    key_intersections = identify_key_intersections(df)
    
    if not key_intersections:
        print("⚠️  Aucune intersection majeure trouvée, utilisation de nœuds principaux")
        # Utiliser les nœuds les plus connectés comme intersections
        nodes_count = defaultdict(int)
        for _, row in df.iterrows():
            nodes_count[row['u']] += 1
            nodes_count[row['v']] += 1
        key_intersections = sorted(nodes_count.items(), key=lambda x: x[1], reverse=True)[:2]
        key_intersections = [node for node, _ in key_intersections]
    
    # Générer les branches réalistes
    branches = generate_realistic_branches(df, key_intersections)
    
    # Mettre à jour la configuration réseau
    network_config = update_network_config(branches)
    
    print(f"\n🎉 ADAPTATION TERMINÉE !")
    print(f"   • Configuration réseau mise à jour avec {len(branches)} branches")
    print(f"   • Basée sur {len(key_intersections)} intersections clés")
    print(f"   • Fichier: configs/network_real.yaml")
    
    print(f"\n📖 Prochaines étapes :")
    print(f"   • Tester: python demo.py 1")
    print(f"   • Entraîner: python train.py --use-mock --timesteps 5000")
    print(f"   • Utiliser réseau réel dans les configs")

if __name__ == "__main__":
    main()
