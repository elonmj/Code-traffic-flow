# 🌍 Système de Collecte de Données de Corridors

Ce dossier contient les outils pour extraire et préparer les données géographiques de corridors routiers pour la simulation ARZ et l'entraînement RL.

## 📂 Structure

```
collect_data/
├── corridor_generator.py      # Générateur principal (modifie les configs ici)
├── analyze_corridor.py        # Utilitaire d'analyse rapide
├── corridor_*_segments.csv    # Données de segments (sortie)
└── corridor_*_metadata.json   # Métadonnées du corridor (sortie)
```

## 🚀 Utilisation

### Générer un corridor

```bash
# Lister les corridors disponibles
python corridor_generator.py list

# Générer le corridor Cotonou Vêdoko
python corridor_generator.py cotonou_vedoko

# Générer un autre corridor
python corridor_generator.py lagos_victoria_island
```

### Analyser un corridor généré

```bash
python analyze_corridor.py corridor_cotonou_vedoko_triangle_segments.csv
```

## ➕ Ajouter un Nouveau Corridor

1. Ouvrir `corridor_generator.py`
2. Localiser le dictionnaire `CORRIDORS`
3. Copier le template `_template` et le renommer
4. Remplir les champs :
   - `bbox` : Bounding box (south, west, north, east)
   - `force_signals` : Coordonnées des feux (vérifiées via OpenStreetMap)
   - `rl_suitability` : Évaluation de l'adéquation pour le RL

### Exemple de Configuration

```python
"ma_ville_corridor": {
    "name": "MaVille_MonCorridor",
    "description": "Description du corridor",
    
    "bbox": {
        'south': 5.0, 'west': 1.0, 
        'north': 5.5, 'east': 1.5
    },
    
    "highway_types": ["trunk", "primary", "secondary"],
    
    "force_signals": [
        {"name": "Carrefour Central", "lat": 5.25, "lon": 1.25},
    ],
    
    "signal_match_radius": 100.0,
    
    "defaults": {"lanes": 2, "maxspeed": 50, "surface": "asphalt"},
    
    "rl_suitability": {
        "asymmetric_branches": True,
        "conflicting_movements": True,
        "variable_bottleneck": True,
        "irregular_spacing": True,
        "score": "EXCELLENT"
    }
}
```

## 📊 Format de Sortie CSV

| Colonne | Description |
|---------|-------------|
| `segment_id` | Identifiant unique du segment |
| `road_name` | Nom de la route |
| `length_m` | Longueur en mètres |
| `lanes` | Nombre de voies |
| `maxspeed_kmh` | Vitesse limite |
| `has_signal_end` | **1 si feu à la fin du segment** (contrôlable par RL) |
| `u_lat`, `u_lon` | Coordonnées du début |
| `v_lat`, `v_lon` | Coordonnées de la fin |
| `geometry_wkt` | Géométrie WKT pour SIG |

## 🚦 À Propos du Forçage des Feux

En Afrique (et dans beaucoup de pays), **OSM n'a souvent pas les tags `highway=traffic_signals`** à jour. Utilisez `force_signals` pour ajouter manuellement les feux connus.

### Comment trouver les coordonnées exactes ?

1. Aller sur [OpenStreetMap](https://www.openstreetmap.org)
2. Rechercher le carrefour (ex: "Carrefour Vêdoko, Cotonou")
3. Cliquer sur le nœud "Junction"
4. Noter les coordonnées exactes

## 🤖 Critères d'Adéquation RL

Un corridor est "bon" pour le RL s'il a :

- ✅ **Branches asymétriques** : Multiple directions avec flux différents
- ✅ **Mouvements conflictuels** : Tournants à gauche qui interagissent
- ✅ **Goulot variable** : Position du goulot change selon l'heure
- ✅ **Espacement irrégulier** : Onde verte difficile à synchroniser

Un corridor **linéaire** avec espacement régulier est souvent **optimal pour Fixed-Time** → le RL ne peut pas le battre !

## 📈 Corridors Actuels

| ID | Ville | Score RL | Segments | Feux |
|----|-------|----------|----------|------|
| `cotonou_vedoko` | Cotonou, Bénin | EXCELLENT | 645 | 4 |
| `lagos_victoria_island` | Lagos, Nigeria | MODERATE | - | - |

---

*Auteur: Thesis RL Traffic Control - Novembre 2025*
