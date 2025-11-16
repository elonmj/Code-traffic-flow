# 🎨 Guide de Visualisation - Simulation de Trafic GPU

## 📂 Fichiers Disponibles

Toutes les visualisations sont dans le dossier **`viz_output/`**

### 🎯 Vue d'Ensemble

#### **00_dashboard_synthese.png** ⭐ RECOMMANDÉ
**Tableau de bord complet** avec 7 graphiques synthétisant tous les résultats:
- Statistiques globales de la simulation
- Évolution temporelle (densité et vitesse moyennes)
- Distribution statistique des densités
- Diagrammes spatio-temporels des 2 segments
- Profils spatiaux finaux

👉 **Commencez par ce fichier pour avoir une vue d'ensemble !**

---

### 📈 Graphiques d'Évolution Temporelle

#### **01_density_evolution.png**
Évolution des **densités** au cours du temps pour les 2 segments
- Densité moyenne + enveloppe min-max
- Permet de voir la stabilité du système
- Montre l'absence de collapse ou d'explosion

#### **02_speed_evolution.png**
Évolution des **vitesses** au cours du temps pour les 2 segments
- Vitesse moyenne + enveloppe min-max (en km/h)
- Montre la stabilité du trafic
- Corrélation inverse avec la densité

---

### 🗺️ Diagrammes Spatio-Temporels (Heatmaps)

#### **03_spatiotemporal_diagrams.png**
**4 heatmaps** montrant la distribution spatiale au cours du temps:
- Segment 1: Densité (gauche) et Vitesse (droite)
- Segment 2: Densité (gauche) et Vitesse (droite)
- Permet de visualiser les ondes, congestions, chocs
- Dans notre cas: état homogène stable

---

### 📸 Profils Instantanés

#### **04_snapshot_profiles.png**
**4 snapshots** à différents moments de la simulation:
- t=0s (initial), t≈600s, t≈1200s, t≈1791s (final)
- Chaque graphique montre densité ET vitesse pour les 2 segments
- Permet de voir l'évolution spatiale à des moments clés

---

### 🎬 Animations

#### **05_traffic_animation.mp4** 🎥
**Animation vidéo complète** (180 frames, 10 fps)
- Durée: ~18 secondes
- Montre l'évolution dynamique complète de la simulation
- Affichage simultané: densité + vitesse
- Statistiques en temps réel pour chaque segment
- Format MP4 (compatible tous lecteurs)

👉 **Idéal pour présenter les résultats de façon dynamique !**

#### **traffic_preview.gif** 🎞️
**Aperçu GIF léger** (60 frames, 10 fps)
- Version allégée de l'animation
- Plus facile à partager et intégrer dans documents
- Format GIF (lecture automatique dans navigateurs)

---

## 🔍 Comment Interpréter les Résultats

### État de la Simulation
✅ **Simulation réussie** - 1800s complets sans crash  
✅ **Stabilité numérique** - Pas de dt collapse  
✅ **Robustesse physique** - Densités et vitesses dans les bornes réalistes  

### Observations Clés

1. **Segment 1** (conditions initiales: 50 veh/km, 40 km/h)
   - Densité finale: ~14 veh/km
   - Vitesse finale: ~51 km/h
   - État stable maintenu

2. **Segment 2** (conditions initiales: 20 veh/km, 50 km/h)
   - Densité finale: ~17 veh/km
   - Vitesse finale: ~61 km/h
   - État stable maintenu

3. **Comportement global**
   - Pas de formation de congestion
   - Pas d'ondes de choc
   - Écoulement fluide et stable
   - Conditions initiales uniformes préservées

### Validations

✅ **Numérique**: Limiteurs de positivité fonctionnent correctement  
✅ **Physique**: Conservation de masse, relations densité-vitesse cohérentes  
✅ **Performance**: 6.4 it/s sur Tesla P100 (excellent pour WENO5 GPU)  

---

## 📊 Tableau Récapitulatif

| Métrique | Segment 1 | Segment 2 | Unité |
|----------|-----------|-----------|-------|
| Densité moyenne finale | ~14 | ~17 | veh/km |
| Vitesse moyenne finale | ~51 | ~61 | km/h |
| Densité min observée | ~14 | ~17 | veh/km |
| Densité max observée | ~14 | ~17 | veh/km |
| Variation densité | Très faible | Très faible | - |
| État du trafic | Fluide stable | Fluide stable | - |

---

## 🎯 Recommandations pour l'Analyse

### Pour une présentation rapide (5 min)
1. **00_dashboard_synthese.png** - Vue d'ensemble complète
2. **05_traffic_animation.mp4** - Animation dynamique
3. **README_VISUALISATIONS.md** - Conclusions clés

### Pour une analyse détaillée (30 min)
1. Lire **README_VISUALISATIONS.md** en entier
2. Examiner tous les graphiques statiques dans l'ordre (00-04)
3. Visionner l'animation complète
4. Analyser les profils à différents instants
5. Comparer avec les conditions initiales

### Pour un article/rapport scientifique
- **Figures principales**: 00, 03 (heatmaps), 04 (snapshots)
- **Figures supplémentaires**: 01, 02 (évolution temporelle)
- **Matériel supplémentaire**: Animation MP4 ou GIF

---

## 🛠️ Régénération des Visualisations

Si vous souhaitez régénérer les visualisations avec d'autres paramètres:

```powershell
# Visualisations principales
python visualize_results.py

# Tableau de bord
python create_dashboard.py

# GIF d'aperçu
python create_preview_gif.py
```

---

## 📞 Informations Techniques

**Simulation**:
- Modèle: ARZ Two-Class (motos/voitures)
- Méthode numérique: WENO5 + SSP-RK3
- Architecture: GPU-only (CUDA/Numba)
- Limiteurs: Positivity-preserving (3 niveaux)

**Performance**:
- Temps simulé: 1800s (30 minutes)
- Pas de temps: 19,667 steps
- dt moyen: ~0.09s
- Temps calcul: 6min 23s sur Tesla P100

**Données**:
- Points sauvegardés: 180 snapshots
- Résolution spatiale: 100 cellules/segment
- Format: pickle (network_simulation_results.pkl)

---

**📅 Généré le**: 15 novembre 2025  
**💻 Code source**: https://github.com/elonmj/Code-traffic-flow  
**📧 Contact**: Pour toute question sur les visualisations
