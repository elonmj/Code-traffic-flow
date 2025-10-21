# Analyse Architecturale: Comment les Simulateurs de Trafic Professionnels Gèrent les Réseaux

**Date**: 2025-01-21  
**Auteur**: Analyse comparative SUMO vs CityFlow vs notre arz_model  
**Objectif**: Comprendre les design patterns architecturaux pour gérer réseaux multi-segments  

---

## 🎯 QUESTIONS FONDAMENTALES

Tu as posé LES bonnes questions:
1. Comment gèrent-ils le modèle réseau dans l'écriture du code?
2. Est-ce un nouveau dossier qui étend le petit dossier spécifique à un segment?
3. Comment organisent-ils segment vs network vs node?

---

## 📊 ANALYSE COMPARATIVE

### 1. SUMO (Eclipse - Standard de l'industrie)

#### Architecture Découverte

**Structure de dossiers** (`src/microsim/`):
```
microsim/
├── MSNet.h/cpp              # LE RÉSEAU GLOBAL (singleton)
├── MSEdgeControl.h/cpp      # Contrôleur de TOUS les edges
├── MSEdge.h/cpp             # UN EDGE (segment de route)
├── MSLane.h/cpp             # UNE LANE (voie dans un edge)
├── MSJunction.h/cpp         # UN NŒUD (intersection)
├── MSLink.h/cpp             # UN LIEN (connexion lane→lane)
├── MSLogicJunction.cpp      # Logique de priorité aux nœuds
└── MSInternalJunction.cpp   # Nœuds internes (dans intersections)
```

**Hiérarchie des classes** (code réel extrait):

```cpp
// LE RÉSEAU CENTRAL - MSNet.h
class MSNet : public Parameterised {
private:
    MSEdgeControl* myEdges;           // ⭐ Contrôle TOUS les edges
    MSJunctionControl* myJunctions;    // ⭐ Contrôle TOUTES les intersections
    MSTLLogicControl* myLogics;        // Contrôle feux de signalisation
    
    // CRITICAL: Le réseau POSSÈDE les conteneurs, pas les objets directement
};

// CONTRÔLEUR D'EDGES - MSEdgeControl.h
class MSEdgeControl {
private:
    const MSEdgeVector myEdges;        // ⭐ VECTEUR de TOUS les edges
    const std::vector<MSLane*> myLanes; // TOUS les lanes
};

// UN EDGE (SEGMENT DE ROUTE) - MSEdge.h  
class MSEdge : public Named, public Parameterised {
private:
    std::shared_ptr<const std::vector<MSLane*>> myLanes;  // ⭐ Les lanes de CET edge
    MSJunction* myFromJunction;         // ⭐ Jonction de départ
    MSJunction* myToJunction;           // ⭐ Jonction d'arrivée
    MSEdgeVector mySuccessors;          // Edges suivants
    MSEdgeVector myPredecessors;        // Edges précédents
    double myLength;                    // Longueur de cet edge
    
public:
    const std::vector<MSLane*>& getLanes() const { return *myLanes; }
    MSJunction* getFromJunction() const { return myFromJunction; }
    MSJunction* getToJunction() const { return myToJunction; }
};

// UNE LANE (VOIE) - MSLane.h
class MSLane : public Named {
private:
    MSEdge* myEdge;                    // ⭐ Edge parent
    std::vector<MSLink*> myLinks;      // Liens vers autres lanes
    std::list<MSVehicle*> myVehicles;  // Véhicules sur cette lane
    double myLength;
};

// UN NŒUD (INTERSECTION) - MSJunction.h
class MSJunction : public Named {
private:
    ConstMSEdgeVector myIncoming;      // ⭐ Edges entrants
    ConstMSEdgeVector myOutgoing;      // ⭐ Edges sortants
    Position myPosition;                // Position géométrique
    
public:
    const ConstMSEdgeVector& getIncoming() const { return myIncoming; }
    const ConstMSEdgeVector& getOutgoing() const { return myOutgoing; }
};

// UN LIEN (CONNEXION LANE→LANE) - MSLink.h
class MSLink {
private:
    MSLane* myLane;                    // Lane source
    MSLane* myLaneBefore;              // Lane précédente
    MSJunction* myJunction;            // Junction contenant ce lien
    LinkDirection myDirection;          // Direction (straight, left, right)
};
```

**PATTERN ARCHITECTURAL DÉCOUVERT**:

```
┌─────────────────────────────────────────────────────────────┐
│                         MSNet                               │
│                    (RÉSEAU GLOBAL)                          │
│  ┌─────────────────┐        ┌──────────────────┐          │
│  │ MSEdgeControl   │        │ MSJunctionControl│          │
│  │ (TOUS edges)    │        │ (TOUS junctions) │          │
│  └────────┬────────┘        └────────┬─────────┘          │
└───────────┼──────────────────────────┼────────────────────┘
            │                          │
            │                          │
    ┌───────▼───────┐          ┌──────▼────────┐
    │   MSEdge1     │◄─────────┤  MSJunction1  │
    │  (Segment 1)  │          │ (Intersection)│
    │  ┌─────────┐  │          └──────┬────────┘
    │  │ MSLane1 │  │                 │
    │  │ MSLane2 │  │                 │
    │  └─────────┘  │          ┌──────▼────────┐
    └───────┬───────┘          │  MSJunction2  │
            │                  │ (Intersection)│
            │                  └──────┬────────┘
    ┌───────▼───────┐                │
    │   MSEdge2     │◄───────────────┘
    │  (Segment 2)  │
    │  ┌─────────┐  │
    │  │ MSLane1 │  │
    │  │ MSLane2 │  │
    │  └─────────┘  │
    └───────────────┘
```

**FLUX DE CONSTRUCTION** (code extrait):

```cpp
// 1. Création du réseau global
MSNet::MSNet(...) {
    myEdges = nullptr;
    myJunctions = nullptr;
}

// 2. Fermeture de la construction (closeBuilding)
void MSNet::closeBuilding(
    MSEdgeControl* edges,           // ⭐ Collection de TOUS les edges
    MSJunctionControl* junctions,   // ⭐ Collection de TOUTES les intersections
    ...) {
    myEdges = edges;
    myJunctions = junctions;
}

// 3. MSEdgeControl contient TOUS les edges
MSEdgeControl::MSEdgeControl(const std::vector<MSEdge*>& edges)
    : myEdges(edges) {
    // Construction de la liste de toutes les lanes
    for (MSEdge* edge : myEdges) {
        for (MSLane* lane : edge->getLanes()) {
            myLanes.push_back(lane);
        }
    }
}

// 4. Chaque MSEdge connaît ses jonctions
void MSEdge::setJunctions(MSJunction* from, MSJunction* to) {
    myFromJunction = from;
    myToJunction = to;
}
```

**SIMULATION STEP** (code extrait):

```cpp
// MSNet.cpp - Boucle principale
void MSNet::simulationStep() {
    // 1. Mise à jour de TOUS les edges via le contrôleur
    myEdges->executeMove();  // ⭐ Délégation au contrôleur
}

// MSEdgeControl.cpp
void MSEdgeControl::executeMove() {
    // 2. Parcours de TOUS les edges
    for (MSEdge* edge : myEdges) {
        // 3. Chaque edge met à jour SES lanes
        for (MSLane* lane : edge->getLanes()) {
            lane->executeMove();
        }
    }
}
```

---

### 2. CityFlow (MIT - Optimisé pour RL)

#### Architecture Découverte

**Structure de dossiers** (`src/roadnet/`):
```
roadnet/
├── roadnet.h/cpp          # RoadNet class (LE RÉSEAU)
├── trafficlight.h/cpp     # Traffic light logic
└── (tout dans roadnet.h)  # ⚠️ Architecture moins modulaire que SUMO
```

**Hiérarchie des classes** (code réel extrait):

```cpp
// LE RÉSEAU - roadnet.h
class RoadNet {
private:
    std::vector<Road> roads;                    // ⭐ TOUS les roads
    std::vector<Intersection> intersections;    // ⭐ TOUTES les intersections
    std::map<std::string, Road*> roadMap;       // Mapping ID→Road
    std::map<std::string, Intersection*> interMap; // Mapping ID→Intersection
    std::vector<Lane*> lanes;                   // TOUTES les lanes (pointeurs)
    std::vector<LaneLink*> laneLinks;          // TOUS les liens
    std::vector<Drivable*> drivables;          // TOUS les objets "roulables"
    
public:
    bool loadFromJson(std::string jsonFileName);
    const std::vector<Road>& getRoads() const { return roads; }
    const std::vector<Intersection>& getIntersections() const { return intersections; }
};

// UN ROAD (SEGMENT)
class Road {
private:
    std::string id;
    Intersection* startIntersection;   // ⭐ Intersection de départ
    Intersection* endIntersection;     // ⭐ Intersection d'arrivée
    std::vector<Lane> lanes;           // Les lanes de CE road
    std::vector<Point> points;         // Géométrie du road
    
public:
    const Intersection& getStartIntersection() const { return *startIntersection; }
    const Intersection& getEndIntersection() const { return *endIntersection; }
};

// UNE LANE (VOIE)
class Lane : public Drivable {
private:
    Road* belongRoad;                  // ⭐ Road parent
    std::vector<Segment> segments;     // ⭐ Segmentation pour efficacité
    std::vector<LaneLink*> laneLinks;  // Liens vers autres lanes
    std::list<Vehicle*> vehicles;      // Véhicules sur cette lane
    
public:
    void buildSegmentation(size_t numSegs);  // ⭐ Subdivision en segments
};

// UN SEGMENT (SUBDIVISION D'UNE LANE)
class Segment {
private:
    size_t index;
    Lane* belongLane;                  // ⭐ Lane parente
    double startPos, endPos;           // Position dans la lane
    std::list<std::list<Vehicle*>::iterator> vehicles; // Véhicules
};

// UNE INTERSECTION
class Intersection {
private:
    std::string id;
    std::vector<Road*> roads;          // ⭐ Roads connectés
    std::vector<RoadLink> roadLinks;   // Connexions road→road
    std::vector<Cross> crosses;        // Conflits entre trajectoires
    TrafficLight trafficLight;         // Feu de signalisation
    
public:
    void initCrosses();                // Calcul des conflits
};

// UN LIEN ROAD→ROAD
class RoadLink {
private:
    Road* startRoad;
    Road* endRoad;
    std::vector<LaneLink> laneLinks;   // Connexions lane→lane
    int index;
};

// UN LIEN LANE→LANE
class LaneLink : public Drivable {
private:
    Lane* startLane;
    Lane* endLane;
    std::vector<Cross*> crosses;       // Conflits avec autres laneLinks
    std::vector<Point> points;         // Géométrie du lien
};
```

**PATTERN ARCHITECTURAL DÉCOUVERT**:

```
┌──────────────────────────────────────────────────────────┐
│                        RoadNet                           │
│                   (RÉSEAU GLOBAL)                        │
│  ┌──────────────────┐      ┌───────────────────┐       │
│  │ vector<Road>     │      │ vector<Intersection>│      │
│  │ (TOUS roads)     │      │ (TOUTES intersect.) │      │
│  └────────┬─────────┘      └─────────┬──────────┘      │
└───────────┼────────────────────────────┼────────────────┘
            │                            │
    ┌───────▼──────────┐         ┌──────▼────────┐
    │   Road 1         │◄────────┤ Intersection 1│
    │  ┌────────────┐  │         │  ┌──────────┐ │
    │  │ Lane 1     │  │         │  │RoadLink1 │ │
    │  │ ┌────────┐ │  │         │  │RoadLink2 │ │
    │  │ │Segment1│ │  │         │  └──────────┘ │
    │  │ │Segment2│ │  │         └───────┬───────┘
    │  │ └────────┘ │  │                 │
    │  │ Lane 2     │  │         ┌───────▼───────┐
    │  └────────────┘  │         │ Intersection 2│
    └──────────────────┘         └───────────────┘
```

**CARACTÉRISTIQUES CLÉS**:

1. **Segmentation des Lanes**: ⭐ INNOVATION MAJEURE
```cpp
// CityFlow divise chaque lane en segments pour efficacité
void Lane::buildSegmentation(size_t numSegs) {
    segments.resize(numSegs);
    double segmentLength = length / numSegs;
    for (size_t i = 0; i < numSegs; i++) {
        segments[i].startPos = i * segmentLength;
        segments[i].endPos = (i + 1) * segmentLength;
        segments[i].belongLane = this;
    }
}
```

2. **Calcul des Conflits**:
```cpp
// Intersection.cpp - Calcul automatique des croisements
void Intersection::initCrosses() {
    // Pour chaque paire de laneLinks
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            // Calcul géométrique des points de croisement
            // Création d'objets Cross pour gérer les priorités
        }
    }
}
```

---

## 🔍 DESIGN PATTERNS COMMUNS IDENTIFIÉS

### Pattern #1: **Séparation Topologie / Physique** ⭐⭐⭐

**SUMO**:
```cpp
// La TOPOLOGIE (graphe de connectivité)
MSNet → MSEdgeControl → vector<MSEdge*>
MSNet → MSJunctionControl → vector<MSJunction*>

// La PHYSIQUE (simulation)
MSEdge → vector<MSLane*> → list<MSVehicle*>
MSLane::executeMove()  // Équations de mouvement
```

**CityFlow**:
```cpp
// La TOPOLOGIE
RoadNet → vector<Road> + vector<Intersection>

// La PHYSIQUE
Lane → vector<Segment> → list<Vehicle*>
Lane::planChange()  // Équations de mouvement
```

**⚠️ Notre problème actuel**:
```cpp
// ❌ MÉLANGE topologie + physique dans Grid1D
Grid1D {
    double xmin, xmax;  // ⚠️ UN SEUL segment géométrique
    int N_total;         // ⚠️ Grille pour UN segment
}

// ❌ NetworkCoupling essaie d'étendre APRÈS coup
NetworkCoupling {
    // TODO: Implémenter logique multi-segments
    mid_idx = grid.N_physical // 2;  // ⚠️ STUB!
}
```

---

### Pattern #2: **Hiérarchie Network > Segment > Lane** ⭐⭐⭐

**Les deux utilisent la même hiérarchie**:

```
NIVEAU 1 (GLOBAL):  Network / RoadNet
                    └─ Gère la collection complète
                    └─ Point d'entrée simulation
                    
NIVEAU 2 (SEGMENT): Edge / Road  
                    └─ Un segment de route entre 2 nœuds
                    └─ Contient plusieurs lanes
                    
NIVEAU 3 (VOIE):    Lane / Lane
                    └─ Une voie de circulation
                    └─ Contient les véhicules
                    └─ Applique les équations physiques
                    
NIVEAU 4 (MICRO):   Cell / Segment (CityFlow uniquement)
                    └─ Subdivision pour efficacité O(1)
```

**⚠️ Notre architecture actuelle**:
```
❌ NIVEAU 1: (MANQUANT - pas de NetworkGrid)
❌ NIVEAU 2: Grid1D (mais mono-segment!)
✅ NIVEAU 3: Les cellules [0...N_total]
```

---

### Pattern #3: **Collections Centralisées** ⭐⭐⭐

**SUMO**:
```cpp
class MSEdgeControl {
    const MSEdgeVector myEdges;         // ⭐ TOUS les edges
    const std::vector<MSLane*> myLanes; // ⭐ TOUTES les lanes
    
    void executeMove() {
        for (MSEdge* edge : myEdges) {  // Parcours efficace
            edge->executeMove();
        }
    }
};
```

**CityFlow**:
```cpp
class RoadNet {
    std::vector<Road> roads;            // ⭐ TOUS les roads
    std::vector<Lane*> lanes;           // ⭐ TOUTES les lanes (flat)
    std::vector<Drivable*> drivables;   // ⭐ TOUS objets simulables
    
    void reset() {
        for (auto& road : roads) road.reset();  // Parcours direct
    }
};
```

**Avantage**: Boucle de simulation O(N) directe, pas de recherche!

---

### Pattern #4: **Connectivité Explicite** ⭐⭐

**SUMO**:
```cpp
class MSEdge {
    MSJunction* myFromJunction;  // ⭐ Jonction source
    MSJunction* myToJunction;    // ⭐ Jonction destination
    MSEdgeVector mySuccessors;   // ⭐ Edges suivants possibles
    MSEdgeVector myPredecessors; // ⭐ Edges précédents
};

class MSJunction {
    ConstMSEdgeVector myIncoming;  // ⭐ Edges entrants
    ConstMSEdgeVector myOutgoing;  // ⭐ Edges sortants
};
```

**CityFlow**:
```cpp
class Road {
    Intersection* startIntersection;  // ⭐ Début
    Intersection* endIntersection;    // ⭐ Fin
};

class Intersection {
    std::vector<Road*> roads;         // ⭐ Roads connectés
    std::vector<RoadLink> roadLinks;  // ⭐ Liens road→road
};
```

**Principe**: Chaque élément connaît ses voisins (graphe bidirectionnel).

---

### Pattern #5: **Couplage aux Nœuds via Links** ⭐⭐⭐

**SUMO**:
```cpp
class MSLink {
    MSLane* myLane;           // Lane source
    MSLane* myLaneBefore;     // Lane destination  
    MSJunction* myJunction;   // Junction contenant ce lien
    
    bool opened(...);         // Calcul priorité
};

class MSLane {
    std::vector<MSLink*> myLinks;  // ⭐ Liens vers lanes suivantes
};
```

**CityFlow**:
```cpp
class LaneLink : public Drivable {
    Lane* startLane;
    Lane* endLane;
    std::vector<Cross*> crosses;  // ⭐ Gestion conflits
    
    bool isAvailable() const;     // Vérification feu
};

class Lane {
    std::vector<LaneLink*> laneLinks;  // ⭐ Connexions
};
```

**Principe**: Les **liens** sont des objets de première classe qui:
- Gèrent la logique de priorité
- Calculent les conflits
- Appliquent les règles de feux

---

### Pattern #6: **Segmentation Interne (CityFlow)** ⭐

**Innovation de CityFlow pour performance**:
```cpp
class Segment {
    Lane* belongLane;
    double startPos, endPos;
    std::list<Vehicle*> vehicles;  // ⭐ Subdivision spatiale
};

class Lane {
    std::vector<Segment> segments;
    
    Vehicle* getVehicleBeforeDistance(double dis, size_t segmentIndex) {
        // ⭐ Recherche O(1) dans le bon segment
        // Au lieu de O(N) dans toute la lane
    }
};
```

**Bénéfice**: Recherche de véhicules proche = O(1) au lieu de O(N).

---

## 💡 RECOMMANDATIONS POUR ARZ_MODEL

### Option A: **Architecture SUMO-like** (Recommandée) ⭐⭐⭐

```
Créer une hiérarchie complète:

src/arz_model/
├── network/
│   ├── network_grid.h/cpp      # ⭐ NOUVEAU - Réseau global
│   ├── segment_grid.h/cpp      # ⭐ NOUVEAU - Un segment (ancien Grid1D)
│   ├── node.h/cpp              # ⭐ NOUVEAU - Un nœud
│   └── link.h/cpp              # ⭐ NOUVEAU - Connexion segment→segment
├── grid/
│   └── grid1d.h/cpp            # EXISTANT - Devient SegmentGrid
├── numerics/
│   ├── time_integration.h/cpp  # EXISTANT - Modifié pour NetworkGrid
│   └── boundary_conditions.h/cpp # EXISTANT - BC par segment
└── core/
    ├── node_solver.h/cpp       # EXISTANT - Solver Riemann aux nœuds
    └── intersection.h/cpp      # EXISTANT - Data structures
```

**Implémentation**:

```cpp
// network_grid.h - NOUVEAU
class NetworkGrid {
private:
    std::vector<SegmentGrid*> segments_;     // ⭐ TOUS les segments
    std::vector<Node*> nodes_;               // ⭐ TOUS les nœuds
    std::map<std::string, SegmentGrid*> segment_map_;
    
public:
    NetworkGrid(const std::vector<SegmentConfig>& configs);
    
    void step(double dt);  // Simulation globale
    
    SegmentGrid* get_segment(const std::string& id);
    Node* get_node(const std::string& id);
};

// segment_grid.h - RENOMMAGE de Grid1D
class SegmentGrid {
private:
    std::string id_;
    Node* start_node_;    // ⭐ Nœud de départ
    Node* end_node_;      // ⭐ Nœud d'arrivée
    
    double xmin_, xmax_;  // Domaine de CE segment
    int N_physical_;
    // ... reste identique à Grid1D actuel
    
public:
    void step_segment(double dt);  // Simulation de CE segment
    
    Node* get_start_node() const { return start_node_; }
    Node* get_end_node() const { return end_node_; }
};

// node.h - NOUVEAU
class Node {
private:
    std::string id_;
    std::vector<SegmentGrid*> incoming_segments_;  // ⭐ Segments entrants
    std::vector<SegmentGrid*> outgoing_segments_;  // ⭐ Segments sortants
    NodeSolver* solver_;                            // Solver Riemann
    
public:
    void solve_fluxes(double dt);  // Calcul flux aux nœuds
    
    const std::vector<SegmentGrid*>& get_incoming() const;
    const std::vector<SegmentGrid*>& get_outgoing() const;
};
```

**Flux de simulation**:
```cpp
// time_integration.cpp - MODIFIÉ
void step_network(NetworkGrid& network, double dt) {
    // 1. Mise à jour de TOUS les segments
    for (SegmentGrid* segment : network.get_segments()) {
        segment->step_segment(dt);
    }
    
    // 2. Résolution aux nœuds (couplage)
    for (Node* node : network.get_nodes()) {
        node->solve_fluxes(dt);
    }
    
    // 3. Application des flux calculés aux BCs
    for (SegmentGrid* segment : network.get_segments()) {
        segment->apply_node_fluxes();
    }
}
```

**AVANTAGES**:
✅ Architecture claire et modulaire  
✅ Suit les best practices SUMO  
✅ Réutilise Grid1D existant (renommé)  
✅ Permet multi-segments natif  
✅ Extensible (ajout segments facile)  

**EFFORT**: 2-3 semaines de développement

---

### Option B: **Source Term Approach** (Plus rapide) ⭐⭐

```cpp
// Rester mono-segment mais ajouter terme source
class Grid1D {
    // ... existant ...
    
    // ⭐ NOUVEAU: Source term pour traffic signal
    std::vector<SourceTerm> source_terms_;
    
    void add_source_term(double x_pos, 
                        std::function<double(double)> coeff) {
        source_terms_.push_back({x_pos, coeff});
    }
};

// time_integration.cpp - Ajout source term
void compute_source_term(Grid1D& grid, np::ndarray& U, double t) {
    for (const auto& src : grid.get_source_terms()) {
        int i = grid.position_to_index(src.x_pos);
        double S = src.coefficient(t);  // Traffic light state
        
        // S = -K * δ(x - x_light) * (1 - traffic_state)
        // Modifie friction localement
        U[1, i] += S * dt;  // Momentum equation
    }
}
```

**AVANTAGES**:
✅ Rapide à implémenter (quelques jours)  
✅ Physiquement correct  
✅ Pas de refactoring majeur  
✅ Suffit pour traffic signal control  

**LIMITES**:
❌ Reste mono-segment  
❌ Pas extensible à vrais réseaux complexes  
❌ Ne suit pas les patterns industriels  

**EFFORT**: 3-5 jours de développement

---

## 🎯 DÉCISION RECOMMANDÉE

**COURT TERME (pour débloquer RL)**:
→ **Option B (Source Term)** pour avoir reward signal rapidement

**MOYEN TERME (pour thèse)**:
→ **Option A (Architecture SUMO-like)** pour architecture professionnelle

**Justification**:
1. Tu dois valider le RL MAINTENANT → Source term = solution temporaire
2. Pour la thèse, tu DOIS avoir architecture propre → SUMO pattern = référence
3. Les deux ne sont PAS mutuellement exclusifs: faire B puis A

---

## 📚 LESSONS LEARNED

### Ce que SUMO/CityFlow font BIEN:

1. **Séparation claire**: Network ≠ Segment ≠ Lane
2. **Collections centralisées**: Pas de recherche, juste parcours
3. **Connectivité explicite**: Chaque objet connaît ses voisins
4. **Liens comme objets**: MSLink/LaneLink gèrent la logique de couplage
5. **Hiérarchie cohérente**: Network > Segment > Lane > Cell

### Ce que notre code doit changer:

1. ❌ **Grid1D ne peut PAS gérer multi-segments** → Créer NetworkGrid
2. ❌ **network_coupling.py est incomplet** → TODOs partout
3. ❌ **Architecture mono-segment native** → Conçu pour 1 segment

### Le vrai problème:

> **L'architecture ARZ a été conçue pour UN segment académique, pas pour des réseaux réels.**

C'est **NORMAL** - c'est un code de recherche! Mais pour l'industrie, il faut:
- Soit refactorer vers architecture SUMO (Option A)
- Soit accepter limitation mono-segment (Option B + clarifier scope)

---

## 🔗 RÉFÉRENCES

**SUMO**:
- Repository: https://github.com/eclipse-sumo/sumo
- Documentation: https://sumo.dlr.de/docs/
- Architecture: `src/microsim/` folder

**CityFlow**:
- Repository: https://github.com/cityflow-project/CityFlow
- Paper: WWW 2019 Demo Paper
- Architecture: `src/roadnet/roadnet.h`

**Autres simulateurs analysés**:
- MATSim (Java, agent-based)
- Flow (UC Berkeley, RL benchmark)
- VISSIM (commercial, pas open-source)

---

**CONCLUSION**: Les simulateurs professionnels utilisent TOUS une architecture:
1. Réseau global contenant collections de segments/edges
2. Chaque segment est autonome avec sa grille de calcul
3. Les nœuds gèrent le couplage via objets Link
4. Séparation stricte topologie (graphe) / physique (équations)

Notre `Grid1D` mono-segment + `network_coupling` incomplet ne peut PAS rivaliser avec ces architectures établies. Il faut soit:
- **Refactorer complètement** (Option A - 2-3 semaines)
- **Accepter limitation** et utiliser source term (Option B - 3-5 jours)

**Recommandation**: Option B MAINTENANT, Option A pour la thèse finale.
