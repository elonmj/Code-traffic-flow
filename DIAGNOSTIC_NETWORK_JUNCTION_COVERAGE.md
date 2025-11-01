# 🔍 Diagnostic: Junction Info Coverage in Network

**Date**: October 31, 2025  
**Issue**: Vérifier si TOUS les segments reçoivent junction_info correctement

---

## ❓ Question Centrale

**Dans `_prepare_junction_info()`, la boucle `for link in self.links` suffit-elle ?**

```python
# Code actuel (network_grid.py:337-363)
for link in self.links:
    from_seg_id = link.from_segment
    node = link.via_node
    
    if node.traffic_lights is not None:
        # Set junction_info sur from_seg
        ...
```

### Problèmes Potentiels

1. **Segments sans Link** : Si segment a `end_node` mais pas de Link explicite
2. **Nœuds sans feu** : Si segment → nœud sans `traffic_lights`
3. **Multi-segments** : Plusieurs segments vers même nœud

---

## 🧪 Test Actuel: test_arz_congestion_formation.py

### Configuration Réseau

```yaml
Segments: 2
├─ upstream: node_0 (x=0) → node_1 (x=500m)
└─ downstream: node_1 (x=500m) → node_2 (x=1000m)

Nodes: 3
├─ node_0: boundary_inflow
├─ node_1: signalized_intersection (RED 60s, GREEN 30s)
└─ node_2: boundary_outflow

Links: ???
```

### ❓ Questions Critiques

1. Combien de `Link` sont créés ?
2. Le segment `upstream` reçoit-il `junction_info` ?
3. Le segment `downstream` a-t-il besoin de `junction_info` ?

---

## 🏗️ Architecture SUMO vs Notre Approche

### SUMO (MSNet Architecture)

**Source**: `src/microsim/MSNet.cpp`, `MSEdge.h`, `MSJunction.cpp`

```cpp
// SUMO: Junction gère TOUS les edges entrants/sortants
class MSJunction {
    std::vector<MSEdge*> incoming_edges;
    std::vector<MSEdge*> outgoing_edges;
    
    // À chaque step:
    void MSJunction::postMoveReminder() {
        // Pour CHAQUE incoming edge:
        for (MSEdge* edge : incoming_edges) {
            // Gère les véhicules qui veulent sortir
            edge->handleOutgoingVehicles(traffic_light_state);
        }
    }
};

// Edge stocke sa junction de sortie
class MSEdge {
    MSJunction* to_junction;  // NULL si boundary
    
    void handleOutgoingVehicles(TrafficLightState state) {
        if (to_junction && to_junction->has_red_light()) {
            // BLOCK vehicles at edge end
            block_outflow();
        }
    }
};
```

**Architecture SUMO** :
- ✅ Chaque `MSEdge` connaît sa `to_junction`
- ✅ Pas besoin de `Link` explicites
- ✅ Junction itère sur `incoming_edges` directement

### CityFlow (RoadNet Architecture)

**Source**: `src/roadnet/roadnet.cpp`, `roadnet.h`

```cpp
// CityFlow: Road stocke destination intersection
class Road {
    Intersection* start_intersection;  // Can be null (boundary)
    Intersection* end_intersection;    // Can be null (boundary)
    
    void Road::planRoute() {
        if (end_intersection && end_intersection->hasSignal()) {
            // Check traffic light
            bool can_pass = end_intersection->canEnter(this);
        }
    }
};

class Intersection {
    std::vector<Road*> roads_in;   // All incoming roads
    std::vector<Road*> roads_out;  // All outgoing roads
    TrafficLight* signal;
    
    bool canEnter(Road* from_road) {
        if (!signal) return true;
        return signal->isGreen(from_road);
    }
};
```

**Architecture CityFlow** :
- ✅ Chaque `Road` connaît `end_intersection` directement
- ✅ Pas de structure `Link` séparée
- ✅ Intersection référencée par les routes

---

## 🔴 Notre Architecture Actuelle (PROBLÈME ?)

```python
# Notre code (network_grid.py)
class NetworkGrid:
    self.segments: Dict[str, Grid1D]  # Segments isolés
    self.nodes: Dict[str, Node]       # Nœuds isolés
    self.links: List[Link]            # Connections EXPLICITES
    
    def _prepare_junction_info(self):
        # ⚠️ ITÈRE SUR LINKS, PAS SUR SEGMENTS !
        for link in self.links:
            from_seg_id = link.from_segment
            ...
```

### 🤔 Comparaison

| Architecture | Segment → Node | Découverte Junctions |
|--------------|----------------|---------------------|
| **SUMO** | `MSEdge.to_junction` | Direct (attribut) |
| **CityFlow** | `Road.end_intersection` | Direct (attribut) |
| **Notre code** | `segment['end_node']` | ⚠️ Via `links` (indirect) |

### ⚠️ Problème Identifié

Notre code :
1. ✅ Stocke `segment['end_node']` (comme SUMO/CityFlow)
2. ❌ Mais **n'utilise PAS** cet attribut dans `_prepare_junction_info()`
3. ❌ Itère sur `self.links` au lieu de `self.segments`

**Conséquence** : Si un segment a `end_node` mais pas de Link, il ne reçoit PAS de `junction_info` !

---

## ✅ Solution Recommandée

### Option 1: Itérer sur Segments (comme SUMO)

```python
def _prepare_junction_info(self, current_time: float):
    """Set junction info for ALL segments with outgoing nodes."""
    from ..network.junction_info import JunctionInfo
    
    # Clear existing
    for segment in self.segments.values():
        if hasattr(segment['grid'], 'junction_at_right'):
            segment['grid'].junction_at_right = None
    
    # Set junction info for segments with end_node
    for seg_id, segment in self.segments.items():
        end_node_id = segment.get('end_node')
        
        if end_node_id is not None:  # Segment has outgoing junction
            node = self.nodes[end_node_id]
            
            # Check if junction has traffic light
            if node.traffic_lights is not None:
                green_segments = node.traffic_lights.get_current_green_segments(current_time)
                
                if seg_id in green_segments:
                    light_factor = 1.0  # GREEN
                else:
                    light_factor = self.params.red_light_factor  # RED
                
                junction_info = JunctionInfo(
                    is_junction=True,
                    light_factor=light_factor,
                    node_id=node.node_id
                )
                segment['grid'].junction_at_right = junction_info
                
                logger.debug(f"Junction info: {seg_id} → {node.node_id}, "
                           f"factor={light_factor:.2f}")
```

**Avantages** :
- ✅ Garantit TOUS les segments sont traités
- ✅ Pas dépendant de `links` (peut être vide)
- ✅ Conforme architecture SUMO/CityFlow

---

## 🧪 Test de Validation Nécessaire

```python
def test_multi_segment_junction_coverage():
    """Verify ALL segments with junctions get junction_info"""
    
    # Create 3-segment network:
    # seg0 → node1 (RED light) → seg1 → node2 (GREEN light) → seg2
    
    network = NetworkGrid(params)
    network.add_segment('seg0', 0, 100, 50, start_node=None, end_node='node1')
    network.add_segment('seg1', 100, 200, 50, start_node='node1', end_node='node2')
    network.add_segment('seg2', 200, 300, 50, start_node='node2', end_node=None)
    
    # Add nodes with traffic lights
    network.add_node('node1', position=100, traffic_lights=..., ...)
    network.add_node('node2', position=200, traffic_lights=..., ...)
    
    # Step network
    network._prepare_junction_info(current_time=0)
    
    # VERIFY:
    assert network.segments['seg0']['grid'].junction_at_right is not None  # ✅ seg0 → node1
    assert network.segments['seg1']['grid'].junction_at_right is not None  # ✅ seg1 → node2
    assert network.segments['seg2']['grid'].junction_at_right is None      # ✅ seg2 → boundary
    
    print("✅ All segments with junctions have junction_info")
```

---

## 🎯 Conclusion

### Diagnostic

1. ❌ **Architecture actuelle** : Dépend de `links` (peut manquer des segments)
2. ✅ **SUMO/CityFlow** : Utilisent attribut direct `segment.end_node`
3. ⚠️ **Risque** : Segments sans Link ne reçoivent pas junction_info

### Action Requise

**REFACTOR `_prepare_junction_info()` pour itérer sur `self.segments` au lieu de `self.links`**

---

**Status**: 🔴 CRITICAL - Potentiel bug d'architecture réseau
