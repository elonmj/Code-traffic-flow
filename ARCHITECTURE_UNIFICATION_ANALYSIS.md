# 🏗️ Analyse d'Unification Architecturale: Calibration vs Phase 6

**Date**: 22 Octobre 2025  
**Contexte**: Avant de créer le pipeline Lagos, il faut unifier les deux architectures réseau  
**Criticité**: 🔴 BLOQUANT pour Option A

---

## 🔍 Problème Identifié

**Tu as raison!** Les systèmes de calibration et Phase 6 représentent les réseaux **différemment**:

```
┌─────────────────────────────────────────────────────────────┐
│  CALIBRATION ARCHITECTURE (Pré-Phase 6)                    │
├─────────────────────────────────────────────────────────────┤
│  RoadSegment (dataclass)                                    │
│    ├─ segment_id: str                                       │
│    ├─ start_node: str                                       │
│    ├─ end_node: str                                         │
│    ├─ name: str                                             │
│    ├─ length: float                                         │
│    ├─ highway_type: str                                     │
│    ├─ oneway: bool                                          │
│    ├─ lanes: Optional[int]                                  │
│    └─ maxspeed: Optional[float]                             │
│                                                              │
│  NetworkBuilder.build_from_csv()                            │
│    └─ Returns: Dict[segment_id, RoadSegment]                │
│                                                              │
│  SegmentInfo (group_manager.py) - Différent format!         │
└─────────────────────────────────────────────────────────────┘

                        VS

┌─────────────────────────────────────────────────────────────┐
│  PHASE 6 ARCHITECTURE (NetworkGrid + ParameterManager)     │
├─────────────────────────────────────────────────────────────┤
│  YAML Structure:                                            │
│    segments:                                                │
│      seg_id:                                                │
│        length: 500                                          │
│        cells: 50                                            │
│        highway_type: 'primary'                              │
│        lanes: 3                                             │
│        parameters:                                          │
│          V0_c: 13.89                                        │
│          V0_m: 15.28                                        │
│                                                              │
│  NetworkGrid.from_yaml_config()                             │
│    └─ Creates: Segment objects + ParameterManager           │
│                                                              │
│  ParameterManager.get(seg_id, param_name)                   │
│    └─ Returns: Local override or global default             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔥 Incompatibilités Critiques

### **1. Représentation Segment**

| Attribut | Calibration (RoadSegment) | Phase 6 (YAML) | Compatible? |
|----------|---------------------------|----------------|-------------|
| ID | `segment_id` (str) | `seg_id` (dict key) | ⚠️ Naming |
| Topologie | `start_node`, `end_node` | ❌ Pas dans segments YAML | 🔴 MANQUE |
| Géométrie | `length` (meters) | `length` + `cells` | ⚠️ cells calculé |
| Type route | `highway_type` | `highway_type` | ✅ OK |
| Voies | `lanes` | `lanes` | ✅ OK |
| Vitesse | `maxspeed` (km/h) | ❌ Dans `parameters.V0_c` | 🔴 DIFFÉRENT |
| Direction | `oneway` (bool) | ❌ Pas stocké | 🔴 MANQUE |
| Paramètres | ❌ Pas stockés dans RoadSegment | `parameters: {...}` | 🔴 MANQUE |

### **2. Représentation Réseau**

| Concept | Calibration | Phase 6 | Compatible? |
|---------|-------------|---------|-------------|
| **Segments** | `Dict[str, RoadSegment]` | `Dict[str, {...}]` in YAML | ⚠️ Structure |
| **Nodes** | `Dict[str, NetworkNode]` avec `connected_segments` | `Dict[str, {...}]` avec type | ⚠️ Structure |
| **Links** | ❌ Implicite via nodes | `List[{from, to, via_node}]` **explicite** | 🔴 MANQUE |
| **Paramètres** | ❌ Pas de gestion centralisée | **ParameterManager** (global + local) | 🔴 MANQUE |

### **3. Pipeline de Construction**

```python
# CALIBRATION (Actuel):
NetworkBuilder.build_from_csv('corridor.csv')
  └─> Dict[segment_id, RoadSegment]
      └─> Pas de paramètres ARZ
      └─> Pas de structure YAML

# PHASE 6 (Nouveau):
NetworkConfig.load_from_files('network.yml', 'traffic_control.yml')
  └─> NetworkGrid.from_yaml_config()
      └─> Segments avec ParameterManager
      └─> Links explicites via nodes
```

**Problème**: Pas de pont `RoadSegment` → `Phase 6 YAML`

---

## 🎯 Stratégie d'Unification

### **Option 1: Adapter Calibration à Phase 6 (Recommandé 🔥)**

**Principe**: Modifier CalibrationRunner pour produire directement format Phase 6

**Avantages**:
- ✅ Phase 6 devient le **standard unifié**
- ✅ Pas de régression (Phase 6 déjà testé 13/13)
- ✅ Future-proof (tous nouveaux modules utilisent Phase 6)
- ✅ ParameterManager exploité à 100%

**Modifications Requises**:

```python
# 1. Ajouter méthode NetworkBuilder
class NetworkBuilder:
    def to_phase6_yaml(self) -> Dict[str, Any]:
        """
        Convertir RoadSegment → Phase 6 YAML structure.
        
        Returns:
            Dict compatible avec NetworkConfig
        """
        config = {
            'network': {
                'segments': {},
                'nodes': {},
                'links': []
            }
        }
        
        # Convertir segments
        for seg_id, road_seg in self.segments.items():
            config['network']['segments'][seg_id] = {
                'length': road_seg.length,
                'cells': int(road_seg.length / 10),  # 10m/cell
                'highway_type': road_seg.highway_type,
                'lanes': road_seg.lanes or 2,
                # Parameters ajoutés par calibration plus tard
                'parameters': {}
            }
        
        # Convertir nodes
        for node_id, node in self.nodes.items():
            config['network']['nodes'][node_id] = {
                'type': 'junction' if node.is_intersection else 'boundary',
                'incoming_segments': [],  # À remplir depuis connected_segments
                'outgoing_segments': []
            }
        
        # Générer links depuis topologie
        links = self._infer_links_from_topology()
        config['network']['links'] = links
        
        return config

    def _infer_links_from_topology(self) -> List[Dict]:
        """Inférer links depuis start_node/end_node des segments"""
        links = []
        
        # Group segments by end_node (potential connections)
        segments_by_end = defaultdict(list)
        for seg_id, seg in self.segments.items():
            segments_by_end[seg.end_node].append((seg_id, seg))
        
        # Find connections: seg1.end_node == seg2.start_node
        for seg_id, seg in self.segments.items():
            # This segment's end_node
            end_node = seg.end_node
            
            # Find segments starting at this node
            for next_seg_id, next_seg in self.segments.items():
                if next_seg.start_node == end_node and next_seg_id != seg_id:
                    links.append({
                        'from_segment': seg_id,
                        'to_segment': next_seg_id,
                        'via_node': end_node,
                        'from_node': end_node,
                        'to_node': next_seg.end_node,
                        'coupling_type': 'supply_demand'
                    })
        
        return links


# 2. Modifier CalibrationRunner pour exporter Phase 6
class CalibrationRunner:
    def export_to_phase6_yaml(
        self, 
        output_path: str,
        calibrated_params: Dict[str, ParameterSet]
    ) -> str:
        """
        Export calibration results to Phase 6 YAML format.
        
        Args:
            output_path: Where to save network.yml
            calibrated_params: Dict[segment_id, ParameterSet] from calibration
            
        Returns:
            Path to generated YAML file
        """
        # Get base structure from NetworkBuilder
        config = self.network_builder.to_phase6_yaml()
        
        # Add calibrated parameters to each segment
        for seg_id, param_set in calibrated_params.items():
            if seg_id in config['network']['segments']:
                config['network']['segments'][seg_id]['parameters'] = {
                    'V0_c': param_set.V0_c,
                    'V0_m': param_set.V0_m,
                    'tau_c': param_set.tau_c,
                    'tau_m': param_set.tau_m,
                    'rho_max_c': param_set.rho_max_c,
                    'rho_max_m': param_set.rho_max_m
                }
        
        # Write YAML
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"✅ Phase 6 YAML exported to: {output_path}")
        return output_path
```

**Effort**: 2-3h (modification NetworkBuilder + CalibrationRunner + tests)

---

### **Option 2: Bridge Bidirectionnel (Complexe)**

**Principe**: Créer converters dans les 2 sens

**Inconvénients**:
- ❌ Maintenance double (2 formats à synchroniser)
- ❌ Risque d'incohérence
- ❌ Complexité accrue

**Non recommandé** ⛔

---

### **Option 3: Unification Complète (Idéal mais long)**

**Principe**: Refactor complet vers architecture unique

**Avantages**:
- ✅ Architecture unifiée parfaite
- ✅ Pas de conversion nécessaire

**Inconvénients**:
- ❌ 10-15h de refactoring
- ❌ Risque de régression sur calibration existante
- ❌ Bloque pipeline Lagos (délai inacceptable)

**Réservé pour Phase 7** (refactoring majeur) ⏭️

---

## ✅ Recommandation: Option 1 (Adapter Calibration à Phase 6)

### **Plan d'Unification (2-3h)**

```
Phase 1: NetworkBuilder Enhancement (1h)
├─ [_] Ajouter to_phase6_yaml() method
│  ├─ [_] Convertir RoadSegment → segments dict
│  ├─ [_] Convertir NetworkNode → nodes dict
│  └─ [_] Inférer links depuis start_node/end_node
│
├─ [_] Ajouter _infer_links_from_topology() helper
│  └─ [_] Trouver connections: seg1.end == seg2.start

Phase 2: CalibrationRunner Enhancement (0.5h)
├─ [_] Ajouter export_to_phase6_yaml() method
│  ├─ [_] Call NetworkBuilder.to_phase6_yaml()
│  ├─ [_] Add calibrated parameters per segment
│  └─ [_] Write YAML file

Phase 3: Tests Unification (0.5h)
├─ [_] Test conversion RoadSegment → Phase 6 YAML
├─ [_] Test calibrated params export
└─ [_] Test NetworkGrid.from_yaml_config() loads exported YAML

Phase 4: Documentation (0.5h)
└─ [_] Update CalibrationRunner docstring avec workflow unifié

TOTAL: 2.5h
```

---

## 🔄 Workflow Unifié Final

Après unification, le workflow sera:

```
┌─────────────────────────────────────────────────────────────┐
│  PIPELINE LAGOS UNIFIÉ (Post-Unification)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Données Lagos CSV                                       │
│     └─> donnees_trafic_75_segments.csv                     │
│                                                              │
│  2. NetworkBuilder (Calibration)                            │
│     ├─> build_from_csv()                                    │
│     └─> Dict[segment_id, RoadSegment]                       │
│                                                              │
│  3. CalibrationRunner                                       │
│     ├─> calibrate_segment_parameters()                      │
│     └─> Dict[segment_id, ParameterSet]                      │
│                                                              │
│  4. Export Phase 6 (NOUVEAU! 🔥)                            │
│     ├─> CalibrationRunner.export_to_phase6_yaml()           │
│     ├─> NetworkBuilder.to_phase6_yaml()                     │
│     └─> network_lagos_real.yml ✅                           │
│                                                              │
│  5. NetworkGrid (Phase 6)                                   │
│     ├─> NetworkGrid.from_yaml_config()                      │
│     ├─> ParameterManager active                             │
│     └─> Simulation prête! ✅                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Résultat**: Pont seamless Calibration → Phase 6 ✅

---

## 📝 Détails Techniques: Conversion RoadSegment → Phase 6

### **Mapping Attributs**

| RoadSegment (Source) | Phase 6 YAML (Target) | Transformation |
|----------------------|-----------------------|----------------|
| `segment_id` | Segment dict key | Direct |
| `length` | `length` | Direct (meters) |
| - | `cells` | `int(length / 10)` (10m/cell) |
| `highway_type` | `highway_type` | Direct |
| `lanes` | `lanes` | Default 2 if None |
| `maxspeed` (km/h) | `parameters.V0_c` | `maxspeed / 3.6` (→ m/s) |
| `start_node`, `end_node` | `links` via_node | Infer connections |
| - | `parameters.{V0_m, tau_c, ...}` | From calibration |

### **Challenge: Inferring Links**

**Problème**: RoadSegment a `start_node`/`end_node` mais Phase 6 requiert links explicites

**Solution**:
```python
def _infer_links_from_topology(self) -> List[Dict]:
    """
    Infer links from segment topology.
    
    Algorithm:
    - For each segment S1 with end_node N
    - Find all segments S2 with start_node == N
    - Create link: S1 → S2 via N
    """
    links = []
    
    # Build index: node → segments starting there
    segments_by_start = defaultdict(list)
    for seg_id, seg in self.segments.items():
        segments_by_start[seg.start_node].append((seg_id, seg))
    
    # Find connections
    for seg_id, seg in self.segments.items():
        end_node = seg.end_node
        
        # Find segments starting at this segment's end
        for next_seg_id, next_seg in segments_by_start[end_node]:
            if next_seg_id != seg_id:  # Avoid self-loops
                links.append({
                    'from_segment': seg_id,
                    'to_segment': next_seg_id,
                    'via_node': end_node,
                    'from_node': end_node,
                    'to_node': next_seg.end_node,
                    'coupling_type': 'supply_demand'
                })
    
    return links
```

### **Challenge: Boundary vs Junction Nodes**

**Problème**: Phase 6 distingue `boundary` (entry/exit) vs `junction` (internal)

**Solution**:
```python
def _classify_node_type(self, node_id: str) -> str:
    """
    Classify node as boundary or junction.
    
    Heuristic:
    - Boundary: 1 connected segment (dead-end)
    - Junction: 2+ connected segments
    """
    connected_count = len(self.nodes[node_id].connected_segments)
    
    if connected_count == 1:
        return 'boundary'
    else:
        return 'junction'
```

---

## ⚠️ Risques et Mitigations

### **Risque 1: Perte d'Information**
**Détail**: RoadSegment.oneway pas stocké dans Phase 6 YAML  
**Mitigation**: Ajouter `oneway` field à Phase 6 YAML structure  
**Impact**: Faible - oneway rarement utilisé en simulation ARZ

### **Risque 2: Links Incorrects**
**Détail**: Inférence topologique peut créer faux liens  
**Mitigation**: Validation post-export + tests automatisés  
**Impact**: Moyen - détectable via tests

### **Risque 3: Paramètres Manquants**
**Détail**: Calibration peut ne pas optimiser tous params  
**Mitigation**: Fallback sur global defaults via ParameterManager  
**Impact**: Faible - comportement déjà implémenté Phase 6

---

## 🎯 Décision?

**Ma recommandation forte: Option 1 - Adapter Calibration à Phase 6** 🔥

**Justification**:
1. **Pragmatique**: 2.5h vs 10-15h (Option 3)
2. **Safe**: Phase 6 déjà testé 13/13 (pas de régression)
3. **Future-proof**: Phase 6 devient standard unifié
4. **Débloque**: Pipeline Lagos opérationnel après unification

**Next Actions**:
1. ✅ Implémenter `NetworkBuilder.to_phase6_yaml()` (1h)
2. ✅ Implémenter `CalibrationRunner.export_to_phase6_yaml()` (0.5h)
3. ✅ Créer tests unification (0.5h)
4. ✅ Documenter workflow unifié (0.5h)
5. ✅ **PUIS** créer pipeline Lagos (4h Option A)

**Timeline**:
- Unification: 2.5h
- Pipeline Lagos: 4h
- **Total: 6.5h** pour scénario Lagos production-ready

---

## 📋 Checklist Unification

```
Phase 1: NetworkBuilder Enhancement
├─ [_] Créer arz_model/calibration/export/__init__.py
├─ [_] Créer arz_model/calibration/export/phase6_converter.py
│  ├─ [_] to_phase6_yaml(network_builder) → Dict
│  ├─ [_] _infer_links_from_topology() → List[Dict]
│  ├─ [_] _classify_node_type() → str
│  └─ [_] _convert_speed_units() → float (km/h → m/s)
│
├─ [_] Modifier arz_model/calibration/core/network_builder.py
│  └─ [_] Ajouter to_phase6_yaml() method (wrapper)

Phase 2: CalibrationRunner Enhancement
├─ [_] Modifier arz_model/calibration/core/calibration_runner.py
│  └─ [_] Ajouter export_to_phase6_yaml() method

Phase 3: Tests
├─ [_] Créer test_phase6_unification.py
│  ├─ [_] Test 1: RoadSegment → Phase 6 conversion
│  ├─ [_] Test 2: Links inference correcte
│  ├─ [_] Test 3: Calibrated params export
│  ├─ [_] Test 4: NetworkGrid charge YAML exporté
│  └─ [_] Test 5: ParameterManager fonctionne

Phase 4: Documentation
├─ [_] Update CalibrationRunner docstring
└─ [_] Update PHASE6_README.md avec workflow calibration

TOTAL: 2.5h → Architectures unifiées ✅
```

---

**Question pour toi**: 

Es-tu d'accord pour investir **2.5h dans l'unification d'abord**, puis 4h pour le pipeline Lagos (total 6.5h)?

C'est la bonne approche pour éviter dette technique et avoir architecture solide! 🏗️
