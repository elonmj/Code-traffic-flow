# Analyse Stratégique: Création Scénario Lagos Complet

**Date**: 22 Octobre 2025  
**Contexte**: Phase 6 complétée, données Lagos disponibles, objectif = scénario end-to-end

---

## 📊 État des Lieux: Ce Que Nous Avons

### ✅ **1. Phase 6 COMPLÈTE (Hétérogénéité Multi-Segments)**
- **NetworkConfig**: Chargement YAML 2-fichiers (network.yml + traffic_control.yml)
- **ParameterManager**: Paramètres globaux + locaux par segment (O(1))
- **NetworkGrid.from_yaml_config()**: Pipeline YAML → NetworkGrid complet
- **Tests**: 13/13 passés (100%)
- **Status**: Production-ready ✅

### ✅ **2. Données Lagos Réelles**
- **Fichier**: `donnees_trafic_75_segments (2).csv` (116,416 enregistrements)
- **Colonnes**: 
  - `timestamp`, `u`, `v` (nodes OpenStreetMap)
  - `name` (nom de rue)
  - `current_speed`, `freeflow_speed` (km/h)
  - `confidence` (0-1)
  - `api_key_used`
- **Couverture**: 75 segments réseau Victoria Island
- **Période**: Septembre 2024
- **Rues principales**:
  - Akin Adesola Street (artère principale)
  - Adeola Odeku Street (secondaire)
  - Ahmadu Bello Way (artère)
  - Saka Tinubu Street (tertiaire)

### ✅ **3. Système de Calibration**
- **CalibrationRunner**: Orchestration complète calibration
- **NetworkBuilder**: Construction réseau depuis segments
- **DataMapper**: Mapping données speed → ARZ
- **SpeedDataProcessor**: Traitement données Lagos (corrections spécifiques)
- **GroupManager**: Gestion groupes segments pour calibration
- **Status**: Modules existants ✅

### ✅ **4. Configurations Lagos Existantes**
- `traffic_lagos.yaml`: Paramètres trafic Lagos (gap-filling, creeping)
- `env_lagos.yaml`: Configuration environnement RL
- `signals_lagos.yaml`: Timing signaux adaptés Lagos
- `Code_RL/adapt_lagos.py`: Script adaptation paramètres
- **Status**: Configurations prêtes ✅

### ✅ **5. Infrastructure Simulation**
- **SimulationRunner**: Exécution simulations ARZ
- **NetworkGrid**: Coordination multi-segments
- **Traffic Control**: Gestion feux tricolores
- **Boundary Conditions**: Conditions limites configurables
- **Status**: Infrastructure complète ✅

---

## ❌ Ce Qui Manque: Gap Analysis

### 🔴 **GAP 1: Scénario YAML Lagos Complet**
**Problème**: Phase 6 a un exemple générique (Victoria_Island_Corridor) mais pas basé sur vraies données Lagos

**Ce qui manque**:
- `config/examples/lagos/network_lagos_real.yml`
  - 75 segments réels depuis `donnees_trafic_75_segments (2).csv`
  - Topologie complète: nodes, links, junctions
  - Paramètres locaux par segment (V0_c, V0_m basés sur freeflow_speed)
  - Highway types (primary/secondary/tertiary)
  - Lanes counts par segment
  
- `config/examples/lagos/traffic_control_lagos.yml`
  - Intersections clés identifiées (Akin Adesola x Adeola Odeku, etc.)
  - Phases signaux réalistes
  - Timing adapté trafic Lagos (min_green=15s, max_green=90s)

**Impact**: Sans ce fichier, impossible de charger réseau Lagos dans Phase 6

---

### 🟠 **GAP 2: Pipeline Calibration → YAML**
**Problème**: Calibration produit paramètres optimaux mais ne les exporte pas en YAML Phase 6

**Ce qui manque**:
- Fonction `export_calibration_to_phase6_yaml()`
  - Input: `CalibrationResult` depuis `CalibrationRunner`
  - Output: `network_lagos_calibrated.yml` avec paramètres optimaux par segment
  - Mapping: `ParameterSet` (calibration) → `ParameterManager` (Phase 6)

**Workflow actuel**:
```
CSV data → CalibrationRunner → ParameterSet optimisé
                                       ↓
                                   ??? (GAP)
                                       ↓
                            Phase 6 YAML config
```

**Workflow souhaité**:
```
CSV data → CalibrationRunner → ParameterSet optimisé
                                       ↓
                        export_calibration_to_phase6_yaml()
                                       ↓
              network_lagos_calibrated.yml (Phase 6 ready)
```

**Impact**: Pas de pont automatique entre calibration et simulation Phase 6

---

### 🟡 **GAP 3: Script End-to-End Lagos**
**Problème**: Pas de pipeline automatisé "données brutes → scénario simulé"

**Ce qui manque**:
- `scripts/create_lagos_scenario.py`
  - Step 1: Charger données CSV
  - Step 2: Construire topologie réseau (NetworkBuilder)
  - Step 3: Calibrer paramètres (CalibrationRunner)
  - Step 4: Générer YAML Phase 6 (export function)
  - Step 5: Valider scénario (run simulation test)
  - Step 6: Produire rapport

**Impact**: Processus manuel, risque d'erreurs, non reproductible

---

### 🟢 **GAP 4: Validation Scénario Lagos**
**Problème**: Pas de tests automatisés pour valider scénario Lagos complet

**Ce qui manque**:
- `test_lagos_scenario_integration.py`
  - Test 1: YAML Lagos charge correctement (75 segments)
  - Test 2: Paramètres hétérogènes appliqués (arterial vs residential)
  - Test 3: Simulation run sans erreur (t=0 → t=3600s)
  - Test 4: Résultats réalistes (vitesses, densités, débits)
  - Test 5: Comparaison avec données observées (RMSE, MAE)

**Impact**: Pas de garantie que scénario Lagos fonctionne correctement

---

## 🎯 Prochaine Étape Logique: Recommandation

### **Option A: Pipeline Automatisé Complet (Recommandé 🔥)**
**Durée estimée**: 3-4h  
**Impact**: Maximum - Scénario Lagos production-ready + reproductible

**Workflow**:
```
1. Créer script create_lagos_scenario.py (1.5h)
   ├─ Charger donnees_trafic_75_segments.csv
   ├─ Construire topologie NetworkBuilder
   ├─ Calibrer CalibrationRunner
   ├─ Exporter network_lagos_real.yml
   └─ Générer traffic_control_lagos.yml

2. Implémenter export_calibration_to_phase6_yaml() (0.5h)
   └─ Bridge CalibrationResult → Phase 6 YAML

3. Créer test_lagos_scenario_integration.py (1h)
   ├─ Test chargement YAML Lagos
   ├─ Test simulation complète
   └─ Test validation résultats

4. Documenter dans LAGOS_SCENARIO_README.md (0.5h)
   └─ Guide utilisation scénario Lagos
```

**Avantages**:
- ✅ Scénario Lagos complet et testé
- ✅ Pipeline reproductible (réutilisable autres villes)
- ✅ Pont calibration ↔ Phase 6 établi
- ✅ Validation automatisée
- ✅ Documentation complète

**Deliverables**:
1. `scripts/create_lagos_scenario.py` (pipeline automatisé)
2. `arz_model/calibration/export/phase6_exporter.py` (export function)
3. `config/examples/lagos/network_lagos_real.yml` (75 segments)
4. `config/examples/lagos/traffic_control_lagos.yml` (intersections réelles)
5. `test_lagos_scenario_integration.py` (5+ tests)
6. `LAGOS_SCENARIO_README.md` (documentation)

---

### **Option B: YAML Manuel Rapide**
**Durée estimée**: 1-2h  
**Impact**: Moyen - Scénario Lagos basique fonctionnel

**Workflow**:
```
1. Analyser manuellement donnees_trafic_75_segments.csv (0.5h)
   └─ Identifier segments principaux (top 20)

2. Créer network_lagos_simplified.yml manuellement (1h)
   └─ 20 segments clés (vs 75 complets)

3. Test rapide (0.5h)
   └─ Vérifier chargement + simulation run
```

**Avantages**:
- ✅ Rapide à implémenter
- ✅ Scénario fonctionnel immédiat

**Inconvénients**:
- ❌ Pas reproductible
- ❌ Couverture partielle (20/75 segments)
- ❌ Pas de calibration automatique
- ❌ Maintenance manuelle difficile

---

### **Option C: Calibration Seule (Incomplet)**
**Durée estimée**: 2h  
**Impact**: Faible - Paramètres optimaux mais pas de scénario Phase 6

**Workflow**:
```
1. Run CalibrationRunner sur données Lagos (1.5h)
2. Analyser résultats (0.5h)
```

**Problème**: Produit `ParameterSet` optimisé mais **pas de YAML Phase 6**  
→ Bloque l'utilisation du scénario avec NetworkGrid.from_yaml_config()

---

## 🚀 Ma Recommandation Finale

### **🔥 Option A: Pipeline Automatisé Complet**

**Justification**:
1. **Cohérence avec Phase 6**: Exploite pleinement l'infrastructure Phase 6 (NetworkConfig, ParameterManager, NetworkGrid.from_yaml_config())
2. **Reproductibilité**: Template réutilisable pour Paris, NYC, etc.
3. **Qualité Production**: Tests automatisés + documentation complète
4. **Investissement Rentable**: 3-4h pour infrastructure pérenne vs 1-2h pour solution jetable
5. **Pont Calibration-Simulation**: Établit connexion manquante entre modules

**Prochaines Actions**:
1. ✅ Créer `scripts/create_lagos_scenario.py` (pipeline complet)
2. ✅ Implémenter `export_calibration_to_phase6_yaml()` (bridge fonction)
3. ✅ Générer `config/examples/lagos/*.yml` (YAML Lagos réels)
4. ✅ Créer `test_lagos_scenario_integration.py` (validation)
5. ✅ Documenter dans `LAGOS_SCENARIO_README.md`

**Résultat Final**:
- Scénario Lagos complet avec 75 segments réels
- Paramètres calibrés par segment (hétérogénéité)
- Pipeline reproductible (CSV → YAML → Simulation)
- Tests automatisés (garantie qualité)
- Documentation utilisateur complète

---

## 📋 Détails Techniques: Ce Que Le Pipeline Doit Faire

### **Module 1: Extraction Topologie**
```python
# Input: donnees_trafic_75_segments.csv
# Output: Network topology with 75 segments

def extract_network_topology(csv_path):
    """
    Extract network topology from Lagos traffic data.
    
    Returns:
    - segments: List[RoadSegment] (75 segments)
    - nodes: Dict[node_id, Node] (junctions)
    - links: List[Link] (segment connections)
    """
    df = pd.read_csv(csv_path)
    
    # Group by (u, v) pairs to get unique segments
    segments = df.groupby(['u', 'v']).agg({
        'name': 'first',
        'freeflow_speed': 'mean',
        'current_speed': 'mean',
        'confidence': 'mean'
    }).reset_index()
    
    # Infer segment properties
    for seg in segments:
        seg['highway_type'] = infer_highway_type(seg['name'])
        seg['length'] = infer_length(seg['u'], seg['v'])
        seg['lanes'] = infer_lanes(seg['highway_type'])
    
    # Build nodes and links
    nodes = build_nodes(segments)
    links = build_links(segments, nodes)
    
    return segments, nodes, links
```

### **Module 2: Calibration Par Segment**
```python
# Input: segments with observed speeds
# Output: Calibrated parameters per segment

def calibrate_segment_parameters(segments, speed_data):
    """
    Calibrate ARZ parameters for each segment.
    
    Returns:
    - parameters: Dict[segment_id, ParameterSet]
    """
    calibration_runner = CalibrationRunner(config={
        'optimization': {
            'method': 'bayesian',
            'max_iterations': 50
        }
    })
    
    results = {}
    for segment in segments:
        # Filter speed data for this segment
        seg_data = speed_data[
            (speed_data['u'] == segment['u']) & 
            (speed_data['v'] == segment['v'])
        ]
        
        # Calibrate
        params = calibration_runner.calibrate_segment(
            segment_id=f"{segment['u']}_{segment['v']}",
            observed_speeds=seg_data['current_speed'],
            freeflow_speed=segment['freeflow_speed']
        )
        
        results[segment['id']] = params
    
    return results
```

### **Module 3: Export YAML Phase 6**
```python
# Input: calibrated parameters + topology
# Output: network_lagos_real.yml (Phase 6 format)

def export_to_phase6_yaml(segments, nodes, links, parameters, output_path):
    """
    Export calibrated network to Phase 6 YAML format.
    
    Generates:
    - network_lagos_real.yml: Topology + local parameters
    - traffic_control_lagos.yml: Traffic light timing
    """
    config = {
        'network': {
            'name': 'Victoria Island Lagos - Real Data',
            'segments': {},
            'nodes': {},
            'links': []
        }
    }
    
    # Export segments with local parameters
    for segment in segments:
        seg_id = segment['id']
        params = parameters[seg_id]
        
        config['network']['segments'][seg_id] = {
            'length': segment['length'],
            'cells': int(segment['length'] / 10),  # 10m cells
            'highway_type': segment['highway_type'],
            'lanes': segment['lanes'],
            'parameters': {
                'V0_c': params.V0_c,
                'V0_m': params.V0_m,
                'tau_c': params.tau_c,
                'tau_m': params.tau_m,
                'rho_max_c': params.rho_max_c,
                'rho_max_m': params.rho_max_m
            }
        }
    
    # Export nodes (junctions)
    for node_id, node in nodes.items():
        config['network']['nodes'][node_id] = {
            'type': 'junction',
            'incoming_segments': node['incoming'],
            'outgoing_segments': node['outgoing']
        }
    
    # Export links
    for link in links:
        config['network']['links'].append({
            'from_segment': link['from'],
            'to_segment': link['to'],
            'via_node': link['via'],
            'from_node': link['from_node'],
            'to_node': link['to_node'],
            'coupling_type': 'supply_demand'
        })
    
    # Write YAML
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return output_path
```

### **Module 4: Validation Scénario**
```python
# Input: network_lagos_real.yml
# Output: Validation report + test pass/fail

def validate_lagos_scenario(network_yaml_path):
    """
    Validate Lagos scenario completeness and accuracy.
    
    Tests:
    1. YAML loads successfully
    2. 75 segments present
    3. Parameters heterogeneous (arterial ≠ residential)
    4. Simulation runs without error
    5. Results match observed data (RMSE < threshold)
    """
    # Test 1: Load YAML
    params = ModelParameters()
    network = NetworkGrid.from_yaml_config(
        network_path=network_yaml_path,
        traffic_control_path='config/examples/lagos/traffic_control_lagos.yml',
        global_params=params,
        use_parameter_manager=True
    )
    
    assert len(network.segments) == 75, "Expected 75 segments"
    
    # Test 2: Heterogeneity
    pm = network.parameter_manager
    arterial_speed = pm.get('seg_akin_adesola_1', 'V0_c')
    residential_speed = pm.get('seg_saka_tinubu_1', 'V0_c')
    ratio = arterial_speed / residential_speed
    
    assert ratio > 1.5, f"Expected speed heterogeneity >1.5x, got {ratio:.2f}x"
    
    # Test 3: Simulation run
    network.initialize()
    for t in range(0, 3600, 10):  # 1 hour simulation
        network.step(dt=0.1)
    
    # Test 4: Validation vs observed data
    simulated_speeds = extract_speeds(network)
    observed_speeds = load_observed_speeds('donnees_trafic_75_segments.csv')
    
    rmse = compute_rmse(simulated_speeds, observed_speeds)
    assert rmse < 10.0, f"RMSE too high: {rmse:.2f} km/h"
    
    return {
        'tests_passed': 4,
        'rmse': rmse,
        'segments': 75,
        'heterogeneity_ratio': ratio
    }
```

---

## 📊 Estimation Détaillée

| Tâche | Durée | Complexité | Deliverables |
|-------|-------|------------|--------------|
| **1. Script Pipeline** | 1.5h | Moyenne | `create_lagos_scenario.py` (400 lines) |
| **2. Export Function** | 0.5h | Faible | `phase6_exporter.py` (150 lines) |
| **3. Tests Intégration** | 1h | Moyenne | `test_lagos_scenario_integration.py` (300 lines) |
| **4. Documentation** | 0.5h | Faible | `LAGOS_SCENARIO_README.md` (400 lines) |
| **5. Exécution Pipeline** | 0.5h | Variable | `network_lagos_real.yml`, `traffic_control_lagos.yml` |
| **TOTAL** | **4h** | - | **5 fichiers, ~1250 lines** |

---

## 🎯 Décision?

**Ma recommandation forte: Option A - Pipeline Automatisé Complet** 🔥

**Raisons**:
1. Investissement 4h pour infrastructure pérenne (vs 1-2h solution jetable)
2. Exploite pleinement Phase 6 (NetworkConfig, ParameterManager)
3. Pont calibration ↔ simulation établi (manque critique comblé)
4. Template reproductible (Paris, NYC, autres villes)
5. Tests automatisés + documentation = qualité production

**Question pour toi**: Es-tu d'accord pour investir 4h dans le pipeline complet, ou préfères-tu une solution rapide (Option B) pour avoir un premier scénario Lagos en 1-2h?

Je recommande **Option A** pour cohérence et qualité long-terme! 🚀
