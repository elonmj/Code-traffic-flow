<!-- markdownlint-disable-file -->

# 🚀 NIVEAU 3 INTEGRATION: Bring Real-World Validation to Unified Architecture

**Date**: 2025-10-22  
**Objective**: Make niveau3_realworld_validation work with new Phase 6/7 architecture  
**Current State**: Nivel3 designed for old architecture - needs bridge  

---

## 📋 PROBLEM ANALYSIS

### Current Niveau 3 Architecture
```
OLD DESIGN (isolated):
  TomTom Data → Trajectories → Feature Extraction
                                      ↓
                              Observed Metrics
                                      ↓
              Compare with SPRINT 3 predictions (hardcoded)
```

**Issue**: 
- Predictions are static/hardcoded (10.0 km/h speed differential, etc.)
- No simulation integrated
- Can't generate predictions dynamically with new scenarios

### New Required Architecture
```
NEW DESIGN (unified):
  Lagos Scenario (CSV) → NetworkBuilder + ParameterManager
                               ↓
                    NetworkGrid.from_network_builder()
                               ↓
                      Run ARZ Simulation
                               ↓
                      Generate Predictions
                               ↓
  TomTom Data → Trajectories → Feature Extraction → Observed Metrics
                                                            ↓
         Compare Predictions ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
```

---

## 🎯 INTEGRATION STRATEGY (QUICK)

### Step 1: Create Simulation Wrapper for Niveau 3
**File**: `niveau3_realworld_validation/arz_simulator.py` (NEW)

**Purpose**: Bridge between new unified architecture and validation

```python
from scenarios.lagos_victoria_island import create_grid
from arz_model.core.parameters import ModelParameters

class ARZSimulatorForValidation:
    """Run ARZ simulation and extract predictions for niveau 3 validation."""
    
    def __init__(self, scenario_name='lagos_victoria_island'):
        """Initialize simulator with scenario."""
        self.scenario_name = scenario_name
        self.grid = None
        self.predictions = {}
    
    def run_simulation(self, duration_seconds=3600, dt=0.1):
        """Run ARZ simulation and extract metrics."""
        # Create network from scenario
        self.grid = create_grid()
        
        # Initialize
        self.grid.initialize()
        
        # Run simulation
        for t in range(int(duration_seconds / dt)):
            self.grid.step(dt)
        
        # Extract predictions
        self.predictions = self._extract_metrics()
        
        return self.predictions
    
    def _extract_metrics(self):
        """Extract metrics matching niveau 3 format."""
        metrics = self.grid.get_network_metrics()
        
        # Convert to niveau 3 format
        return {
            'speed_differential': {
                'delta_v_kmh': self._compute_speed_differential(),
                'v_moto_kmh': metrics.get('avg_speed_m', 0) * 3.6,  # approx
            },
            'throughput_ratio': {
                'throughput_ratio': self._compute_throughput_ratio(),
            },
            'fundamental_diagrams': {
                'q_rho_curves': self._extract_fundamental_diagrams()
            }
        }
    
    def _compute_speed_differential(self):
        """Compute speed differential between vehicle classes."""
        # Extract from simulation state
        pass
    
    def _compute_throughput_ratio(self):
        """Compute throughput ratio (motos/cars)."""
        pass
    
    def _extract_fundamental_diagrams(self):
        """Extract fundamental diagrams."""
        pass
```

### Step 2: Update validation_comparison.py
**Change**: Load predictions from simulator instead of hardcoded values

```python
# OLD
delta_v_pred = 10.0  # Hardcoded!

# NEW
simulator = ARZSimulatorForValidation()
predicted_metrics = simulator.run_simulation()
delta_v_pred = predicted_metrics['speed_differential']['delta_v_kmh']
```

### Step 3: Update quick_test_niveau3.py
**Add**: Simulation step before comparison

```python
# NEW STEP 0: Run ARZ simulation
print("\nSTEP 0: RUNNING ARZ SIMULATION")
simulator = ARZSimulatorForValidation()
predicted_metrics = simulator.run_simulation(duration_seconds=3600)
print(f"✅ Generated predictions: {predicted_metrics}")

# STEP 3: Compare with predictions
comparator = ValidationComparator(
    predicted_metrics=predicted_metrics,  # From simulation, not file
    observed_metrics_path=observed_metrics_path
)
```

---

## 🔧 IMPLEMENTATION PHASES

### Phase 1: Create Bridge (30m)
- [ ] Create `arz_simulator.py` with simulation wrapper
- [ ] Implement metric extraction methods
- [ ] Test simulator runs without errors

### Phase 2: Update Validation (20m)
- [ ] Modify `validation_comparison.py` to accept sim results
- [ ] Update `quick_test_niveau3.py` to run simulation
- [ ] Test full pipeline

### Phase 3: Validate (15m)
- [ ] Run full niveau 3 validation
- [ ] Compare: Predictions vs Observations
- [ ] Generate comparison report

**Total Time**: ~1 hour

---

## 📊 EXPECTED WORKFLOW

```
$ python validation_ch7_v2/scripts/niveau3_realworld_validation/quick_test_niveau3.py

OUTPUT:
=============================================================================
SPRINT 4: REAL-WORLD DATA VALIDATION (NIVEAU 3)
=============================================================================

STEP 0: RUNNING ARZ SIMULATION
  Loading: scenarios.lagos_victoria_island
  Creating NetworkGrid from CSV
  Running 3600 seconds simulation...
  ✅ Generated predictions:
     Speed differential: 12.3 km/h (from simulation!)
     Throughput ratio: 1.52x (from simulation!)

STEP 1: LOADING TRAJECTORIES
  ✅ Loaded 50,000 trajectory points
  Vehicles: 342
  Classes: {'moto': 198, 'car': 144}

STEP 2: EXTRACTING OBSERVED METRICS
  ✅ Extracted metrics:
     Speed differential: 11.8 km/h (from TomTom data!)
     Throughput ratio: 1.49x (from TomTom data!)
     
STEP 3: VALIDATION COMPARISON
  ✅ SPEED DIFFERENTIAL: PASS (error: 4.1%)
  ✅ THROUGHPUT RATIO: PASS (error: 2.0%)
  ✅ FUNDAMENTAL DIAGRAMS: PASS (correlation: 0.82)
  
OVERALL VALIDATION: ✅ PASS (4/4 criteria met)
=============================================================================
```

---

## 📋 FILES TO CREATE/MODIFY

### New Files
```
✓ niveau3_realworld_validation/arz_simulator.py (200 lines)
  → Simulation wrapper
  → Metric extraction
  → Bridge to unified architecture
```

### Modified Files
```
✓ niveau3_realworld_validation/validation_comparison.py
  → Accept predicted_metrics dict (not just file path)
  → Support sim results
  
✓ niveau3_realworld_validation/quick_test_niveau3.py
  → Add simulation step
  → Run simulator before comparison
```

---

## 🎯 SUCCESS CRITERIA

✅ **Niveau 3 integration complete when:**
1. Simulator runs without errors
2. Generates predictions from Lagos scenario
3. Compares predictions vs observations
4. All 4 validation criteria evaluated
5. Generates comprehensive report

---

## 💡 KEY INSIGHT

> **"Unify predictions and observations in one pipeline"**

Before: Static hardcoded predictions  
After: Dynamic predictions from simulation + real observations  

Result: **True validation** (theory vs real-world) ✅

---

## 🚀 READY TO START?

**Recommend**: Phase 1 + 2 + 3 sequentially (~1h total)

**First Action**: Create `arz_simulator.py` with simulation wrapper

Want me to **start building** the integration?

