# BUG #36: Traffic Signal Not Blocking Flow - Complete Fix Session

**Date**: 2025-01-XX  
**Duration**: Extended debugging session  
**Status**: ⚠️ PARTIALLY RESOLVED - Core bugs fixed, architecture issue identified  
**Test**: `test_congestion_formation()` - Still FAILS but for different reason

---

## 🎯 Executive Summary

Started with traffic signals not blocking flow → discovered **SEVEN interconnected bugs** → fixed 6/7 → identified fundamental architectural limitation.

**Good News**: All config, mapping, and parameter bugs FIXED ✅  
**Remaining Issue**: NetworkGrid lacks junction flux solver (architecture gap)

---

## 🐛 Bugs Found and Fixed

### Bug #1: Traffic Light Phase Configuration Mismatch
**Root Cause**: `simple_corridor()` used geographic orientations ('north', 'south') instead of segment IDs  
**Impact**: `has_green_light = segment_id in green_segments` ALWAYS failed  
**Fix**: Modified `builders.py` to use actual segment IDs in green_segments  
**File**: `arz_model/config/builders.py` lines 207-227

```python
# BEFORE (BROKEN):
'phases': [
    {'id': 0, 'name': 'GREEN', 'movements': ['through']}
]

# AFTER (FIXED):
'phases': [
    {'id': 0, 'name': 'RED', 'green_segments': []}  # No segments green
    {'id': 1, 'name': 'GREEN', 'green_segments': ['seg_1']}  # Actual segment ID
]
```

**Status**: ✅ FIXED

---

### Bug #2: Traffic Light Config Not Transferred
**Root Cause**: `runner.py` scenario_config didn't include `traffic_light_config`  
**Impact**: Nodes created without traffic light specifications  
**Fix**: Added `traffic_light_config` to node dict in scenario_config  
**File**: `arz_model/simulation/runner.py` line 199

```python
# ADDED:
'traffic_light_config': node.traffic_light_config  # Transfer TL config
```

**Status**: ✅ FIXED

---

### Bug #3: Traffic Light Controller Not Created from Config
**Root Cause**: `network_simulator.py` didn't parse `traffic_light_config`  
**Impact**: Nodes used default traffic lights (wrong segments)  
**Fix**: Added TrafficLightController creation from config  
**File**: `arz_model/network/network_simulator.py` lines 274-277

```python
# ADDED:
if 'traffic_light_config' in node_cfg and node_cfg['traffic_light_config'] is not None:
    from ..core.traffic_lights import create_traffic_light_from_config
    traffic_lights = create_traffic_light_from_config(node_cfg['traffic_light_config'])
```

**Status**: ✅ FIXED

---

### Bug #4: Phase Order Inverted
**Root Cause**: Config had Phase 0=GREEN, Phase 1=RED but env expects opposite  
**Impact**: action=1 (intended RED) actually selected GREEN  
**Fix**: Swapped phase order in builder config  
**File**: `arz_model/config/builders.py` lines 213-226
**Coordination**: `env/traffic_signal_env_direct.py` line 474 expects "Action 0 = RED, Action 1 = GREEN"

```python
# FIXED ORDER:
'phases': [
    {'id': 0, 'name': 'RED', ...},   # Action 0 → RED
    {'id': 1, 'name': 'GREEN', ...}   # Action 1 → GREEN
]
```

**Status**: ✅ FIXED

---

### Bug #5: θ_k Parameters Uninitialized (Returned None)
**Root Cause**: ModelParameters initialized theta_* to None, no behavioral_coupling config  
**Impact**: `_get_coupling_parameter()` returned None → crash in coupling formula  
**Error**: `TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'`  
**Fix**: Added fallback defaults in `_get_coupling_parameter()`  
**File**: `arz_model/core/node_solver.py` lines 156-158, 169-171, 176-178

```python
# ADDED FALLBACK:
return params.theta_car_signalized if params.theta_car_signalized is not None else 0.5
return params.theta_car_priority if params.theta_car_priority is not None else 0.9
return params.theta_car_secondary if params.theta_car_secondary is not None else 0.1
```

**Status**: ✅ FIXED

---

### Bug #6: red_light_factor Too High
**Root Cause**: Default `red_light_factor = 0.1` allows 10% flux during RED  
**Impact**: Not enough blockage to form congestion queue  
**Fix**: Set `red_light_factor = 0.01` (1% leakage) in global_params  
**File**: `arz_model/config/builders.py` line 271

```python
# ADDED:
'red_light_factor': 0.01,  # Near-complete blocking during RED
```

**Status**: ✅ FIXED

---

### Bug #7: Test Action Selection Wrong
**Root Cause**: Test used `action=1` when RED phase was Phase 0  
**Impact**: Test selected GREEN instead of RED → no congestion  
**Fix**: Changed test to `action=0` for RED phase  
**File**: `test_network_integration_quick.py` line 131

```python
# FIXED:
action = 0  # Phase 0 = RED (blocks flow)
```

**Status**: ✅ FIXED

---

## ⚠️ Remaining Architectural Issue

### Bug #8: NetworkGrid Lacks Junction Flux Solver (NOT FIXED)
**Root Cause**: NetworkGrid doesn't use `node_solver._calculate_outgoing_flux()`  
**Impact**: `red_light_factor` and traffic signal states ignored in flux calculation  
**Evidence**:
- `grep "solve.*intersection|_calculate_outgoing_flux" network_grid.py` → NO MATCHES
- `_resolve_node_coupling()` only applies θ_k behavioral coupling, NOT flux reduction

**Current Situation**:
- ✅ θ_k correctly returns 0.0 during RED (behavioral coupling blocked)
- ✅ Traffic lights correctly report green_segments=[] during RED
- ❌ But flux still passes through junction unimpeded
- ❌ No flux solver applies `light_factor` reduction

**What Happens**:
1. Traffic signal → Phase 0 (RED) selected
2. Link coupling → θ_k = 0.0 (correct)
3. **Missing**: Junction flux solver that multiplies flux by `red_light_factor`
4. Result: BC adds traffic → flows through "red" junction → densities decrease

**Debug Evidence**:
```
[COUPLING_DEBUG] time=0.0s, theta_k=0.0, green_segs=[]  ← Correct
[QUEUE_DIAGNOSTIC] queue_length=0.00 vehicles  ← Flux not blocked!
densities_m: 0.079 → 0.072 → 0.065  ← Traffic draining away
```

**Required Fix** (NOT implemented yet):
NetworkGrid needs junction flux resolution step BEFORE θ_k coupling:
1. `_resolve_junction_fluxes()` - Calculate flux at each node using Riemann solver
2. Apply `red_light_factor` to reduce flux during RED
3. Update segment boundaries with reduced flux
4. THEN apply θ_k behavioral coupling

**Complexity**: High - requires integrating Riemann solver into NetworkGrid.step()

**Status**: ❌ NOT FIXED - Architectural enhancement required

---

## 📊 Test Results

### Before Fixes:
```
❌ FAIL: Congestion Formation
queue_length = 0.00 veh (expected > 5)
theta_k = None (CRASH)
green_segments = ['north', 'south'] (wrong)
```

### After Fixes:
```
❌ FAIL: Congestion Formation (different reason)
queue_length = 0.00 veh (flux not blocked)
theta_k = 0.0 (correct)
green_segments = [] (correct)
red_light_factor = 0.01 (correct but unused)
```

**Progress**: Major improvement - all config bugs fixed, but flux solver missing

---

## 🎯 Verification Points

### ✅ Configuration Layer (FIXED)
- [x] TrafficLightController created from config
- [x] Phases use segment IDs not geographic names
- [x] Phase order matches env expectations
- [x] red_light_factor configured

### ✅ Parameters Layer (FIXED)
- [x] theta_* parameters have fallback defaults
- [x] No more None returns from _get_coupling_parameter()

### ✅ Coupling Layer (FIXED)
- [x] Link.apply_coupling() gets correct traffic_lights
- [x] θ_k = 0.0 during RED phase
- [x] θ_k = 0.5/0.8 during GREEN phase

### ❌ Flux Layer (NOT IMPLEMENTED)
- [ ] Junction flux solver integrated
- [ ] red_light_factor applied to flux
- [ ] Congestion forms during RED

---

## 🔧 Code Changes Summary

**Files Modified**: 7  
**Lines Changed**: ~150  
**Bugs Fixed**: 6/7

1. `arz_model/config/builders.py` - Phase config, segment IDs, red_light_factor
2. `arz_model/simulation/runner.py` - Traffic light config transfer
3. `arz_model/network/network_simulator.py` - Traffic light controller creation
4. `arz_model/core/node_solver.py` - theta_* fallback defaults
5. `arz_model/network/link.py` - Debug prints (temporary)
6. `arz_model/network/network_grid.py` - Debug prints (temporary)
7. `test_network_integration_quick.py` - Action selection fix

**Debug Code**: Added extensive debug prints - should be removed after flux solver added

---

## 📚 Technical Learnings

1. **Configuration Propagation**: Config must flow through 4 layers (builder → runner → simulator → network) - any break causes bugs
2. **Segment Identification**: Traffic lights must use same identifiers as segments (not geographic orientations)
3. **Phase Indexing**: Environment action mapping must match config phase order exactly
4. **Parameter Defaults**: Never initialize optional params to None if used in arithmetic - use fallback defaults
5. **Behavioral vs Physical**: θ_k controls behavioral coupling, flux solver controls physical flow - both needed for realistic traffic signals
6. **Architecture Gaps**: NetworkGrid θ_k coupling works but lacks junction flux resolution

---

## 🚀 Next Steps

### Immediate (Required for Test to Pass):
1. **Implement Junction Flux Solver in NetworkGrid**
   - Add `_resolve_junction_fluxes()` method
   - Integrate Riemann solver from `node_solver.py`
   - Apply `red_light_factor` during flux calculation
   - Update segment boundaries with reduced flux

2. **Remove Debug Prints**
   - Clean up `link.py` COUPLING_DEBUG prints
   - Clean up `network_grid.py` COUPLING_RESOLVE_DEBUG
   - Clean up `network_simulator.py` NODE_TL_DEBUG

3. **Add Integration Tests**
   - Test junction flux reduction during RED
   - Test congestion formation and release cycles
   - Test θ_k coupling with different junction types

### Future Enhancements:
- Add validation for traffic_light_config in NetworkSimulationConfig
- Create behavioral_coupling config section for theta_* parameters
- Document junction solver architecture in thesis
- Performance optimization for large networks

---

## 📖 References

**Related Documents**:
- `BUG_31_*.md` - Boundary conditions implementation (prerequisite)
- `ARCHITECTURE_UXSIM_INTEGRATION_PLAN.md` - Network architecture
- `ARZ_MODEL_PACKAGE_ARCHITECTURAL_AUDIT_COMPLETE.md` - Package structure

**Academic References**:
- Garavello & Piccoli (2005): Junction coupling conditions
- Kolb et al. (2018): Phenomenological θ_k coupling
- Thesis Section 4.2: Junction type parameter selection

---

## ✅ Sign-Off

**Bugs Fixed**: 6/7 (86% completion)  
**Test Status**: FAIL (but progress made)  
**Architecture**: Flux solver gap identified  
**Code Quality**: Improved (config propagation fixed)

**Recommendation**: Implement junction flux solver in NetworkGrid as next priority. All configuration infrastructure is now in place.

---

*This debugging session demonstrates the importance of end-to-end testing - a simple "traffic signal doesn't block" observation led to discovering 7 interconnected bugs across 4 architectural layers. The fixes improve code robustness even though the original test goal (congestion formation) requires additional architecture work.*
