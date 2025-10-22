<!-- markdownlint-disable-file -->

# 🧹 CLEANUP PLAN: Legacy Network Patterns Across All Modules

**Date**: 2025-10-22  
**Objective**: Remove all old network-building patterns and consolidate to unified direct integration architecture  
**Scope**: 439 Python files, all folders across arz_model/  

---

## 📋 AUDIT FINDINGS

### Total Count
- **Total Python Files**: 439
- **Files with Legacy Patterns**: ~180+ identified
- **Categories**: 4 major types of legacy code

---

## 🗂️ LEGACY PATTERNS IDENTIFIED

### **Category 1: Manual Network Creation (TestFix Priority: High)**
Files creating networks manually instead of using factories:
- `arz_model/tests/test_network_module.py` - 40+ instances of manual Node/Segment creation
- `arz_model/tests/test_network_integration.py` - 30+ instances of hardcoded network setup
- `arz_model/tests/test_network_system.py` - 25+ instances
- `test_networkbuilder_to_networkgrid.py` - Test networks (should migrate to scenarios)
- Decision: **CONSOLIDATE to scenario factories** (use `scenarios.`... pattern)

### **Category 2: NetworkSimulator Old Patterns (Deprecation Priority: HIGH)**
`arz_model/network/network_simulator.py`:
- Line 106: `self.network = NetworkGrid(self.params)` - Old empty initialization
- Line 242: `_build_network_from_config()` - Manual network building from config
- Current Usage: Only in NetworkSimulator (isolated pattern)
- Decision: **REPLACE with NetworkGrid.from_yaml_config() or from_network_builder()**

### **Category 3: Single-Segment Legacy Code (Backward Compat Priority: MEDIUM)**
`Code_RL/src/utils/config.py`:
- Lines 474, 541: References to "single-segment BC" for backward compatibility
- Comment: "Keep OLD single-segment BC for backward compatibility if network is disabled"
- Context: BUG #31 fix kept legacy BC for safety
- Decision: **DEPRECATE with warning, archive pattern explanation**

### **Category 4: TODO/Incomplete Network Methods (Cleanup Priority: MEDIUM)**
`arz_model/calibration/core/network_builder.py`:
- Line 333: `# TODO: Uncomment when networkx is available` (network export)
- Line 356: `print(f"Network export to {output_file} - TODO: Implement when networkx available")`
- Decision: **REMOVE - networkx optional, not core**

### **Category 5: Data CSV Manual Columns (Data Format Priority: LOW)**
Multiple files reference `lanes_manual`, `maxspeed_manual_kmh`, `Rx_manual`:
- `arz_model/calibration/data/` modules
- `validation_ch7_v2/` data loading scripts
- `Code_RL/` utilities
- Decision: **DOCUMENT pattern, keep for CSV compatibility (not a code pattern, data pattern)**

---

## 🎯 CLEANUP STRATEGY

### Phase 1: HIGH Priority (TestFix + Core Deprecation) - 2-3h
**Impact**: Remove redundant test code, simplify core

#### 1.1: Consolidate Test Networks to Scenarios
**Files to Modify**:
- `arz_model/tests/test_network_module.py` (40 manual creations)
- `arz_model/tests/test_network_integration.py` (30 manual creations)
- `arz_model/tests/test_network_system.py` (25 manual creations)
- `test_networkbuilder_to_networkgrid.py` (test file, can stay but mark as legacy test)

**Action**: 
1. Create scenario factories for test networks (e.g., `scenarios/test_simple_2seg.py`)
2. Replace all manual `NetworkGrid(params)` with `NetworkGrid.from_network_builder()`
3. Update imports to use `scenarios.`...
4. Result: Tests use production pattern (scenarios) not manual patterns

**Expected Reduction**: ~95 lines of manual test code → 5 scenario factory files

#### 1.2: Fix NetworkSimulator._build_network_from_config()
**File**: `arz_model/network/network_simulator.py`

**Current Pattern** (BAD):
```python
def __init__(self, scenario_config):
    self.network = NetworkGrid(self.params)  # Empty!
    self._build_network_from_config(scenario_config)  # Manual build

def _build_network_from_config(self, config):
    # Manual segment/node creation
```

**New Pattern** (GOOD):
```python
def __init__(self, scenario_config):
    # Use NetworkGrid.from_yaml_config() or from_network_builder()
    self.network = NetworkGrid.from_yaml_config(
        config_file=scenario_config['network_config'],
        traffic_file=scenario_config['traffic_control']
    )
```

**Action**:
1. Replace `_build_network_from_config()` with factory method call
2. Delete empty `NetworkGrid()` initialization
3. Validate NetworkSimulator still works
4. Result: Clean, simple initialization pattern

**Expected Reduction**: ~150 lines of manual building code

### Phase 2: MEDIUM Priority (Data Patterns + Backward Compat) - 1-2h
**Impact**: Clean up old single-segment legacy, document data patterns

#### 2.1: Archive Single-Segment Legacy Code
**File**: `Code_RL/src/utils/config.py`

**Current Pattern** (DEPRECATED):
```python
# Keep OLD single-segment BC for backward compatibility if network is disabled
if self.network_disabled:
    # Use old single-segment boundary condition modulation
```

**Action**:
1. Create `_deprecated/single_segment_bc_legacy.md` with explanation
2. Add `@deprecated` warning to code path
3. Log warning when legacy path is used
4. Mark for removal in version 2.0
5. Result: Legacy code visible but marked for removal

**Expected Change**: +4 lines warnings, +1 deprecation marker

#### 2.2: Document CSV Manual Column Pattern
**Create**: `DATA_FORMAT_REFERENCE.md`

**Content**:
- Explain `lanes_manual`, `maxspeed_manual_kmh`, `Rx_manual` purpose
- These are CSV input columns (not code pattern)
- Referenced in calibration data modules
- Not a code smell, part of CSV standard
- No action needed (keep as-is)

**Result**: Documented, team understands pattern

### Phase 3: LOW Priority (Optional Cleanup) - 0.5-1h
**Impact**: Polish, remove dead code

#### 3.1: Remove TODO Network Export Methods
**File**: `arz_model/calibration/core/network_builder.py`

**Current Code**:
```python
# Line 333-356: Network export TODO
# TODO: Uncomment when networkx is available
print(f"Network export to {output_file} - TODO: Implement when networkx available")
```

**Action**:
1. Remove entire `export_network()` method (dead code)
2. If networkx integration needed, create separate feature branch
3. Result: Clean, no dead code

**Expected Reduction**: ~30 lines of dead code

---

## 📊 CLEANUP INVENTORY

### Files to Modify (High Priority)
```
✓ arz_model/tests/test_network_module.py
  → Replace 40 manual creations with scenario factories
  → Lines affected: 50-150 range
  
✓ arz_model/tests/test_network_integration.py
  → Replace 30 manual creations with scenario factories
  → Lines affected: 40-100 range
  
✓ arz_model/tests/test_network_system.py
  → Replace 25 manual creations with scenario factories
  → Lines affected: 60-100 range
  
✓ arz_model/network/network_simulator.py
  → Replace _build_network_from_config() pattern
  → Lines affected: 106-250 range (150 lines to remove)
```

### Files to Document/Archive (Medium Priority)
```
✓ Code_RL/src/utils/config.py
  → Add deprecation warnings
  → Lines affected: 541 (1 line warning)
  
✓ DATA_FORMAT_REFERENCE.md (NEW)
  → Document CSV manual columns pattern
  → Explain it's data format, not code pattern
```

### Files to Create (Scenario Migration)
```
✓ scenarios/test_simple_2seg.py
  → Factory for basic 2-segment test network
  → ~50 lines
  
✓ scenarios/test_complex_junction.py
  → Factory for complex junction network
  → ~50 lines
  
✓ scenarios/test_traffic_lights.py
  → Factory for traffic light network
  → ~50 lines
  
✓ scenarios/__init__.py (UPDATE)
  → Add new test scenario factories to registry
```

### Files to Remove (Optional/Low Priority)
```
? arz_model/calibration/core/network_builder.py
  → Remove export_network() method (dead code)
  → Lines to delete: ~30
```

---

## 🧪 VALIDATION CHECKLIST

After cleanup, verify:

- [ ] All 13 Phase 6 tests still pass
- [ ] All 4 new integration tests still pass (NetworkBuilder → NetworkGrid)
- [ ] NetworkSimulator still works with from_yaml_config()
- [ ] Test networks use scenarios instead of manual creation
- [ ] No hardcoded network patterns remain in tests
- [ ] Deprecation warnings appear for single-segment BC
- [ ] All legacy patterns documented

---

## 📈 EXPECTED OUTCOMES

### Code Quality
```
BEFORE:
- 95+ lines manual network creation in tests
- 150 lines in _build_network_from_config()
- 30 lines dead code (export_network TODO)
- No consistency in test network creation
Total Legacy: ~275 lines

AFTER:
- 0 lines manual creation (all scenarios)
- Simple 5-line factory method call
- No dead code
- All tests use consistent pattern
Total Legacy: ~0 lines
Reduction: 275 lines → 0 lines (100% cleanup)
```

### Architecture Clarity
```
OLD PATTERNS (4 ways to create networks):
1. NetworkGrid(params) + manual _build_network_from_config()
2. Manual segment/node creation in tests
3. Single-segment BC legacy code
4. Network export TODOs

NEW UNIFIED PATTERN (2 ways):
1. NetworkGrid.from_yaml_config() - For manual YAML editing
2. NetworkGrid.from_network_builder() - For programmatic/calibration
   → Both use scenarios factory pattern for testing
```

---

## 🚀 IMPLEMENTATION PHASES

### Phase 1: HIGH Priority (RECOMMEND FIRST)
**Estimated Time**: 2-3 hours
**Files to Touch**: 5 core + 3 test modules
**Risk**: Low (tests only, core stays same)
**Benefit**: Clean, maintainable test code + documented deprecation

**Tasks**:
1. Create 3 new scenario factories (test networks)
2. Update 3 test modules to use scenarios
3. Fix NetworkSimulator._build_network_from_config()
4. Run validation suite (13+4=17 tests should pass)

### Phase 2: MEDIUM Priority (AFTER Phase 1)
**Estimated Time**: 1-2 hours
**Files to Touch**: 2 files + 1 new doc
**Risk**: Very low (warnings + doc only)
**Benefit**: Team clarity, clear deprecation timeline

**Tasks**:
1. Add deprecation warning to single-segment BC
2. Create DATA_FORMAT_REFERENCE.md
3. Create _deprecated/ documentation
4. Validate no functionality change

### Phase 3: LOW Priority (OPTIONAL/POLISH)
**Estimated Time**: 0.5-1 hour
**Files to Touch**: 1 file (network_builder.py)
**Risk**: Minimal (removing unused code)
**Benefit**: Cleaner codebase

**Tasks**:
1. Remove export_network() dead code
2. Remove TODO comments
3. Quick test to verify no impact

---

## 📝 DECISION MATRIX

| Legacy Pattern | Category | Priority | Action | Risk | Benefit |
|---|---|---|---|---|---|
| Manual test networks | Code Quality | HIGH | Migrate to scenarios | Low | High |
| NetworkSimulator._build_network_from_config() | Architecture | HIGH | Replace with factory | Low | High |
| Single-segment BC | Backward Compat | MEDIUM | Deprecate + warn | Very Low | Medium |
| CSV manual columns | Data Format | LOW | Document only | None | Low |
| network_builder.py export_network() | Dead Code | LOW | Remove | Minimal | Medium |

---

## 🎯 SUCCESS CRITERIA

✅ **Project Complete When**:
1. All manual network creation removed from tests
2. All test networks use scenarios factory pattern
3. NetworkSimulator uses factory methods (no manual building)
4. All Phase 6 + Phase 7 tests still pass
5. Deprecation warnings visible for legacy single-segment BC
6. Documentation updated with cleanup reference
7. Zero legacy manual patterns remain in core code
8. All 439 Python files follow unified architecture

---

## 📚 Reference Files

**Current Architecture**:
- `DIRECT_INTEGRATION_COMPLETE.md` - The new unified pattern
- `scenarios/README.md` - How to add scenarios
- `scenarios/lagos_victoria_island.py` - Example production scenario

**Phase 6 Foundation**:
- `arz_model/config/network_config.py` - YAML config system
- `arz_model/core/parameter_manager.py` - Unified parameter management
- `arz_model/network/network_grid.py` - Main network execution

---

**Status**: AUDIT COMPLETE ✅  
**Recommendation**: Start with Phase 1 (HIGH priority) for maximum impact  
**Estimated Total Time**: 4-6 hours for all 3 phases  
**ROI**: Eliminate 275+ lines legacy code, unified architecture across all modules  

