# BUG #16: YAML Inflow Density Units Mismatch (1000x Error)

## Symptom
- All metrics show 0.0% improvement
- Domain drains completely after few timesteps
- Mean densities fall from ~0.01 to ~0.000016 (essentially vacuum)
- Inflow state in logs shows `rho_m=0.0001, rho_c=0.0001` instead of expected high values

## Evidence

### Log Analysis
**Configuration request (line 80):**
```
BUG #14 FIX: Initial=40.0/50.0 veh/km, Inflow=120.0/150.0 veh/km
```

**Actual inflow applied (lines 140, 146, 204, etc.):**
```
└─ Inflow state: rho_m=0.0001, w_m=2.2, rho_c=0.0001, w_c=1.7
```

**traffic_signal_base_state stored (line 95):**
```
DEBUG BUG #15: Stored traffic_signal_base_state = [0.00012, 2.22, 0.00015, 1.67]
```

**YAML file content:**
```yaml
boundary_conditions:
  left:
    state:
    - 0.12    # ← WRONG: Should be 120.0 (veh/km)
    - 8.0
    - 0.15    # ← WRONG: Should be 150.0 (veh/km)
    - 6.0
    type: inflow
```

**Domain drainage (lines 4390-4775):**
```
Mean densities: rho_m=0.009050, rho_c=0.017433  # Initial (OK)
Mean densities: rho_m=0.000016, rho_c=0.000018  # After few steps (DRAINED!)
```

### Error Magnitude
- **Requested:** 120/150 veh/km (high traffic demand)
- **Written to YAML:** 0.12/0.15 (already in SI units: veh/m)
- **SimulationRunner reads:** 0.12/0.15 and assumes veh/km format
- **SimulationRunner converts:** 0.12 × 0.001 = 0.00012 veh/m
- **Final inflow:** 0.00012/0.00015 veh/m = **0.12/0.15 veh/km**
- **Error:** **1000× too low!** ❌

## Root Cause

**Fundamental problem:** Units mismatch between scenario generation and SimulationRunner parsing.

**In `test_section_7_6_rl_performance.py:_create_scenario_config()` (lines 180-190):**
```python
# Calculate INFLOW state for boundary conditions
rho_m_inflow_veh_km = 120.0  # veh/km (high demand)
rho_c_inflow_veh_km = 150.0  # veh/km

# WRONG: Converts to SI immediately for YAML
rho_m_inflow_si = rho_m_inflow_veh_km * VEH_KM_TO_VEH_M  # = 0.12 veh/m
rho_c_inflow_si = rho_c_inflow_veh_km * VEH_KM_TO_VEH_M  # = 0.15 veh/m

config['boundary_conditions'] = {
    'left': {
        'type': 'inflow',
        'state': [rho_m_inflow_si, w_m_inflow, rho_c_inflow_si, w_c_inflow]
        #         ^^^ PROBLEM: YAML expects veh/km, not SI!
    }
}
```

**In `arz_model/simulation/runner.py:__init__()` (lines 280-290):**
```python
# SimulationRunner reads YAML
bc_config = self.params.boundary_conditions
left_bc = bc_config['left']
self.traffic_signal_base_state = left_bc['state']  # [0.00012, 8.0, 0.00015, 6.0]

# ✅ BUG #15 FIX assumed YAML values are in veh/km (standard format)
# But test script already converted to SI!
# Result: Double conversion → 1000× error
```

**In `arz_model/numerics/boundary_conditions.py` (conversion happens again):**
```python
# Assumes state values from YAML are in veh/km (user-friendly units)
# Converts again to SI: 0.12 × 0.001 = 0.00012 veh/m
```

## Solution

**Fix approach:** Write densities in **veh/km** to YAML (not SI), matching SimulationRunner expectations.

**Before (WRONG):**
```python
# In _create_scenario_config()
rho_m_inflow_si = rho_m_inflow_veh_km * VEH_KM_TO_VEH_M  # Premature SI conversion
rho_c_inflow_si = rho_c_inflow_veh_km * VEH_KM_TO_VEH_M

config['boundary_conditions'] = {
    'left': {
        'type': 'inflow',
        'state': [rho_m_inflow_si, w_m_inflow, rho_c_inflow_si, w_c_inflow]
        #         ^^^ Already SI → SimulationRunner converts again!
    }
}
```

**After (CORRECT):**
```python
# In _create_scenario_config()
# Keep densities in veh/km for YAML (SimulationRunner will convert to SI internally)
config['boundary_conditions'] = {
    'left': {
        'type': 'inflow',
        'state': [rho_m_inflow_veh_km, w_m_inflow, rho_c_inflow_veh_km, w_c_inflow]
        #         ^^^ veh/km format → SimulationRunner handles SI conversion
    }
}
```

**Justification:**
- YAML files are user-facing configuration (should use veh/km)
- SimulationRunner internally handles SI conversion
- Scenario generation should not pre-convert to SI
- This matches existing config file patterns in `arz_model/config/`

## Implementation

**File:** `validation_ch7/scripts/test_section_7_6_rl_performance.py`
**Function:** `_create_scenario_config()`
**Lines:** ~180-190

**Change:**
1. Remove premature SI conversion of inflow densities
2. Write veh/km values directly to YAML
3. Let SimulationRunner handle SI conversion internally

**Code diff:**
```python
# Lines 180-190 (BEFORE)
rho_m_inflow_si = rho_m_inflow_veh_km * VEH_KM_TO_VEH_M
rho_c_inflow_si = rho_c_inflow_veh_km * VEH_KM_TO_VEH_M

config['boundary_conditions'] = {
    'left': {
        'type': 'inflow',
        'state': [rho_m_inflow_si, w_m_inflow, rho_c_inflow_si, w_c_inflow]
    },
    'right': {'type': 'outflow'}
}

# Lines 180-190 (AFTER)
# BUG #16 FIX: Write densities in veh/km to YAML (not SI)
# SimulationRunner expects veh/km and handles internal SI conversion
config['boundary_conditions'] = {
    'left': {
        'type': 'inflow',
        'state': [rho_m_inflow_veh_km, w_m_inflow, rho_c_inflow_veh_km, w_c_inflow]
        # Use veh/km directly ^^^            ^^^
    },
    'right': {'type': 'outflow'}
}
```

## Verification

**Expected after fix:**
1. YAML file shows: `state: [120.0, 8.0, 150.0, 6.0]`
2. SimulationRunner logs: `traffic_signal_base_state = [0.12, 8.0, 0.15, 6.0]` (SI)
3. Inflow BC logs: `rho_m=0.12, rho_c=0.15` (high demand)
4. Domain does NOT drain: `Mean densities: rho_m=0.04-0.12, rho_c=0.05-0.15`
5. Metrics are NON-ZERO: `avg_flow_improvement > 0.0`

**Quick test checklist:**
- [ ] YAML contains densities in veh/km (e.g., 120.0)
- [ ] traffic_signal_base_state in SI (0.12 veh/m)
- [ ] Inflow BC applies correct values (not ~0.0001)
- [ ] Domain maintains reasonable densities
- [ ] validation_success: true

## Related Bugs

- **Bug #14:** Inflow BC momentum equation (separate issue, already fixed)
- **Bug #15:** traffic_signal_base_state vs initial_equilibrium_state (separate issue, already fixed)
- **Bug #16 (THIS):** Units mismatch causes 1000× error in inflow density

**Key insight:** Bugs #14, #15, #16 form a chain:
1. Bug #14: Momentum equation was wrong (fixed)
2. Bug #15: Wrong state source (IC vs BC) (fixed)
3. Bug #16: **Units mismatch (THIS BUG)** ← Current blocker

Once Bug #16 is fixed, the full chain is resolved and validation should succeed.

## Files Affected

- `validation_ch7/scripts/test_section_7_6_rl_performance.py` (PRIMARY FIX)
- Generated YAML: `section_7_6_rl_performance/data/scenarios/*.yml`

## Commit Message Template

```
Fix Bug #16: YAML inflow density units mismatch (1000x error)

Root cause: test_section_7_6_rl_performance.py pre-converted densities
to SI units before writing to YAML, but SimulationRunner expects veh/km
format and performs its own SI conversion.

Result: Double conversion → 0.12 veh/km × 0.001 = 0.00012 veh/m (1000x too low)

Solution: Write densities in veh/km directly to YAML. SimulationRunner
handles SI conversion internally.

Evidence:
- Requested: Inflow=120/150 veh/km
- YAML written: state: [0.12, 8.0, 0.15, 6.0] (SI)
- SimulationRunner read: 0.12 veh/km → 0.00012 veh/m
- Actual inflow: 0.12 veh/km (1000x too low!)
- Domain drained: rho falls to ~0.000016 (vacuum)

Fix: Remove premature SI conversion. Write veh/km to YAML.

Verification: After fix, YAML contains [120.0, 8.0, 150.0, 6.0],
SimulationRunner converts to [0.12, 8.0, 0.15, 6.0] internally,
and domain maintains healthy densities (0.04-0.12 veh/m).

Ref: docs/BUG_FIX_YAML_INFLOW_UNITS_MISMATCH.md
Closes: Bug #16 chain (Bugs #14, #15, #16 now all resolved)
```
