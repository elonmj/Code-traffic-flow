# ARZ Model: Architectural Fix for Boundary Conditions

## Problem Statement

The current architecture has a **fundamental design flaw**: Initial Conditions (IC) and Boundary Conditions (BC) are entangled.

### Current (Broken) Architecture:

```
IC (t=0) → initial_equilibrium_state → BC (all t)
```

**Flow:**
1. User specifies initial conditions (IC) in YAML
2. System calculates `initial_equilibrium_state` from IC
3. System uses `initial_equilibrium_state` as FALLBACK for BC
4. **Result**: BC are ALWAYS derived from IC, ignoring user's BC config

### Why This Is Architecturally Wrong:

| Concept | Purpose | Should Be |
|---------|---------|-----------|
| **Initial Conditions** | Domain state at t=0 | Independent: "How does domain start?" |
| **Boundary Conditions** | Flux at domain edges for all t | Independent: "What enters/exits continuously?" |
| **Current System** | IC → BC coupling | **WRONG**: BC should not depend on IC |

**Example showing the bug:**
```yaml
initial_conditions:
  type: uniform
  state: [0.02, 10.0, 0.02, 10.0]  # Light traffic initially

boundary_conditions:
  left:
    type: inflow
    state: [0.150, 8.0, 0.120, 6.0]  # Heavy traffic entering
```

**What should happen:**
- t=0: Domain has light traffic (0.02 veh/m)
- t>0: Heavy traffic enters (0.150 veh/m)
- Result: Wave of congestion propagates through domain

**What actually happens:**
- System ignores BC state: `[0.150, 8.0, 0.120, 6.0]`
- Uses IC-derived fallback: `[0.02, 10.0, 0.02, 10.0]`
- Result: Light traffic forever, no congestion

---

## Root Cause Analysis

### File: `arz_model/simulation/runner.py`

#### Problem 1: IC stores equilibrium state (lines 361-410)

```python
def _create_initial_state(self) -> np.ndarray:
    # ...
    self.initial_equilibrium_state = None  # BAD: Mixing IC with BC
    
    if ic_type == 'uniform':
        # ...
        self.initial_equilibrium_state = state_vals  # STORES IC state
```

**Issue**: The IC calculation stores a "equilibrium state" that will be used for BC.

#### Problem 2: BC initialization uses IC fallback (lines 454-463)

```python
def _initialize_boundary_conditions(self):
    # ...
    if self.initial_equilibrium_state is not None:  # BAD: IC → BC coupling
        if self.current_bc_params.get('left', {}).get('type') == 'inflow':
            if 'state' not in self.current_bc_params['left']:
                # BUG: Uses IC-derived state for BC!
                self.current_bc_params['left']['state'] = self.initial_equilibrium_state
```

**Issue**: If user doesn't specify BC `state`, system uses IC-derived `initial_equilibrium_state`.

#### Problem 3: Traffic signal control uses IC fallback (lines 806-811)

```python
def set_traffic_signal_state(self, ...):
    # Fallback to initial_equilibrium_state if traffic_signal_base_state not set
    base_state = (self.traffic_signal_base_state 
                  if hasattr(self, 'traffic_signal_base_state') 
                  else self.initial_equilibrium_state)  # BAD: IC fallback
```

**Issue**: Traffic lights modulate BC, but fall back to IC if BC not configured.

---

## Architectural Fix: Complete Separation

### New Architecture:

```
IC (t=0):  User → _create_initial_state() → U_init
           ↓ (NO COUPLING)
BC (all t): User → _initialize_boundary_conditions() → bc_state
```

**Principles:**
1. ✅ IC and BC are **completely independent**
2. ✅ BC **MUST** be explicitly configured (no IC fallback)
3. ✅ If BC missing → **ERROR** (fail fast, not silent fallback)
4. ✅ Traffic signal base state comes from **BC config**, never IC

---

## Implementation Plan

### Phase 1: Remove IC→BC Coupling (CRITICAL)

**File: `arz_model/simulation/runner.py`**

**Change 1:** Remove `initial_equilibrium_state` attribute entirely
```python
# BEFORE (lines 361-410)
def _create_initial_state(self) -> np.ndarray:
    self.initial_equilibrium_state = None  # ❌ DELETE THIS
    # ... IC logic ...
    self.initial_equilibrium_state = state_vals  # ❌ DELETE ALL ASSIGNMENTS

# AFTER
def _create_initial_state(self) -> np.ndarray:
    # Just create and return U_init
    # NO storage of any "equilibrium state"
    # IC is for t=0 ONLY
```

**Change 2:** BC initialization must parse YAML directly
```python
# BEFORE (lines 454-463)
def _initialize_boundary_conditions(self):
    if self.initial_equilibrium_state is not None:  # ❌ DELETE
        if 'state' not in self.current_bc_params['left']:
            self.current_bc_params['left']['state'] = self.initial_equilibrium_state  # ❌ DELETE

# AFTER
def _initialize_boundary_conditions(self):
    # Parse BC from YAML config ONLY
    # NO fallback to IC
    
    # Left BC
    left_bc = self.current_bc_params.get('left', {})
    if left_bc.get('type') == 'inflow':
        if 'state' not in left_bc:
            raise ValueError(
                "Inflow BC requires 'state': [rho_m, w_m, rho_c, w_c]. "
                "Boundary conditions must be explicitly configured."
            )
        # Validate state format
        state = left_bc['state']
        if not isinstance(state, (list, tuple)) or len(state) != 4:
            raise ValueError(
                f"Inflow BC 'state' must be [rho_m, w_m, rho_c, w_c], got: {state}"
            )
    
    # Similar for right BC
```

**Change 3:** Traffic signal base state from BC config
```python
# BEFORE (lines 806-811)
def set_traffic_signal_state(self, ...):
    base_state = (self.traffic_signal_base_state 
                  if hasattr(self, 'traffic_signal_base_state') 
                  else self.initial_equilibrium_state)  # ❌ DELETE IC FALLBACK

# AFTER
def set_traffic_signal_state(self, ...):
    # Base state MUST come from BC configuration
    if not hasattr(self, 'traffic_signal_base_state') or self.traffic_signal_base_state is None:
        raise RuntimeError(
            "Traffic signal requires 'traffic_signal_base_state' from BC configuration. "
            "Ensure boundary_conditions.left.state is specified for inflow BC."
        )
    base_state = self.traffic_signal_base_state
```

---

### Phase 2: Improve BC Configuration (ENHANCEMENT)

**Support multiple BC state formats:**

```yaml
# Format 1: Explicit state vector (current, keep)
boundary_conditions:
  left:
    type: inflow
    state: [0.150, 8.0, 0.120, 6.0]

# Format 2: Named parameters (NEW, more readable)
boundary_conditions:
  left:
    type: inflow
    rho_m: 0.150  # veh/m
    w_m: 8.0      # m/s
    rho_c: 0.120  # veh/m
    w_c: 6.0      # m/s

# Format 3: Equilibrium-based (NEW, for convenience)
boundary_conditions:
  left:
    type: inflow
    rho_m: 0.150  # veh/m
    rho_c: 0.120  # veh/m
    R_val: 2      # Road quality
    # System calculates equilibrium w_m, w_c
```

**Implementation:**

```python
def _parse_bc_state(self, bc_config: dict, bc_name: str) -> list:
    """
    Parse BC state from multiple possible formats.
    
    Returns: [rho_m, w_m, rho_c, w_c]
    """
    # Format 1: Explicit state vector
    if 'state' in bc_config:
        state = bc_config['state']
        if not isinstance(state, (list, tuple)) or len(state) != 4:
            raise ValueError(
                f"{bc_name} BC 'state' must be [rho_m, w_m, rho_c, w_c], got: {state}"
            )
        return list(state)
    
    # Format 2: Named parameters
    if all(k in bc_config for k in ['rho_m', 'w_m', 'rho_c', 'w_c']):
        return [
            bc_config['rho_m'],
            bc_config['w_m'],
            bc_config['rho_c'],
            bc_config['w_c']
        ]
    
    # Format 3: Equilibrium-based
    if all(k in bc_config for k in ['rho_m', 'rho_c', 'R_val']):
        from ..simulation.initial_conditions import _calculate_equilibrium_velocities
        rho_m = bc_config['rho_m']
        rho_c = bc_config['rho_c']
        R_val = bc_config['R_val']
        
        # Calculate equilibrium velocities
        w_m, w_c = _calculate_equilibrium_velocities(
            rho_m, rho_c, R_val, self.params
        )
        return [rho_m, w_m, rho_c, w_c]
    
    # No valid format found
    raise ValueError(
        f"{bc_name} inflow BC requires one of:\n"
        f"  1. 'state': [rho_m, w_m, rho_c, w_c]\n"
        f"  2. 'rho_m', 'w_m', 'rho_c', 'w_c' parameters\n"
        f"  3. 'rho_m', 'rho_c', 'R_val' for equilibrium-based"
    )
```

---

### Phase 3: Validation and Testing

**Add BC validation tests:**

```python
# tests/test_boundary_conditions_parsing.py

def test_bc_state_explicit():
    """Test explicit state vector format"""
    config = {'left': {'type': 'inflow', 'state': [0.15, 8.0, 0.12, 6.0]}}
    runner = SimulationRunner(...)
    assert runner.current_bc_params['left']['state'] == [0.15, 8.0, 0.12, 6.0]

def test_bc_state_named():
    """Test named parameters format"""
    config = {
        'left': {
            'type': 'inflow',
            'rho_m': 0.15,
            'w_m': 8.0,
            'rho_c': 0.12,
            'w_c': 6.0
        }
    }
    runner = SimulationRunner(...)
    assert runner.current_bc_params['left']['state'] == [0.15, 8.0, 0.12, 6.0]

def test_bc_state_missing_fails():
    """Test that missing BC state raises error"""
    config = {'left': {'type': 'inflow'}}  # Missing state
    with pytest.raises(ValueError, match="requires 'state'"):
        runner = SimulationRunner(...)

def test_bc_independent_of_ic():
    """Test that BC does not depend on IC"""
    config = {
        'initial_conditions': {
            'type': 'uniform',
            'state': [0.02, 10.0, 0.02, 10.0]  # Light IC
        },
        'boundary_conditions': {
            'left': {
                'type': 'inflow',
                'state': [0.15, 8.0, 0.12, 6.0]  # Heavy BC
            }
        }
    }
    runner = SimulationRunner(...)
    # IC should be light
    assert np.allclose(runner.U_current[0, :], 0.02, atol=0.01)
    # BC should be heavy (not derived from IC)
    assert runner.current_bc_params['left']['state'][0] == 0.15
```

---

## Migration Guide for Existing Configs

### Step 1: Identify configs using IC→BC coupling

```bash
# Find configs that rely on IC fallback
grep -r "boundary_conditions:" --include="*.yml" -A 5 | grep -B 5 "type: inflow" | grep -v "state:"
```

### Step 2: Add explicit BC state

**BEFORE (broken):**
```yaml
initial_conditions:
  type: uniform
  state: [0.02, 10.0, 0.02, 10.0]

boundary_conditions:
  left:
    type: inflow  # Silently uses IC state as fallback
```

**AFTER (fixed):**
```yaml
initial_conditions:
  type: uniform
  state: [0.02, 10.0, 0.02, 10.0]

boundary_conditions:
  left:
    type: inflow
    state: [0.15, 8.0, 0.12, 6.0]  # EXPLICIT BC state
```

### Step 3: Update all affected configs

**Affected files:**
- `Code_RL/configs/*.yml` - All RL scenarios
- `arz_model/config/scenarios/*.yml` - Test scenarios
- Any validation test configs in `validation_ch7*/`

---

## Benefits of Architectural Fix

### Before (Broken):
- ❌ BC silently fall back to IC
- ❌ Impossible to have IC ≠ BC
- ❌ Debugging nightmare (silent failures)
- ❌ RL training receives wrong traffic
- ❌ Validations may be invalid

### After (Fixed):
- ✅ BC must be explicitly configured
- ✅ IC and BC are independent
- ✅ Errors are explicit and loud
- ✅ RL training receives correct traffic
- ✅ Validations are trustworthy

---

## Implementation Checklist

### Critical Path (Must Do):

- [ ] Remove `initial_equilibrium_state` from `runner.py`
- [ ] Remove IC→BC fallback in `_initialize_boundary_conditions()`
- [ ] Add BC validation (fail if state missing)
- [ ] Fix traffic signal to use BC only
- [ ] Add explicit BC state to all RL configs
- [ ] Re-run congestion formation test
- [ ] Verify RL environment creates congestion

### Enhanced Features (Nice to Have):

- [ ] Support named BC parameters (`rho_m`, `w_m`, etc.)
- [ ] Support equilibrium-based BC (`rho_m`, `rho_c`, `R_val`)
- [ ] Add comprehensive BC validation tests
- [ ] Document BC configuration formats
- [ ] Add BC validation helper script

---

## Risk Assessment

### Breaking Changes:
- **HIGH**: Any config relying on IC→BC fallback will break
- **Mitigation**: Add explicit error messages guiding users to fix configs

### Testing Required:
- **Unit tests**: BC parsing with different formats
- **Integration tests**: Full simulation with explicit BC
- **Validation tests**: Re-run Section 7.3, 7.5, 7.6
- **RL tests**: Verify congestion formation in RL environment

### Rollback Plan:
- Keep `initial_equilibrium_state` as deprecated attribute
- Add warning instead of error initially
- Full removal after all configs migrated

---

## Timeline

**Week 1: Core Fix**
- Remove IC→BC coupling
- Add BC validation
- Fix critical RL configs

**Week 2: Enhanced Parsing**
- Support multiple BC formats
- Add comprehensive tests
- Document new architecture

**Week 3: Migration & Validation**
- Update all configs
- Re-run all validations
- Verify RL training works

---

## Conclusion

This is not a bug fix - it's an **architectural refactoring** that:

1. **Separates concerns**: IC (t=0) ≠ BC (all t)
2. **Explicit over implicit**: No silent fallbacks
3. **Fail fast**: Errors at config parsing, not runtime
4. **Correct by construction**: BC MUST be configured

**This fix will make ALL future work on this codebase more reliable.**

---

Date: 25 October 2024
Status: ARCHITECTURAL DESIGN - Ready for Implementation
Priority: CRITICAL - Blocks all RL training and some validations
