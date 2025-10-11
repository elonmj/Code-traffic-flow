# Bug #17: Initial Conditions Units Mismatch (1000x Error)

**Status:** IDENTIFIED - Fix implementation in progress  
**Severity:** CRITICAL - Causes immediate CFL instability and simulation crash  
**Date Discovered:** 2025-01-30  
**Iteration:** Cycle 3 (kernel dyrm analysis)  
**Related Bugs:** Bug #16 (BC units mismatch) - identical pattern  

---

## Symptom

After successfully fixing Bug #16 (BC units mismatch), kernel dyrm crashed immediately with:

```
CFL Check (GPU): Extremely large max_abs_lambda detected (1.1400e+06 m/s), stopping simulation.
```

- **Error location**: Step 0 (initialization phase)
- **max_abs_lambda value**: 1.14 × 10⁶ m/s (physically impossible - sound speed ~340 m/s)
- **Impact**: Both training and control simulation crashed
- **Runtime**: Only 173s (vs expected 15+ min for quick test)

---

## Evidence

### 1. CFL Violation in Kernel dyrm Log

```
validation_output/results/elonmj_arz-validation-76rlperformance-dyrm/arz-validation-76rlperformance-dyrm.log
```

Lines showing error pattern:
```
📊 Training initiated: 100 total timesteps
  Training SimulationRunner initialized on device=cuda
CFL Check (GPU): Extremely large max_abs_lambda detected (1.1400e+06 m/s), stopping simulation.
```

Occurred in both:
- Training initialization (line ~450)
- Control simulation baseline (line ~680)

### 2. Correct YAML Format (Post Bug #16 Fix)

```yaml
initial_conditions:
  type: uniform
  state: [40.0, 15.0, 50.0, 13.0]  # veh/km format ✅

boundary_conditions:
  left:
    type: inflow
    state: [120.0, 8.0, 150.0, 6.0]  # veh/km format ✅
```

Configuration documentation:
```
Initial=40.0/50.0 veh/km, Inflow=120.0/150.0 veh/km
Initial velocity: w_m=15.0 m/s, w_c=13.0 m/s
Inflow velocity: w_m=8.0 m/s, w_c=6.0 m/s
```

### 3. Function Expectations

**File:** `arz_model/simulation/initial_conditions.py` lines 7-18

```python
def uniform_state(grid: Grid1D, rho_m_val: float, w_m_val: float, 
                  rho_c_val: float, w_c_val: float) -> np.ndarray:
    """
    Creates a uniform initial state across the entire grid (including ghost cells).

    Args:
        grid (Grid1D): The grid object.
        rho_m_val (float): Uniform density for motorcycles (veh/m).  # ← EXPECTS SI!
        w_m_val (float): Uniform Lagrangian variable for motorcycles (m/s).
        rho_c_val (float): Uniform density for cars (veh/m).  # ← EXPECTS SI!
        w_c_val (float): Uniform Lagrangian variable for cars (m/s).
    """
```

Docstring explicitly states densities must be in **veh/m** (SI units).

### 4. Missing Conversion in ModelParameters

**File:** `arz_model/core/parameters.py`

**Line 152 - Initial Conditions (NO CONVERSION):**
```python
self.initial_conditions = config.get('initial_conditions', {})
```

**Lines 154-172 - Boundary Conditions (WITH CONVERSION):**
```python
raw_boundary_conditions = config.get('boundary_conditions', {})
self.boundary_conditions = {}
for boundary_side, bc_config in raw_boundary_conditions.items():
    processed_bc_config = copy.deepcopy(bc_config)
    if processed_bc_config.get('type', '').lower() == 'inflow':
        state = processed_bc_config.get('state')
        if state is not None and len(state) == 4:
            # Convert state values from [veh/km, km/h, veh/km, km/h] to [veh/m, m/s, veh/m, m/s]
            processed_bc_config['state'] = [
                state[0] * VEH_KM_TO_VEH_M, # rho_m
                state[1] * KMH_TO_MS,       # w_m
                state[2] * VEH_KM_TO_VEH_M, # rho_c
                state[3] * KMH_TO_MS        # w_c
            ]
    self.boundary_conditions[boundary_side] = processed_bc_config
```

**Conclusion:** BC gets explicit unit conversion, IC does not!

### 5. Direct Pass-Through in SimulationRunner

**File:** `arz_model/simulation/runner.py` lines 292-293

```python
if ic_type == 'uniform':
    state_vals = ic_config.get('state')  # [40.0, 15.0, 50.0, 13.0] from YAML
    U_init = initial_conditions.uniform_state(self.grid, *state_vals)
    # ↑ Passes veh/km directly to function expecting veh/m!
```

No conversion between getting `state_vals` from ModelParameters and passing to `uniform_state()`.

### 6. Proof of Correct Pattern in uniform_equilibrium IC

**File:** `arz_model/simulation/runner.py` lines 300-310

```python
elif ic_type == 'uniform_equilibrium':
    rho_m = ic_config.get('rho_m')
    rho_c = ic_config.get('rho_c')
    R_val = ic_config.get('R_val')
    
    # Convert densities from veh/km (config) to veh/m (SI units)
    rho_m_si = rho_m * VEH_KM_TO_VEH_M  # ← Explicit comment!
    rho_c_si = rho_c * VEH_KM_TO_VEH_M
    
    U_init, eq_state = initial_conditions.uniform_state_from_equilibrium(
        self.grid, rho_m_si, rho_c_si, R_val, self.params
    )
```

**This proves:**
- IC values in YAML are expected to be in veh/km
- Must be converted to SI before passing to initial_conditions functions
- Comment explicitly states the conversion intent

---

## Root Cause Analysis (5 Whys)

**Why #1:** Why did the simulation crash with CFL violation?  
→ max_abs_lambda (eigenvalue) was 1.14 × 10⁶ m/s (impossible value)

**Why #2:** Why was the eigenvalue so large?  
→ ARZ eigenvalue calculation depends on density. High density → high eigenvalue.

**Why #3:** Why was the density so high?  
→ Initial condition densities were 1000x higher than intended:
- YAML: `[40.0, 15.0, 50.0, 13.0]` (veh/km)
- Interpreted as: `40.0 veh/m = 40,000 veh/km`

**Why #4:** Why were IC densities interpreted as 1000x higher?  
→ `uniform_state()` expects veh/m but received veh/km values directly from YAML without conversion.

**Why #5:** Why wasn't the conversion applied?  
→ **ROOT CAUSE:** `ModelParameters.__init__` converts BC state values (lines 154-172) but **does NOT convert IC state values** (line 152 just copies dict directly).

---

## Relationship to Bug #16

Bug #16 and Bug #17 are **twin bugs** with identical pattern:

| Aspect | Bug #16 (BC) | Bug #17 (IC) |
|--------|--------------|--------------|
| **Location** | `test_section_7_6_rl_performance.py` lines 157-177 | `parameters.py` line 152 |
| **Issue** | Test script pre-converted BC to SI | ModelParameters doesn't convert IC |
| **Symptom** | Domain drainage (1000x too low inflow) | CFL violation (1000x too high IC) |
| **Discovery** | Kernel nmpy, iteration 1 | Kernel dyrm, iteration 3 |
| **Resolution** | Removed premature conversion in test script | Need to add conversion in ModelParameters |

**Why Bug #17 was masked by Bug #16:**

Before Bug #16 fix:
- BC was 1000x too low (0.0001 veh/m) → domain drained to vacuum
- IC being 1000x too high didn't matter because BC starvation dominated
- Simulation failed slowly over time as domain emptied

After Bug #16 fix:
- BC now correct (0.12/0.15 veh/m) → healthy inflow maintained
- IC still 1000x too high (40/50 veh/m instead of 0.04/0.05)
- **Immediate eigenvalue explosion at initialization (step 0)**

---

## Solution

### Strategy

Apply the same unit conversion logic to `initial_conditions` that is already implemented for `boundary_conditions` in `parameters.py` lines 154-172.

### Implementation Plan

**File:** `arz_model/core/parameters.py`

**Location:** After line 152 (where `self.initial_conditions` is assigned)

**Conversion needed for:**

1. **`uniform` IC type:**
   - Convert `state` array: `[rho_m, w_m, rho_c, w_c]`
   - Densities: `× VEH_KM_TO_VEH_M`
   - Velocities: Already in m/s (no conversion)

2. **`riemann` IC type:**
   - Convert `U_L` array: `[rho_m, w_m, rho_c, w_c]`
   - Convert `U_R` array: `[rho_m, w_m, rho_c, w_c]`
   - Same pattern as uniform

3. **`uniform_equilibrium` IC type:**
   - Already handled correctly in `runner.py` lines 307-309
   - No change needed in `parameters.py`

4. **Other IC types** (`density_hump`, `sine_wave_perturbation`):
   - Check if they have state arrays needing conversion

### Code Pattern (Mirroring BC Conversion)

```python
# Load initial conditions and perform unit conversion for state arrays
raw_initial_conditions = config.get('initial_conditions', {})
self.initial_conditions = copy.deepcopy(raw_initial_conditions)

ic_type = self.initial_conditions.get('type', '').lower()

if ic_type == 'uniform':
    state = self.initial_conditions.get('state')
    if state is not None and len(state) == 4:
        # Convert from [veh/km, m/s, veh/km, m/s] to [veh/m, m/s, veh/m, m/s]
        self.initial_conditions['state'] = [
            state[0] * VEH_KM_TO_VEH_M,  # rho_m
            state[1],                     # w_m (already m/s)
            state[2] * VEH_KM_TO_VEH_M,  # rho_c
            state[3]                      # w_c (already m/s)
        ]

elif ic_type == 'riemann':
    U_L = self.initial_conditions.get('U_L')
    if U_L is not None and len(U_L) == 4:
        self.initial_conditions['U_L'] = [
            U_L[0] * VEH_KM_TO_VEH_M,  # rho_m
            U_L[1],                     # w_m
            U_L[2] * VEH_KM_TO_VEH_M,  # rho_c
            U_L[3]                      # w_c
        ]
    
    U_R = self.initial_conditions.get('U_R')
    if U_R is not None and len(U_R) == 4:
        self.initial_conditions['U_R'] = [
            U_R[0] * VEH_KM_TO_VEH_M,
            U_R[1],
            U_R[2] * VEH_KM_TO_VEH_M,
            U_R[3]
        ]
```

### Validation Checklist

After applying fix, verify in kernel logs:

- [ ] No CFL violations at step 0
- [ ] Simulation runs to completion (100+ timesteps)
- [ ] Domain maintains healthy densities (not draining, not exploding)
- [ ] Eigenvalues are physically reasonable (<100 m/s)
- [ ] Initial state shows correct SI values:
  - `rho_m = 0.04 veh/m` (not 40.0)
  - `rho_c = 0.05 veh/m` (not 50.0)

---

## Expected Impact

**Before Fix:**
```
YAML IC: [40.0, 15.0, 50.0, 13.0]  (veh/km format)
       ↓ (no conversion)
uniform_state() receives: [40.0, 15.0, 50.0, 13.0]
       ↓ (interpreted as veh/m)
Actual densities: 40.0 veh/m = 40,000 veh/km  ❌ (1000x too high!)
       ↓
Eigenvalue calculation: max_abs_lambda = 1.14e+06 m/s  ❌
       ↓
CFL Check: CRASH at step 0  ❌
```

**After Fix:**
```
YAML IC: [40.0, 15.0, 50.0, 13.0]  (veh/km format)
       ↓ (ModelParameters conversion)
self.initial_conditions['state']: [0.04, 15.0, 0.05, 13.0]  (SI units)
       ↓
uniform_state() receives: [0.04, 15.0, 0.05, 13.0]  (veh/m)
       ↓ (correct interpretation)
Actual densities: 0.04 veh/m = 40 veh/km  ✅
       ↓
Eigenvalue calculation: max_abs_lambda ~10-20 m/s  ✅
       ↓
CFL Check: PASS, simulation continues  ✅
```

---

## Files to Modify

1. **`arz_model/core/parameters.py`:**
   - Add IC unit conversion after line 152
   - Handle `uniform` and `riemann` IC types
   - Mirror BC conversion pattern (lines 154-172)

2. **Test validation:**
   - Run kernel with quick test flag
   - Monitor logs for CFL violations
   - Verify healthy simulation execution

---

## Commit Message Template

```
Fix Bug #17: Initial Conditions Units Mismatch (1000x Error)

BUG #17 ROOT CAUSE:
ModelParameters class converts boundary condition state values from
veh/km to veh/m but does NOT apply the same conversion to initial
condition state values. This causes uniform_state() to interpret IC
densities as 1000x higher than intended, leading to catastrophic
eigenvalue explosion and immediate CFL violation at step 0.

EVIDENCE:
- parameters.py line 152: IC just copied without conversion
- parameters.py lines 154-172: BC explicitly converted
- initial_conditions.py docstring: uniform_state() expects veh/m
- runner.py lines 307-309: uniform_equilibrium IC has correct conversion
- kernel dyrm: max_abs_lambda = 1.14e+06 m/s (impossible value)

SOLUTION:
Apply same unit conversion logic to initial_conditions as boundary_conditions:
- uniform IC: Convert state array [rho_m, w_m, rho_c, w_c]
- riemann IC: Convert U_L and U_R arrays
- Densities: × VEH_KM_TO_VEH_M (0.001)
- Velocities: Already in m/s (no conversion)

RELATIONSHIP TO BUG #16:
Twin bugs with identical pattern:
- Bug #16: Test script pre-converted BC (removed in commit 01861ef)
- Bug #17: ModelParameters doesn't convert IC (fixed in this commit)

Bug #16 fix exposed Bug #17:
- Before: BC 1000x too low masked IC being 1000x too high
- After: Correct BC + wrong IC → immediate eigenvalue explosion

EXPECTED OUTCOME:
- IC densities: 0.04/0.05 veh/m (not 40/50)
- Eigenvalues: ~10-20 m/s (not 1.14e+06)
- CFL checks: PASS at initialization
- Simulation: Runs to completion without crashes

Related: Bug #16 (commit 01861ef)
```

---

## Notes

- This is the **second units conversion bug** discovered in the validation cycle
- Both bugs had **identical root cause pattern**: inconsistent handling of veh/km → veh/m conversion
- Bug #17 was **masked** by Bug #16 until BC fix was applied
- The fix follows **established pattern** from BC conversion (proven correct)
- After this fix, **all state values** (BC and IC) will have consistent units handling

**Lesson:** When fixing units bugs, check **all** code paths that handle similar data (BC, IC, schedules, etc.)
