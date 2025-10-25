# Exact Code Changes - Hardcoded Metrics Fix

## File: `arz_model/numerics/time_integration.py`

### Change 1: Line 24 - `calculate_spatial_discretization_weno()`

**Function Signature**
```python
# BEFORE (3 parameters)
def calculate_spatial_discretization_weno(U: np.ndarray, grid: Grid1D, params: ModelParameters) -> np.ndarray:

# AFTER (4 parameters)
def calculate_spatial_discretization_weno(U: np.ndarray, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None) -> np.ndarray:
```

**Boundary Condition Application**
```python
# BEFORE
boundary_conditions.apply_boundary_conditions(U_bc, grid, params)

# AFTER
boundary_conditions.apply_boundary_conditions(U_bc, grid, params, current_bc_params)
```

---

### Change 2: Line 578 - `compute_flux_divergence_first_order()`

**Function Signature**
```python
# BEFORE (3 parameters)
def compute_flux_divergence_first_order(U: np.ndarray, grid: Grid1D, params: ModelParameters) -> np.ndarray:

# AFTER (4 parameters)
def compute_flux_divergence_first_order(U: np.ndarray, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None) -> np.ndarray:
```

**Boundary Condition Application**
```python
# BEFORE
boundary_conditions.apply_boundary_conditions(U, grid, params)

# AFTER
boundary_conditions.apply_boundary_conditions(U, grid, params, current_bc_params)
```

---

### Change 3a: Lines 477-486 - `solve_hyperbolic_step_ssprk3()` Function Signature

**Function Signature**
```python
# BEFORE (4 parameters)
def solve_hyperbolic_step_ssprk3(U_in: np.ndarray, dt_hyp: float, grid: Grid1D, params: ModelParameters) -> np.ndarray:

# AFTER (5 parameters)
def solve_hyperbolic_step_ssprk3(U_in: np.ndarray, dt_hyp: float, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None) -> np.ndarray:
```

---

### Change 3b: Lines 507-509 - Lambda Functions for Spatial Discretization

**First-Order Scheme Lambda**
```python
# BEFORE
if params.spatial_scheme == 'first_order':
    compute_L = lambda U: -compute_flux_divergence_first_order(U, grid, params)

# AFTER
if params.spatial_scheme == 'first_order':
    compute_L = lambda U: -compute_flux_divergence_first_order(U, grid, params, current_bc_params)
```

**WENO5 Scheme Lambda**
```python
# BEFORE
elif params.spatial_scheme == 'weno5':
    compute_L = lambda U: calculate_spatial_discretization_weno(U, grid, params)

# AFTER
elif params.spatial_scheme == 'weno5':
    compute_L = lambda U: calculate_spatial_discretization_weno(U, grid, params, current_bc_params)
```

---

### Change 4: Line 450 - `strang_splitting_step()` CPU Path Calls

**Context: CPU path in strang_splitting_step function**

The CPU path contains 4 calls to `solve_hyperbolic_step_ssprk3()`:

**Call 1: After calculating U_star (first hyperbolic step)**
```python
# BEFORE
U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params)

# AFTER
U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params, current_bc_params)
```

**Call 2: After first parabolic step**
```python
# BEFORE
U_hh = solve_hyperbolic_step_ssprk3(U_pp, dt, grid, params)

# AFTER
U_hh = solve_hyperbolic_step_ssprk3(U_pp, dt, grid, params, current_bc_params)
```

**Call 3: After second parabolic step**
```python
# BEFORE
U_ss = solve_hyperbolic_step_ssprk3(U_pp_2, dt, grid, params)

# AFTER
U_ss = solve_hyperbolic_step_ssprk3(U_pp_2, dt, grid, params, current_bc_params)
```

**Call 4: Final hyperbolic step**
```python
# BEFORE
U_out = solve_hyperbolic_step_ssprk3(U_pp_3, dt, grid, params)

# AFTER
U_out = solve_hyperbolic_step_ssprk3(U_pp_3, dt, grid, params, current_bc_params)
```

---

## Summary of Changes

### Modified Functions
| Function | Line | Parameter Added | BC Call Updated |
|----------|------|-----------------|-----------------|
| `calculate_spatial_discretization_weno()` | 24 | ✅ `current_bc_params: dict \| None = None` | ✅ Yes |
| `compute_flux_divergence_first_order()` | 578 | ✅ `current_bc_params: dict \| None = None` | ✅ Yes |
| `solve_hyperbolic_step_ssprk3()` | 477 | ✅ `current_bc_params: dict \| None = None` | ✅ Lambdas updated |
| `strang_splitting_step()` | 450 | N/A (existing param) | ✅ 4 calls updated |

### Total Changes
- **Functions modified**: 4
- **Function signatures changed**: 3 (added parameter)
- **Function calls updated**: 4 + 2 lambdas = 6 call sites
- **Parameter additions**: All with default `None` value (backward compatible)
- **Files affected**: 1

### Backward Compatibility
- ✅ All new parameters have default value `None`
- ✅ Existing code that doesn't pass the parameter will still work
- ✅ GPU path unaffected (already had parameter)
- ✅ No breaking changes

---

## Verification Evidence

### Debug Output Confirms Dynamic Path
```
[PERIODIC:1000] BC_DISPATCHER [Call #1000] current_bc_params: <class 'dict'>
[PERIODIC:1000] BC_DISPATCHER Using current_bc_params (dynamic)
[PERIODIC:1000] BC_DISPATCHER Left inflow: [0.0, 0.0, 0.0, 0.0]  <- RED phase
```

```
[PERIODIC:1000] BC_DISPATCHER [Call #3000] current_bc_params: <class 'dict'>
[PERIODIC:1000] BC_DISPATCHER Using current_bc_params (dynamic)
[PERIODIC:1000] BC_DISPATCHER Left inflow: [0.2, 2.469, 0.096, 2.160]  <- GREEN phase
```

### Test Results
- **RED phase**: 0.1067 veh/m (domain empties when inflow blocked)
- **GREEN phase**: 0.1309 veh/m (domain fills when inflow allowed)
- **Difference**: **18.5%** (significant and correct)
- **Status**: ✅ All tests PASSED

---

## Implementation Notes

1. **Default Value**: All new parameters have `dict | None = None` default, allowing backward compatibility
2. **Lambda Closure**: Lambda functions in `solve_hyperbolic_step_ssprk3()` capture `current_bc_params` in their closure
3. **Strang Splitting**: All 4 stages of Strang splitting now receive the dynamic BC parameter
4. **Both Schemes**: Both first-order and WENO5 spatial schemes support the parameter
5. **Physics**: Parameter passes boundary condition information from signal control through to numerical solver

---

## Deploy Checklist

- [x] All functions updated with correct parameter
- [x] All function calls updated to pass parameter
- [x] Parameter has default value (backward compatible)
- [x] No other files modified
- [x] GPU path unaffected
- [x] Comprehensive tests PASSED
- [x] Physics validation PASSED
- [x] Ready for production merge
