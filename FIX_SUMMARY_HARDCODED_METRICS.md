# Fix Summary: Hardcoded Metrics Bug

## Problem Statement
RED and GREEN traffic light control strategies were producing identical flow metrics (40.9 veh/km) despite being fundamentally different. The environment correctly applied the different control actions, but the metrics remained hardcoded/unchanged.

## Root Cause Analysis

### Layer 1: Initial Conditions Too Light ❌
The scenario used very light initial conditions (0.01 veh/m). With no congestion, even blocking inflow created no observable backpressure, making RED and GREEN indistinguishable.

**Solution**: Increased initial density from 0.01 → 0.259 veh/m (70% of jam density)

### Layer 2: Dynamic BC Parameters Lost During Integration ❌ 
Even with higher density, traffic signal control wasn't working because the dynamic boundary condition parameters were not reaching the spatial discretization functions.

**Problem Flow**:
1. `runner.run()` calls `strang_splitting_step()` with `current_bc_params`
2. `strang_splitting_step()` calls `solve_hyperbolic_step_ssprk3()` with `current_bc_params`
3. `solve_hyperbolic_step_ssprk3()` calls `calculate_spatial_discretization_weno()` WITHOUT passing `current_bc_params`
4. `calculate_spatial_discretization_weno()` calls `apply_boundary_conditions()` with `current_bc_params=None`
5. BC function falls back to static `params.boundary_conditions`, ignoring traffic signal changes

**Root Cause**: Missing parameter passing through the numerical integration chain.

## Solution Implemented

### File 1: `arz_model/numerics/time_integration.py`

#### Change 1: `calculate_spatial_discretization_weno()` function (line 24)
```python
# BEFORE:
def calculate_spatial_discretization_weno(U: np.ndarray, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    U_bc = np.copy(U)
    boundary_conditions.apply_boundary_conditions(U_bc, grid, params)  # ❌ Missing param

# AFTER:
def calculate_spatial_discretization_weno(U: np.ndarray, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None) -> np.ndarray:
    U_bc = np.copy(U)
    boundary_conditions.apply_boundary_conditions(U_bc, grid, params, current_bc_params)  # ✅ Passed
```

#### Change 2: `compute_flux_divergence_first_order()` function (line 578)
```python
# BEFORE:
def compute_flux_divergence_first_order(U: np.ndarray, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    U_bc = np.copy(U)
    boundary_conditions.apply_boundary_conditions(U_bc, grid, params)  # ❌ Missing param

# AFTER:
def compute_flux_divergence_first_order(U: np.ndarray, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None) -> np.ndarray:
    U_bc = np.copy(U)
    boundary_conditions.apply_boundary_conditions(U_bc, grid, params, current_bc_params)  # ✅ Passed
```

#### Change 3: `solve_hyperbolic_step_ssprk3()` function (line 477)
```python
# BEFORE:
def solve_hyperbolic_step_ssprk3(U_in: np.ndarray, dt_hyp: float, grid: Grid1D, params: ModelParameters) -> np.ndarray:
    if params.spatial_scheme == 'first_order':
        compute_L = lambda U: -compute_flux_divergence_first_order(U, grid, params)  # ❌ No param
    elif params.spatial_scheme == 'weno5':
        compute_L = lambda U: calculate_spatial_discretization_weno(U, grid, params)  # ❌ No param

# AFTER:
def solve_hyperbolic_step_ssprk3(U_in: np.ndarray, dt_hyp: float, grid: Grid1D, params: ModelParameters, current_bc_params: dict | None = None) -> np.ndarray:
    if params.spatial_scheme == 'first_order':
        compute_L = lambda U: -compute_flux_divergence_first_order(U, grid, params, current_bc_params)  # ✅ Passed
    elif params.spatial_scheme == 'weno5':
        compute_L = lambda U: calculate_spatial_discretization_weno(U, grid, params, current_bc_params)  # ✅ Passed
```

#### Change 4: `strang_splitting_step()` CPU path (line 450)
```python
# BEFORE:
U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params)  # ❌ No param

# AFTER:
U_ss = solve_hyperbolic_step_ssprk3(U_star, dt, grid, params, current_bc_params)  # ✅ Passed
```

### File 2: `Code_RL/src/utils/config.py`

#### Already Fixed in Previous Session
Initial density increased from 0.01 → 0.259 veh/m in `_generate_signalized_network_lagos()` function.

## Test Results

### Before Fix
```
RED_ONLY final density:   0.0098 veh/m
GREEN_ONLY final density: 0.0098 veh/m
Absolute difference: 0.0000 veh/m (Metrics hardcoded!)
```

### After Fix
```
RED_ONLY final density:   0.1067 veh/m
GREEN_ONLY final density: 0.1309 veh/m
Absolute difference: 0.0242 veh/m
Relative difference: 18.5%

✅ SUCCESS! Metrics now DIFFER based on traffic signal control!
```

## Verification

The fix is physically correct:

- **RED Phase** (blocks inflow, ρ_inflow=0): 
  - Vehicles cannot enter from upstream
  - Domain drains normally
  - **Result: Lower steady-state density (0.1067 veh/m)**

- **GREEN Phase** (allows inflow, ρ_inflow=normal):
  - Vehicles flow freely from upstream
  - Domain fills to higher steady state
  - **Result: Higher steady-state density (0.1309 veh/m)**

The 18.5% difference is substantial and demonstrates that traffic signal control now properly affects simulation dynamics.

## Files Modified

1. `arz_model/numerics/time_integration.py` (4 functions updated)
   - `calculate_spatial_discretization_weno()`: Added `current_bc_params` parameter and pass-through
   - `compute_flux_divergence_first_order()`: Added `current_bc_params` parameter and pass-through
   - `solve_hyperbolic_step_ssprk3()`: Added `current_bc_params` parameter and lambda updates
   - `strang_splitting_step()`: Updated CPU path to pass `current_bc_params`

## Impact

- ✅ Hardcoded metrics bug FIXED
- ✅ Traffic signal control now EFFECTIVE
- ✅ RED and GREEN produce MEASURABLY DIFFERENT results
- ✅ Physics-based behavior preserved
- ✅ No breaking changes to existing code
