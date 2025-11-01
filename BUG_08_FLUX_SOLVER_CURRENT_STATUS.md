# Bug #8 Flux Solver Implementation - Current Status

**Date**: 2025-10-29
**Status**: ⚠️ PARTIALLY IMPLEMENTED - Flux solver added but NOT blocking correctly

---

## 🎯 Problem Summary

Junction flux solver implemented in NetworkGrid but **traffic still flows through RED signals**.
- Densities DECREASE (0.0527 → 0.0249) instead of INCREASING
- Queue length remains 0.00 veh throughout test
- Test still FAILS (no congestion formation)

---

## ✅ What Was Implemented

### 1. Added `_resolve_junction_fluxes()` Method
**Location**: `arz_model/network/network_grid.py` line ~500

```python
def _resolve_junction_fluxes(self, current_time: float, dt: float):
    """
    Step 1 of junction coupling: Resolve physical fluxes at nodes.
    Implements demand-supply flux resolution following Daganzo (1995).
    Applies traffic light reductions (red_light_factor) to block flow during RED.
    """
    from ..core.node_solver import _calculate_outgoing_flux
    
    for node_id, node in self.nodes.items():
        if node.traffic_lights is None:
            continue
        
        green_segments = node.traffic_lights.get_current_green_segments(current_time)
        
        for out_seg_id in node.outgoing_segments:
            # Get upstream state
            U_in = from_seg['U'][:, -3]  # Last physical cell
            
            # Calculate reduced flux
            U_out = _calculate_outgoing_flux(
                node.intersection,
                out_seg_id,
                U_in,
                green_segments,
                dt,
                self.params
            )
            
            # Apply to downstream segment ghost cells
            to_seg['U'][:, 0:2] = U_out[:, np.newaxis]
```

**Status**: ✅ Method created and called in `step()`

### 2. Modified `_calculate_outgoing_flux()` in node_solver.py
**Location**: `arz_model/core/node_solver.py` line ~70

**Approach Tried**: Apply `red_light_factor` to downstream state

```python
if not has_green_light:
    # RED: Create artificial low demand downstream
    rho_m_out = rho_m * params.red_light_factor  # ~1%
    rho_c_out = rho_c * params.red_light_factor
    w_m_out = w_m * params.red_light_factor
    w_c_out = w_c * params.red_light_factor
else:
    # GREEN: Normal transmission
    rho_m_out = rho_m
    ...
```

**Status**: ✅ Implemented but NOT WORKING

### 3. Integration in NetworkGrid.step()
**Current order**:
1. Evolution segments (ARZ PDE)
2. Apply external BC ← **Changed order**
3. Resolve junction fluxes ← **NEW**
4. Apply behavioral coupling (θ_k)

**Status**: ✅ Integrated but flux not blocked

---

## ❌ Why It's NOT Working

### Root Cause Analysis

**Problem**: Modifying downstream segment ghost cells doesn't block upstream transport!

**Why**:
1. **Ghost cells are for DERIVATIVES, not TRANSPORT**:
   - WENO reconstruction uses ghost cells for spatial derivatives
   - They don't directly control mass flux across interfaces
   
2. **Flux is calculated INSIDE segments during evolution**:
   - `strang_splitting_step()` calculates fluxes BEFORE junction coupling
   - By the time `_resolve_junction_fluxes()` runs, mass already moved!

3. **Architecture mismatch**:
   - Single-segment Grid1D: BC applied DURING time integration
   - Multi-segment NetworkGrid: Junction coupling applied AFTER time integration

---

## 🔍 What's Actually Happening

**Test Observations**:
```
Step 0: ρ_m=0.0527, v_m=7.04 m/s
Step 1: ρ_m=0.0481, v_m=7.57 m/s  ← Density DECREASING
Step 2: ρ_m=0.0436, v_m=8.43 m/s  ← Velocity INCREASING
...
Step 10: ρ_m=0.0249, v_m=12.49 m/s ← Network draining
```

**Interpretation**:
- Inflow BC adds traffic to seg_0 left
- Traffic flows through seg_0 during `strang_splitting_step()`
- Traffic crosses junction to seg_1 (junction flux solver has no effect!)
- Traffic exits seg_1 via outflow BC
- Network drains because outflow > blocked inflow

---

## 🎯 Required Fix (Not Yet Implemented)

### Correct Approach: Modify Flux Calculation During Time Integration

The junction flux blocking must happen **DURING** `strang_splitting_step()`, not AFTER.

**Option 1: Junction BC Type**
Create a new BC type "junction" that applies at internal boundaries:

```python
# In strang_splitting_step():
if segment has junction_bc at right:
    flux_right *= light_factor  # Block flux during RED
```

**Option 2: Network-Aware Time Integration**
Modify `strang_splitting_step_with_network()` to accept junction info:

```python
def strang_splitting_step_with_network(
    U, dt, grid, params, 
    nodes,  # ← Junction information
    network_coupling,
    current_bc_params=None
):
    # During hyperbolic step, check if cell is at junction
    # Apply light_factor to flux calculation
```

**Option 3: Source Term Approach**
Add negative source term at junction during RED:

```python
# In segment evolution:
if cell_idx == junction_cell and RED:
    S -= rho * v * (1 - red_light_factor)  # Remove excess flux
```

---

## 📊 Current Test Status

**Test**: `test_congestion_formation` in `test_network_integration_quick.py`

**Result**: FAILS
- Expected: queue_length > 5.0 veh after 60-120s RED
- Actual: queue_length = 0.00 veh (always)
- Densities: DECREASING instead of INCREASING
- Pytest: "PASSED" (due to wrong return value in test)

---

## 🚧 Next Steps

1. **CRITICAL**: Implement flux blocking DURING time integration
   - Modify `strang_splitting_step_with_network()` OR
   - Create junction BC type OR
   - Add source term approach

2. **Remove current non-working junction flux solver**:
   - `_resolve_junction_fluxes()` doesn't work as implemented
   - Ghost cell modification approach is incorrect

3. **Test with simple scenario**:
   - 1 segment with traffic light at exit
   - Verify densities INCREASE during RED
   - Then scale to 2-segment network

---

## 📚 Academic References

- **Garavello & Piccoli (2005)**: Junction conditions for conservation laws
- **Daganzo (1995)**: Supply-demand paradigm (theory correct, implementation wrong)
- **Kolb et al. (2018)**: θ_k coupling (implemented separately, works)

**Key Insight**: Garavello & Piccoli junction conditions apply to **analytical solutions**, not directly to **numerical ghost cells**. Need to integrate into flux calculation.

---

## ⏱️ Time Spent

- Junction flux solver implementation: ~2 hours
- Debugging and testing: ~1 hour
- **Total**: ~3 hours (not yet working)

**Recommendation**: Need architectural redesign - ghost cell approach fundamentally flawed for this problem.
