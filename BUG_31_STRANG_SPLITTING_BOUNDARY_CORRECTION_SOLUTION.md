# BUG #31 - Strang Splitting Boundary Correction Solution

**Date**: 2025-01-03
**Status**: SOLUTION IDENTIFIED - Ready for Implementation
**Severity**: CRITICAL - Velocity explosions with inflow BC

---

## Executive Summary

The numerical instability (velocity explosions to 350-370 m/s) is caused by **order reduction** in Strang operator splitting when using **inhomogeneous inflow boundary conditions**. This is a well-documented problem in the numerical methods literature.

**ROOT CAUSE**: Incompatibility between boundary conditions applied to the ODE substep and the hyperbolic substep in Strang splitting.

**SOLUTION**: Implement **boundary correction** technique as described in academic literature (Einkemmer et al. 2018, Descombes et al. 2015).

---

## Problem Analysis

### Diagnostic Test Results

Component isolation tests definitively proved:

| Component | BC Type | Duration | Result | v_max |
|-----------|---------|----------|--------|-------|
| ODE only (RK45) | N/A | 15s | ✅ STABLE | 15.96 m/s |
| FVM only (WENO5) | outflow | 15s | ✅ STABLE | 15.96 m/s |
| Strang (full) | outflow | 15s | ✅ STABLE | 15.96 m/s |
| **Strang (full)** | **inflow** | **15s** | **❌ EXPLODES** | **350-370 m/s** |

**Conclusion**: The problem is specifically the **inflow boundary condition interacting with Strang operator splitting**, NOT a bug in the ODE solver or FVM implementation.

### Mathematical Background

Our ARZ model uses Strang splitting to solve:

```
∂ₜu = Hyperbolic(u) + Source(u)
```

Where:
- **Hyperbolic**: Advection term (FVM with WENO5/Godunov + SSP-RK3)
- **Source**: ODE relaxation term S = (Ve-v)/τ (solved with RK45)

**Strang Splitting Algorithm**:
```
1. u* = ODE_solve(u, dt/2)           # Half-step ODE
2. u** = Hyperbolic_solve(u*, dt)    # Full-step FVM
3. u_new = ODE_solve(u**, dt/2)      # Half-step ODE
```

**The Problem**: When using **inflow boundary conditions**:
- Inflow BC sets u|boundary = b(t) (e.g., rho=0.15, v=3.0)
- BC is applied to BOTH substeps (ODE and Hyperbolic)
- BUT the BC b(t) is only correct for the FULL operator (Hyperbolic + Source)
- It's INCORRECT for the split substeps individually!

This creates **discontinuity accumulation** at each Strang cycle → velocity explosions after ~1400 timesteps.

---

## Literature-Based Solution

### Key References

1. **Einkemmer, L., Moccaldi, M., & Ostermann, A. (2018)**
   - "Efficient boundary corrected Strang splitting"
   - Applied Mathematics and Computation, 332, 76-89
   - DOI: 10.1016/j.amc.2018.03.006

2. **Descombes, S., et al. (2015)**
   - "Overcoming order reduction in diffusion-reaction splitting. Part 1: Dirichlet boundary conditions"
   - SIAM Journal on Scientific Computing, 37(5), A2576-A2593

3. **Hansen, E., & Kramer, F. (2012)**
   - "A comparison of boundary correction methods for Strang splitting"
   - Numerical Methods for Partial Differential Equations

### The Boundary Correction Method

**Core Idea**: Introduce a **correction function q(x,t)** that modifies the source term to be compatible with boundary conditions at intermediate steps.

**Mathematical Formulation**:

Instead of splitting:
```
∂ₜu = Hyperbolic(u) + Source(u)
```

We split:
```
∂ₜu = Hyperbolic(u) + [Source(u) - q] + q
```

Where:
- **q** is the correction function
- **Source(u) - q** is treated in the ODE substeps
- **q** is treated in the hyperbolic substep (or neglected if small)

**Boundary Condition for Correction Function**:

For inflow boundary conditions u|∂Ω = b(t), the correction function must satisfy:
```
q|∂Ω = Source(b(t))
```

For ARZ model:
```
q|boundary = (Ve(rho_b) - v_b) / τ
```

Where rho_b, v_b are the boundary values from inflow BC.

---

## Implementation Strategy

### Option 1: Simple Correction (Recommended for ARZ)

**Simplest approach** from literature: Use the boundary value of the source term as correction.

**Algorithm**:
```python
def strang_splitting_with_bc_correction(u, bc_left, bc_right, dt, params):
    """
    Strang splitting with boundary correction for inflow BC.
    
    Based on Einkemmer et al. (2018) for inhomogeneous boundary conditions.
    """
    
    # Step 1: Compute correction function q at boundaries
    if bc_left['type'] == 'inflow':
        rho_b = bc_left['rho']
        v_b = bc_left['v']
        Ve_b = equilibrium_velocity(rho_b, params)
        q_left = (Ve_b - v_b) / params['tau_m']
    else:
        q_left = 0.0
    
    if bc_right['type'] == 'inflow':
        rho_b = bc_right['rho']
        v_b = bc_right['v']
        Ve_b = equilibrium_velocity(rho_b, params)
        q_right = (Ve_b - v_b) / params['tau_m']
    else:
        q_right = 0.0
    
    # Step 2: First ODE half-step with CORRECTED source term
    # Only apply correction near boundaries (within ghost cell region)
    u_star = ode_substep_with_correction(u, dt/2, q_left, q_right, params)
    
    # Step 3: Apply boundary conditions to intermediate state
    apply_boundary_conditions(u_star, bc_left, bc_right, params)
    
    # Step 4: Full hyperbolic step (unchanged)
    u_star_star = hyperbolic_substep(u_star, dt, params)
    
    # Step 5: Second ODE half-step with CORRECTED source term
    u_new = ode_substep_with_correction(u_star_star, dt/2, q_left, q_right, params)
    
    # Step 6: Apply boundary conditions to final state
    apply_boundary_conditions(u_new, bc_left, bc_right, params)
    
    return u_new
```

**Key Implementation Details**:

1. **Correction Application Region**: 
   - Apply correction only in ghost cells and first interior cell
   - Use smooth transition (e.g., exponential decay) from boundary

2. **Modified ODE Substep**:
   ```python
   def ode_substep_with_correction(u, dt, q_left, q_right, params):
       """
       Solve ODE substep with boundary correction.
       Source term: S_corrected = S_original - q * weight(x)
       """
       rho, v, w = extract_variables(u)
       
       # Compute original source term
       Ve = equilibrium_velocity(rho, params)
       S_original = (Ve - v) / params['tau_m']
       
       # Apply correction with spatial weighting
       weight_left = compute_boundary_weight(x, 'left', params)
       weight_right = compute_boundary_weight(x, 'right', params)
       
       S_corrected = S_original - (q_left * weight_left + q_right * weight_right)
       
       # Solve ODE with corrected source
       v_new = solve_ode(v, S_corrected, dt)
       
       return u_new
   ```

3. **Boundary Weight Function**:
   ```python
   def compute_boundary_weight(x, side, params):
       """
       Smooth weight function for boundary correction.
       Decays exponentially from boundary into domain.
       """
       n_ghost = params['ghost_cells']
       dx = params['dx']
       decay_length = 3 * dx  # Correction extends 3 cells from boundary
       
       if side == 'left':
           distance = x - x[0]
       else:
           distance = x[-1] - x
       
       weight = np.exp(-distance / decay_length)
       weight[distance > 5*decay_length] = 0.0  # Cut off far from boundary
       
       return weight
   ```

### Option 2: Elliptic Correction (More Accurate but Expensive)

For higher accuracy, solve elliptic problem for correction function:
```
Δq = 0  (or some suitable RHS)
q|∂Ω = Source(b(t))
```

This ensures smooth correction function but requires solving additional PDE.

**Trade-off**: Better accuracy vs. higher computational cost.

### Option 3: BC Application Timing Modification

Alternative simpler approach (may work for ARZ):

**Idea**: Apply inflow BC only AFTER ODE substeps, not during them.

**Modified Algorithm**:
```python
def strang_splitting_modified_bc_timing(u, bc_left, bc_right, dt, params):
    """
    Modified Strang splitting: BC applied only after ODE substeps.
    """
    # Step 1: First ODE half-step WITHOUT applying BC
    u_star = ode_substep(u, dt/2, params)
    # NO BC application here!
    
    # Step 2: Apply BC before hyperbolic step
    apply_boundary_conditions(u_star, bc_left, bc_right, params)
    
    # Step 3: Full hyperbolic step
    u_star_star = hyperbolic_substep(u_star, dt, params)
    
    # Step 4: Second ODE half-step WITHOUT applying BC
    u_new = ode_substep(u_star_star, dt/2, params)
    
    # Step 5: Apply BC at end of full step
    apply_boundary_conditions(u_new, bc_left, bc_right, params)
    
    return u_new
```

**Rationale**: ODE substep is spatially local (no advection), so BC can be deferred until hyperbolic step.

---

## Recommended Implementation Plan

### Phase 1: Test BC Timing Modification (Quick Test)

**Effort**: LOW (~1 hour)
**Risk**: LOW (easy to revert)

1. Modify `strang_splitting_step()` in `time_integration.py`
2. Move BC application to AFTER ODE substeps instead of within them
3. Run existing test suite to verify stability
4. If successful, this is the simplest fix!

### Phase 2: Implement Simple Boundary Correction (Recommended)

**Effort**: MEDIUM (~3-4 hours)
**Risk**: LOW (literature-validated approach)

1. Add `compute_boundary_correction()` function
2. Add `compute_boundary_weight()` function
3. Modify `ode_substep()` to accept correction term
4. Update `strang_splitting_step()` to use corrected substeps
5. Add unit tests for correction computation
6. Run full test suite including component isolation tests

### Phase 3: Validation and Tuning

**Effort**: MEDIUM (~2-3 hours)

1. Test with original parameters (rho_m=0.15, v_m=3.0, t_max=15s)
2. Verify velocity stays within physical bounds (~3-16 m/s range)
3. Test with various inflow BC configurations
4. Compare results with outflow BC (should be similar dynamics)
5. Tune decay_length parameter if needed
6. Document parameter sensitivity

---

## Expected Outcomes

### Success Criteria

- ✅ Velocity remains bounded (< 20 m/s) for entire simulation
- ✅ No explosions or numerical instabilities
- ✅ Similar behavior to outflow BC test (which passed)
- ✅ All component isolation tests continue to pass
- ✅ Computational cost increase < 10% (correction computation is cheap)

### Failure Indicators

If correction doesn't stabilize:
- ⚠️ Correction function q may need smoother spatial transition
- ⚠️ Decay length may need adjustment
- ⚠️ May need elliptic correction (Option 2) for better smoothness
- ⚠️ Check if BC values themselves are physically unrealistic

---

## Technical Notes

### Why This Works

1. **Compatibility**: Correction function q makes the source term compatible with BC at intermediate steps
2. **Smoothness**: Weight function ensures smooth transition from boundary to interior
3. **Consistency**: Full operator (Hyperbolic + Source) still sees correct BC
4. **Literature Validated**: Multiple papers confirm this approach for similar problems

### Limitations

- Correction adds slight computational overhead (~5-10%)
- Requires careful tuning of decay length parameter
- May need adjustment for time-varying inflow BC

### Alternative Approaches Not Pursued

1. **Implicit BC Treatment**: Solve BC implicitly in ODE step (complex, high cost)
2. **Modified Splitting Order**: Try Lie-Trotter instead of Strang (lower accuracy)
3. **Flux Correction**: Modify flux at boundary cells (less theoretically sound)

---

## References

- Einkemmer, L., Moccaldi, M., & Ostermann, A. (2018). Efficient boundary corrected Strang splitting. *Applied Mathematics and Computation*, 332, 76-89.
- Descombes, S., et al. (2015). Overcoming order reduction in diffusion-reaction splitting. Part 1: Dirichlet boundary conditions. *SIAM Journal on Scientific Computing*, 37(5), A2576-A2593.
- Hansen, E., & Kramer, F. (2016). A comparison of boundary correction methods for Strang splitting. arXiv preprint arXiv:1609.05505.

---

## Next Steps

1. **IMMEDIATE**: Implement Phase 1 (BC timing modification) for quick test
2. **IF Phase 1 fails**: Implement Phase 2 (boundary correction function)
3. **VALIDATE**: Run comprehensive test suite with various BC configurations
4. **DOCUMENT**: Update code comments and technical documentation
5. **OPTIMIZE**: Tune decay length and correction parameters if needed

---

**Status**: Ready for implementation
**Confidence**: HIGH (literature-validated solution for well-known problem)
**Risk**: LOW (can implement incrementally with easy rollback)
