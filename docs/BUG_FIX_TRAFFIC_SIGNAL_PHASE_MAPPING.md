# BUG #4: Traffic Signal Phase Mapping Inverted

## Date Discovered
2025-10-10

## Severity
**CRITICAL** - Fundamental conceptual error causing domain drainage

## Summary
The traffic signal phase mapping in `runner.py` is **inverted**. Phase 0 (RED) uses OUTFLOW boundary condition which **blocks inflow and allows outflow**, causing the domain to drain to vacuum.

## Root Cause Analysis

### Current (BUGGY) Implementation
File: `arz_model/simulation/runner.py` lines 700-715

```python
# Phase 0 = red (outflow/free BC to drain traffic) ← WRONG CONCEPT!
# Phase 1 = green (inflow BC to allow traffic)
if phase_id == 0:
    # Red phase: use outflow boundary condition
    bc_config = {
        'type': 'outflow',  # ← BLOCKS INFLOW, allows outflow
        'extrapolation_order': 1
    }
elif phase_id == 1:
    # Green phase: use inflow with equilibrium state
    bc_config = {
        'type': 'inflow',  # ← Allows inflow
        'state': self.initial_equilibrium_state
    }
```

### The Problem

**What happens with current mapping:**

1. **Phase 0 (RED):**
   - BC type: `outflow`
   - Effect: Traffic **EXITS** the domain (right boundary extrapolates)
   - Effect: Traffic **CANNOT ENTER** (left boundary is in outflow mode)
   - Real-world analogy: Red light that **blocks cars from entering** an intersection ❌

2. **Phase 1 (GREEN):**
   - BC type: `inflow`  
   - Effect: Traffic **ENTERS** the domain (left boundary injects equilibrium state)
   - Effect: Traffic **EXITS** the domain (right boundary typically outflow)
   - Real-world analogy: Green light allows entry ✅

**Simulation Behavior:**
```
Step 0 (Phase 1 - GREEN): rho_m = 0.037 (initial shock)
Step 1 (Phase 0 - RED):   rho_m = 0.008 (outflow drains, no inflow)
Step 2 (Phase 1 - GREEN): rho_m = 0.000022 (VACUUM!)
```

The domain drains because:
- During RED phases (50% of time), traffic exits but nothing enters
- Even with BC fix imposing inflow momentum, the **BC TYPE is wrong**
- Outflow BC extrapolates interior state to boundary, doesn't inject anything

### Conceptual Error

The code comment says "Phase 0 = red (outflow/free BC to **drain traffic**)"

This reveals the misunderstanding: **A red light doesn't drain traffic from upstream!**

In reality:
- **RED light**: Blocks traffic **AT** the intersection (stops outflow)
- **GREEN light**: Allows traffic **THROUGH** the intersection (enables outflow)

The boundary at `x=0` represents the **upstream entry point**, NOT the intersection itself!

## Correct Interpretation

For traffic signal control at a boundary representing upstream entry:

### Option A: Boundary IS the intersection (current model assumption)
- **Phase 0 (RED)**: Should use **INFLOW** to maintain upstream traffic supply
- **Phase 1 (GREEN)**: Should use **INFLOW** with potentially higher rate
- Control mechanism: Vary inflow rate or state, not BC type

### Option B: Boundary is upstream, intersection is interior
- Keep **INFLOW** at boundary always (traffic always arrives from upstream)
- Model traffic signal as **interior source/sink** term
- This requires modifying PDE solver, not just BCs

## Current Impact

### Why validation fails with 0% improvement:
1. ✅ Bug #1 fixed: BaselineController now alternates correctly (1.0 → 0.0 → 1.0...)
2. ✅ Bug #2 fixed: No 10-step diagnostic limit
3. ✅ Bug #3 fixed: Inflow BC imposes full state [rho_m, w_m, rho_c, w_c]
4. ❌ **Bug #4 active**: But Phase 0 uses OUTFLOW, so Bug #3 fix never activates during red phases!

### Evidence from logs:
```
Step 0 (Action 1.0 → Phase 1 GREEN → INFLOW):  rho_m=0.037
Step 1 (Action 0.0 → Phase 0 RED → OUTFLOW):   rho_m=0.008
Step 2 (Action 1.0 → Phase 1 GREEN → INFLOW):  rho_m=0.000022
```

Even with green phases injecting traffic, red phases drain it faster than it accumulates!

## Proposed Solutions

### Solution 1: Always use INFLOW (recommended)
Keep boundary in inflow mode always, vary the inflow state parameters:

```python
if phase_id == 0:
    # Red phase: reduced inflow (models congestion backup)
    bc_config = {
        'type': 'inflow',
        'state': self.reduced_inflow_state  # Lower w, same rho
    }
elif phase_id == 1:
    # Green phase: normal inflow
    bc_config = {
        'type': 'inflow',
        'state': self.initial_equilibrium_state
    }
```

### Solution 2: Control via interior terms
Model the traffic signal as interior boundary within domain:
- Keep left BC always as inflow (upstream supply)
- Add interior source/sink term at signal location
- More physically accurate but requires solver modifications

### Solution 3: Reinterpret phases
If outflow must be used:
- Phase 0 (RED) → Action blocks interior flow → Use INFLOW at boundary  
- Phase 1 (GREEN) → Action allows interior flow → Use INFLOW at boundary
- Make phases control something OTHER than BC type

## Recommended Fix

**Immediate fix for validation:**

```python
def set_traffic_signal_state(self, intersection_id: str, phase_id: int) -> None:
    # ALWAYS use inflow at left boundary (traffic always arrives from upstream)
    # Phase controls inflow characteristics, not BC type
    
    if phase_id == 0:
        # Red phase: Congested inflow (traffic backs up)
        # Lower velocity to model queue formation
        if hasattr(self, 'initial_equilibrium_state'):
            base_state = self.initial_equilibrium_state
            # Reduce velocity by 50% for red phase
            red_state = [
                base_state[0],           # rho_m (same)
                base_state[1] * 0.5,     # w_m (reduced)
                base_state[2],           # rho_c (same)
                base_state[3] * 0.5      # w_c (reduced)
            ]
            bc_config = {'type': 'inflow', 'state': red_state}
        else:
            bc_config = {'type': 'inflow', 'state': None}
    
    elif phase_id == 1:
        # Green phase: Free-flow inflow
        bc_config = {
            'type': 'inflow',
            'state': self.initial_equilibrium_state if hasattr(self, 'initial_equilibrium_state') else None
        }
```

This maintains traffic supply while allowing control differentiation!

## Testing Plan

1. Apply fix to `runner.py`
2. Rerun local test with baseline controller
3. Verify domain does NOT drain to vacuum
4. Verify baseline and RL show different behaviors
5. Rerun Kaggle validation with fix
6. Expect non-zero improvement metrics

## Related Bugs
- Bug #1: BaselineController.update() not called (FIXED)
- Bug #2: 10-step diagnostic limit (FIXED)
- Bug #3: Inflow BC extrapolates momentum (FIXED)
- Bug #4: Traffic signal phase mapping inverted (THIS BUG)

All 4 bugs must be fixed for meaningful RL vs baseline comparison!
