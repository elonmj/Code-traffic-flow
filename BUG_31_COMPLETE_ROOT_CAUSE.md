# 🔴 BUG #31: FINAL ROOT CAUSE - Fundamental Design Issue with Boundary Modulation

## COMPLETE DIAGNOSIS

### The Problem Chain

1. **Initial Discovery**: Reward always = 0 (no learning signal)
   ↓
2. **Found**: No queues detected (velocity always at free speed)
   ↓
3. **Found**: Traffic signal BC updates ARE applied (RED phase reduces inflow velocity)
   ↓
4. **Found**: But velocities in domain are STILL at free speed (not affected by boundary!)
   ↓
5. **ROOT CAUSE**: Boundary condition modulation approach is fundamentally incompatible with ARZ model

---

## ROOT CAUSE: Boundary Condition Velocity ≠ Domain Vehicle Velocity

### The Physics Problem

**ARZ Model Relaxation Equation:**
```
dw/dt = (V_e(ρ) - w) / τ + source_term
```

Where:
- w = momentum (prescribed at boundary: 1.2 m/s in RED phase)
- V_e = equilibrium velocity (determined by density via LWR flux)
- τ = relaxation time ≈ 1-1.2 s

**What happens:**
1. Boundary imposes w = 1.2 m/s at ghost cells
2. Ghost cells have inflow density ρ ≈ 200 veh/km
3. Equilibrium velocity V_e(200 veh/km) ≈ ... ?
4. Flux function determines realistic velocity given the density

The issue: **The boundary condition w is JUST AN INITIAL VALUE that gets immediately relaxed by the conservation law!**

### Concrete Example

**RED Phase at Ghost Cell:**
```
Imposed: [rho_m=0.2 veh/m, w_m=1.2 m/s, ...]
Equilibrium V_e: (depends on density via LWR formula) ≈ 3 m/s
Actual velocity in next step: Relaxes from 1.2 m/s toward 3 m/s
```

But then vehicles FLOW into the domain with velocity 3 m/s, and the DOMAIN's flux function determines the NEXT velocity based on THAT density.

**Result**: The velocity is determined by the DENSITY distribution in the domain, not by what we imposed at the boundary!

### Why Vehicles Stay at 11 m/s

The domain has LOW density (0.04 veh/m ≈ 40 veh/km), which means:
- V_e = 0.6 + (8.89 - 0.6) * (1 - 40/370) ≈ 8.8 m/s
- Vehicles freely accelerate to this (eventually relaxing toward 11 m/s free speed)

So NO MATTER what boundary velocity we impose, once inside the LOW-DENSITY domain, vehicles accelerate to free speed!

---

## Why Current Fix Doesn't Work

**Wrong Assumption**: "If we reduce boundary velocity in RED phase, queues will form"

**Reality**: 
- Boundary velocity is ONLY for the ghost cells
- Domain vehicles are governed by LOCAL DENSITY, not boundary conditions
- Low-density domain → vehicles at free speed (regardless of boundary)

### Attempted Fix Failure

Changed inflow velocities:
- Before: w_m = 1.0 m/s (RED), 1.0 m/s (GREEN) - no modulation possible
- After: w_m = 1.2 m/s (RED), 2.5 m/s (GREEN) - still no effect!

**Why it failed**: Both boundary velocities are MUCH LOWER than the free speed (8.8 m/s) that the low-density domain can sustain.

---

## CORRECT FIX APPROACHES

### Option 1: Increase Inflow Density High Enough
Force the domain to stay CONGESTED by having sustained high inflow

**Requirements:**
- Inflow density: 250-300 veh/km (near jam)
- This would force domain velocity down through conservation laws
- RED phase: block outflow instead of reducing inflow → queue BACKUPS upstream
- GREEN phase: allow outflow → queue DRAINS

**Problem**: This isn't realistic for Lagos traffic (not continuously jammed)

### Option 2: Use OUTPUT (Right) Boundary Modulation Instead
More realistic: Restrict OUTFLOW to create backpressure

```
RED phase:
  - Left boundary: normal inflow (rho=200 veh/km, v~30 km/h)
  - Right boundary: restrictive (70% outflow) → backs up traffic
  - Result: Queue forms through conservation laws

GREEN phase:
  - Left boundary: normal inflow
  - Right boundary: free outflow (100%)
  - Result: Queue drains
```

**Advantage**: 
- Physically realistic (traffic light restrictions downstream)
- Uses domain's natural conservation laws
- Clear RED/GREEN signal

### Option 3: Proper Traffic Light Model (Network Nodes)
Use `network_coupling` + `node_solver` for actual intersection handling

```python
# Currently DISABLED (has_network=False)
# Would provide:
#   - Proper intersection modeling
#   - Queue management at nodes
#   - Realistic traffic light enforcement
```

**Advantage**: Most physically accurate, aligns with traditional TSC literature

---

##RECOMMENDATION

**Short-term (Immediate Fix for RL Training):**
→ **Option 2: Outflow modulation**
- Modify set_traffic_signal_state to restrict RIGHT boundary instead of LEFT
- More realistic Lagos scenario (signal restricts downstream exits)
- Clear RL learning signal

**Long-term (Proper Implementation):**
→ **Option 3: Enable network mode**
- Activate has_network=True in scenario YAML
- Configure proper node solver
- Matches academic TSC literature

---

## Implementation Plan for Option 2

### Step 1: Modify `set_traffic_signal_state()` in `runner.py`

Change from modulating LEFT inflow to modulating RIGHT outflow:

```python
def set_traffic_signal_state(self, intersection_id: str, phase_id: int) -> None:
    """
    RED phase: Restrict outflow (0.5 * normal)
    GREEN phase: Allow free outflow
    """
    base_state = self.traffic_signal_base_state or self.initial_equilibrium_state
    
    if phase_id == 0:  # RED: restrict outflow
        # Reduce outflow rate by constraining boundary (simulation effect)
        # For outflow BC: just set type='outflow' but monitor how much escapes
        # Problem: outflow BC doesn't have 'state' param
        # Solution: Use zero-order extrapolation but add artificial restriction?
        pass
    elif phase_id == 1:  # GREEN: normal outflow
        pass
```

**Problem**: ARZ doesn't have an easy way to restrict outflow BC!

### Alternative: Modify Inflow Strategy

Instead of boundary modulation, create QUEUES through initial/boundary conditions:
1. Set inflow density = 250 veh/km (sustained high)
2. Set inflow velocity = low (2-3 m/s) - creates natural congestion
3. RED: even MORE restrictive (0.5 m/s)
4. GREEN: less restrictive (3-5 m/s)

This way, queues form naturally through mass conservation!

---

## Current Status

**Root Cause**: ✅ Identified - Boundary velocity incompatible with domain conservation laws
**Problem**: 🔴 Not solved - Inflow modulation approach fundamentally broken
**Impact**: 🔴 CRITICAL - RL cannot learn without proper congestion/queue signals

**Options**: Wait for proper fix (Option 2 or 3) before continuing training

---

**Date**: 2025-10-21  
**Diagnosis**: Complete  
**Severity**: 🔴 CRITICAL - Blocks RL training  
**Root Cause**: Architectural (boundary modulation vs conservation laws)  
**Estimated Fix Time**: 2-3 hours (implement Option 2 or enable Option 3)
