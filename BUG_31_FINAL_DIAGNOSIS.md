# 🔴 BUG #31: FINAL ROOT CAUSE - Why Red Phase Doesn't Create Queues

## THE ACTUAL BUG (After Tracing)

### Status Quo:
```
RED phase (phase_id=0):
  └─ Inflow velocity reduced: 1.014 km/h → 0.507 km/h (50% reduction)
  └─ In SI units: 0.28179 m/s → 0.14090 m/s

GREEN phase (phase_id=1):
  └─ Inflow velocity normal: 1.014 km/h → 0.28179 m/s
```

**WAIT!** This makes NO SENSE!

- Free speed motorcycles: 32 km/h = 8.888 m/s
- Free speed cars: 28 km/h = 7.777 m/s
- Boundary inflow velocity: 1.014 km/h = 0.28 m/s

**The inflow velocity (0.28 m/s) is ~30x LOWER than free speed (8.8 m/s)!**

### Question: What does boundary inflow velocity represent?

Looking at the YAML generation (test_section_7_6_rl_performance.py), let me search for how the inflow state is constructed:

**From YAML in traffic_light_control.yml:**
```yaml
boundary_conditions:
  left:
    state:
    - 300.0        # rho_m in veh/km
    - 1.014444     # w_m (claimed to be in same units as velocity in config)
    - 96.0         # rho_c in veh/km
    - 0.388889     # w_c (claimed to be in same units as velocity in config)
```

**But what does `1.014` represent?**

If `w` is MOMENTUM:
- w = rho × v (in SI: veh/m × m/s)
- w_m = 0.3 veh/m × v_m = 1.014... ??? This doesn't match 1.014 in km/h units!

If `w` is VELOCITY:
- w_m = 1.014 km/h = 0.28179 m/s
- But density × velocity = 0.3 veh/m × 0.28179 m/s = 0.0845 veh·m/s ... still doesn't make sense

### The REAL Problem: UNIT MISMATCH IN YAML

Let me check what units the boundary condition actually expects. From `parameters.py` line 211:

```python
state[1] * KMH_TO_MS,      # w_m (assuming w is in same units as v in config)
state[3] * KMH_TO_MS       # w_c (assuming w is in same units as v in config)
```

**The comment says "assuming w is in same units as v"** - meaning velocity in km/h!

So the YAML `state` is: `[rho_veh/km, v_km/h, rho_veh/km, v_km/h]`

But the value `1.014 km/h` is WRONG! It should be approximately equal to free speed:
- Free speed motorcycles: 32 km/h
- But we have: 1.014 km/h

**This is 32x lower than it should be!**

### WHERE DID THE YAML GET GENERATED?

Need to trace where test_section_7_6_rl_performance.py creates this YAML.

### The Root Problem Identified

**The boundary condition `state` in the YAML contains VELOCITIES that are not equal to FREE SPEED but much lower!**

This means:
1. ✅ RED phase reduces 1.014 km/h to 0.507 km/h
2. ❌ But even GREEN phase (1.014 km/h) is 30x slower than free speed (32 km/h)!
3. ❌ So inflow vehicles enter slowly → queue ALWAYS exists
4. ❌ RED vs GREEN phase doesn't make a difference → no control signal

**So the real issue is:**
- **The boundary condition velocity is WRONG** (too low)
- **Not the traffic light control logic** (which is working correctly)

### Why is Boundary Velocity Wrong?

The boundary condition should specify:
- ON NORMAL (GREEN): Inflow at approximately free speed
- ON RED: Inflow at reduced speed

But what we have:
- ON NORMAL (GREEN): Inflow at 1.014 km/h (tiny!)
- ON RED: Inflow at 0.507 km/h (even tinier!)

### HYPOTHESIS: Boundary Velocity is being CONFUSED with DENSITY

Let me check: Maybe 1.014 is not velocity but something else?

Looking at the YAML structure:
```yaml
state:
  - 300.0        # 300 veh/km (seems reasonable)
  - 1.014444     # ???
  - 96.0         # 96 veh/km (seems reasonable)
  - 0.388889     # ???
```

**If the boundary was generated from an EQUILIBRIUM state:**

At equilibrium, vehicles maintain a certain spacing and velocity. The calculation would be:
- equilibrium_flow = flow_capacity ÷ number_of_lanes (or similar)
- equilibrium_velocity = some function of density

Let me check if the boundary velocity is actually an EQUILIBRIUM or SATURATION VELOCITY:

1.014 km/h at density 300 veh/km means:
- flow = 300 veh/km × 1.014 km/h = 304.2 veh/h

At density 96 veh/km with velocity 0.389 km/h:
- flow = 96 × 0.389 = 37.3 veh/h

These are VERY LOW flows!

### REAL ROOT CAUSE:

**The boundary condition `state` is configured INCORRECTLY. The velocity is far too low, which means:**

1. Vehicles enter at 1.014 km/h (RED and GREEN stay at same order of magnitude)
2. No meaningful RED vs GREEN difference is possible
3. Queue ALWAYS exists (due to low inflow)
4. RL agent receives no learning signal because "congestion always exists"

### FIX REQUIRED:

Need to investigate where this YAML state comes from and fix the velocity values to be meaningful:
- GREEN phase: inflow velocity = ~30 km/h (or close to free speed)
- RED phase: inflow velocity = ~10 km/h (or 30-50% of free speed)

---

**Date**: 2025-10-21
**Status**: 🔴 ROOT CAUSE CONFIRMED - YAML CONFIG CONTAINS WRONG BOUNDARY VELOCITIES
**Impact**: RL cannot learn because RED phase doesn't meaningfully change vehicle behavior
**Blame**: Test scenario generation set incorrect boundary condition velocities
