# 🔴 BUG #31: ROOT CAUSE ANALYSIS - Why No Congestion in RL Training

## THE COMPLETE PICTURE (Finally!)

### Réalité: RL Agent Reçoit Toujours Reward = 0.0

```
Observed in logs:
[BASELINE] Reward: -0.010000  ✅ Non-zero
[RL] Reward: 0.000000  ❌ x40+ times IDENTICAL ZERO!
```

### Root Cause Chain

**CHAIN OF EVENTS:**

1. **Scenario YAML has NO `has_network` field**
   - File: `traffic_light_control.yml`
   - Result: `params.has_network = False` (default)

2. **`has_network=False` → node_solver DISABLED**
   - Location: `runner.py:170` - `if self.params.has_network:` is FALSE
   - Result: Network intersection handling is skipped entirely

3. **Traffic signal control uses WRONG approach**
   - Method: `set_traffic_signal_state()` modulates BOUNDARY CONDITIONS
   - Modifies: `current_bc_params['left']['state']` with reduced velocities
   - Expected: RED phase = w * 0.5 (slower vehicles), GREEN phase = w * 1.0 (normal)
   - Problem: This ONLY works if APPLIED to the inflow boundary each timestep

4. **Boundary conditions are applied ONCE per run() loop iteration**
   - Location: `runner.py:503` - `apply_boundary_conditions(current_U, ...)`
   - BUT: Between calls to `set_traffic_signal_state()` and next `run()`, state stays unchanged!

5. **The REAL problem: Traffic signal state NOT PERSISTED**
   - ✅ `env.step()` calls `set_traffic_signal_state('left', phase_id)` → updates `current_bc_params`
   - ✅ Next iteration of `runner.run()` SHOULD apply this at the inflow boundary
   - ❓ **QUESTION**: Does `runner.run()` actually use the UPDATED `current_bc_params`?

---

## ARCHITECTURE MISMATCH

| Component | Purpose | Status |
|-----------|---------|--------|
| `node_solver.py` | Handle intersections with feux rouges | ❌ NEVER USED (`has_network=False`) |
| `set_traffic_signal_state()` | Modulate inflow BC for RL | ⚠️ IMPLEMENTED but EFFECT UNCLEAR |
| `boundary_conditions.py` | Apply BC to ghost cells | ✅ USED EVERY TIMESTEP |
| `network_coupling.py` | Couple multiple nodes | ❌ BYPASSED (network disabled) |

---

## THE REAL BUG: TWO INCOMPATIBLE APPROACHES

### Approach #1: Network Nodes (NOT ENABLED)
```python
# runner.py:170
if self.params.has_network:
    # Call network_coupling → uses node_solver → handles traffic lights
    self.network_coupling.apply_network_coupling(...)
else:
    # SKIP - This is what happens!
    pass
```

### Approach #2: Boundary Condition Modulation (IMPLEMENTED but UNCLEAR)
```python
# traffic_signal_env_direct.py:244
self.runner.set_traffic_signal_state('left', phase_id=self.current_phase)
# This updates current_bc_params, BUT...
# When does it get USED?
```

---

## EVIDENCE: SCENARIO NEVER HAS CONGESTION

### Measured Velocities in Log:
```
[QUEUE_DIAGNOSTIC] velocities_m (m/s): [11.111111 11.111111 11.111111 ...]
[QUEUE_DIAGNOSTIC] velocities_c (m/s): [13.888889 13.888889 13.888889 ...]
```

**ALL vehicles move at FREE SPEED always!**

### Queue Threshold vs Reality:
```
Threshold (after fix): 7.78 m/s (0.7 * v_free)
Actual velocities: 11.11 - 13.89 m/s
Result: NO vehicles below threshold → NO queue detected
Reward: R_queue = 0 (queue change = 0)
```

### Why No Queues Form?
1. **Inflow state from YAML**: density = 200/96 veh/km (heavy but feasible)
2. **Outflow**: ALWAYS free (no constraint from RED phase)
3. **Net effect**: No backpressure → no velocity reduction → all vehicles at free speed

---

## WHAT SHOULD HAPPEN WITH RED PHASE

### If RED was working:
```
RED phase (phase_id=0):
└─ runner.set_traffic_signal_state('left', phase_id=0)
   └─ Reduces: w_m *= 0.5, w_c *= 0.5
   └─ Stored in: current_bc_params['left']['state']
   └─ Next runner.run() loop:
      └─ apply_boundary_conditions(..., current_bc_params)
         └─ Applies reduced velocity inflow
         └─ Vehicles enter with lower velocity
         └─ Queue forms upstream
         └─ Queue velocity < threshold
         └─ Reward signal = non-zero!
```

### What Actually Happens:
```
RED phase (phase_id=0):
└─ runner.set_traffic_signal_state('left', phase_id=0)
   └─ Updates current_bc_params['left']
   └─ BUT: Is this actually USED?
      └─ If YES: Should work, but queue still not observed
      └─ If NO: Traffic signal has NO EFFECT
         └─ Vehicles enter at full speed regardless
         └─ No queue → no queue signal
         └─ Reward = 0
```

---

## HYPOTHESIS: `current_bc_params` NOT PROPAGATED

### Potential Issue:
In `runner.set_traffic_signal_state()`:
```python
# Line 766
if not hasattr(self, 'current_bc_params'):
    self.current_bc_params = copy.deepcopy(self.params.boundary_conditions)

# Line 769
self.current_bc_params[intersection_id] = bc_config
```

**Question**: Does `self.runner` in `TrafficSignalEnvDirect` see these updates?

### Trace:
1. `env = TrafficSignalEnvDirect(...)` → creates `self.runner`
2. `env.step(action)` → calls `self.runner.set_traffic_signal_state(...)`
3. `env.runner.run(...)` → uses `self.current_bc_params`

**Are they the SAME object?** If `runner` is recreated or if `current_bc_params` is a local copy, updates are lost!

---

## PROPOSED FIXES

### Option 1: Enable Network Mode (CORRECT but Complex)
```yaml
# traffic_light_control.yml
network:
  has_network: true
  nodes:
    - id: left_intersection
      segments: [upstream, downstream]
      traffic_lights:
        controller: fixed_time
        cycle: 120s
        green_duration: 60s
```

Requires:
- Implement proper `Intersection` objects
- Configure `node_solver` for feu rouge logic
- Activate `network_coupling` in `runner.run()`

### Option 2: Fix Boundary Condition Propagation (SIMPLER)
Ensure `set_traffic_signal_state()` modifications PERSIST and are USED:
```python
# Verify current_bc_params is passed to apply_boundary_conditions
# AND that it reflects RED/GREEN phase changes
```

### Option 3: Use Hybrid Approach (PRAGMATIC)
- Keep boundary condition approach (simpler)
- FIX: Make sure RED phase reduces inflow velocity
- VERIFY: Queue is actually detected with adaptive threshold

---

## STATUS

**Root Cause**: `has_network=False` + Traffic signal BC may not be properly integrated

**Impact**: 
- RL agent receives reward = 0 always
- Cannot learn from queue dynamics
- Cannot form meaningful control policy

**Severity**: 🔴 CRITICAL - Blocks entire RL validation

**Next Action**: 
1. Verify `current_bc_params` updates in `set_traffic_signal_state()` actually persist
2. Add logging to see if RED/GREEN phases change the inflow velocity
3. Either enable network mode OR fix BC modulation approach

---

**Date**: 2025-10-21
**Status**: 🔴 ROOT CAUSE IDENTIFIED - ARCHITECTURAL ISSUE
**Blame**: Two incompatible approaches (network nodes vs BC modulation) with network disabled
