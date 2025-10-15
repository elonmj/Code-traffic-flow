# BUG #35: PERSISTENT ISSUE - Queues Still Zero After GPU Fix

## DATE: 2025-10-15 23:13 UTC

## STATUS: 🔴 BLOCKING - NOT RESOLVED

---

## SUMMARY

Despite applying fixes to both CPU and GPU ODE solvers to ensure road quality (`d_R`) is passed correctly, queues remain at **0.00 in ALL timesteps** across multiple kernel runs.

### Kernels Tested
1. **obwe** (23:01): First attempt with CPU fix only → Queues = 0.00
2. **tzte** (23:08): Second attempt with GPU fix added → Queues = 0.00

Both kernels show:
- ✅ Road quality loaded correctly (`"Road quality loaded"` in logs)
- ✅ Training rewards diverse (-0.01 to 0.02, 10-13% zeros)
- ❌ **ALL queues = 0.00** throughout training and evaluation
- ❌ Evaluation: 97.5% zeros, agent stuck at action=1

---

## FIXES APPLIED (BUT NOT WORKING)

### Fix 1: CPU ODE Solver (Line 143-155)
```python
if grid.road_quality is None:
     raise ValueError(
         "❌ BUG #35: Road quality array not loaded before ODE solver! "
         "Equilibrium speed Ve calculation requires grid.road_quality. "
     )
else:
    R_local = grid.road_quality[physical_idx]
```

**Status**: Applied, no error raised → Road quality IS loaded
**But**: Simulation runs on GPU, so CPU path not executed

### Fix 2: GPU Network Splitting (Line 947-962)
```python
if params.device == 'gpu':
    from .network_coupling_stable import apply_network_coupling_stable_gpu
    d_U_n = U_n
    
    # ✅ BUG #35 FIX: Pass d_R to GPU ODE solver
    if not hasattr(grid, 'd_R') or grid.d_R is None:
        if grid.road_quality is None:
            raise ValueError("❌ BUG #35: Road quality must be loaded before GPU simulation!")
        grid.d_R = cuda.to_device(grid.road_quality)

    # Étape 1: ODE dt/2
    d_U_star = solve_ode_step_gpu(d_U_n, dt / 2.0, grid, params, grid.d_R)  # Was: None
    
    # Étape 2: Hyperbolique
    d_U_with_bc = apply_network_coupling_stable_gpu(d_U_star, dt, grid, params, time)
    d_U_ss = solve_hyperbolic_step_standard_gpu(d_U_with_bc, dt, grid, params)
    
    # Étape 3: ODE dt/2
    d_U_np1 = solve_ode_step_gpu(d_U_ss, dt / 2.0, grid, params, grid.d_R)  # Was: None
```

**Status**: Applied, no error raised
**But**: Queues still 0.00 → Fix not effective OR wrong diagnosis

---

## EVIDENCE: LOGS ANALYSIS

### Kernel tzte Microscope Logs (First 15 Steps)

```
Step 1: QUEUE: current=0.00 prev=0.00 delta=0.0000 R_queue=-0.0000
Step 2: QUEUE: current=0.00 prev=0.00 delta=0.0000 R_queue=-0.0000
Step 3: QUEUE: current=0.00 prev=0.00 delta=0.0000 R_queue=-0.0000
Step 4: QUEUE: current=0.00 prev=0.00 delta=0.0000 R_queue=-0.0000
Step 5: QUEUE: current=0.00 prev=0.00 delta=0.0000 R_queue=-0.0000
...
Step 100: QUEUE: current=0.00 prev=0.00 delta=0.0000 R_queue=-0.0000
```

**Pattern**: ZERO queues in ALL 140 timesteps (training + evaluation)

### Configuration Verification

```yaml
# traffic_light_control.yml (kernel tzte)
road:
  quality_type: uniform
  quality_value: 2  # ✅ Correct

boundary_conditions:
  left:
    state: [200.0, 2.2577..., 96.0, 1.555...]  # ✅ Equilibrium speeds (Bug #34 fix)
    type: inflow
```

**Road quality**: ✅ Configured correctly
**Inflow BC**: ✅ Equilibrium speeds (Bug #34 fix applied)

### Kernel Logs: Road Quality Loading

```
Output: "Loading road quality type: uniform"
Output: "Uniform road quality value: 2"
Output: "Road quality loaded."
Output: "Transferring initial state and road quality to GPU..."
```

**Confirmed**: Road quality IS being loaded and transferred to GPU

---

## HYPOTHESES: WHY QUEUES STILL ZERO?

### Hypothesis 1: GPU ODE Solver Not Using d_R ⭐ MOST LIKELY

**Evidence**:
- `solve_ode_step_gpu` has validation: `if d_R is None: raise ValueError`
- But no error raised → d_R is passed
- **But**: Maybe GPU kernel is hardcoded to use R=3 fallback?

**Check Needed**:
```python
# In GPU ODE kernel (_ode_rhs_kernel_gpu):
# Does it actually USE d_R[j] to calculate Ve?
# Or does it have its own fallback logic?
```

### Hypothesis 2: Velocities Not Extracted Correctly in Environment

**Evidence**:
- Bug #35 fix targeted ODE solver
- But queue detection happens in `TrafficSignalEnvDirect._calculate_queue_length()`
- Maybe observation extraction returns wrong velocities?

**Check Needed**:
```python
# In traffic_signal_env_direct.py:
velocities_m = obs_data['v_m']  # Are these actual velocities from runner?
# Or are they normalized/transformed incorrectly?
```

### Hypothesis 3: Observation Segments Too Far from Traffic

**Evidence**:
- Default observation segments: [8, 9, 10] upstream, [11, 12, 13] downstream
- These are at x=80-130m
- Traffic might accumulate at x=0-50m (near inflow) but dissipate before reaching x=80m

**Check Needed**:
- Test with segments [2, 3, 4] (x=20-40m) near inflow
- Already tested in local diagnostic → Traffic WAS detected at x=20m
- **But**: In that test, velocities were also 15 m/s (wrong)

### Hypothesis 4: Queue Threshold Too Low

**Evidence**:
- Threshold: `v < 5.0 m/s` to be counted as queue
- At ρ=0.125 veh/m, equilibrium speed Ve ≈ 13 m/s (with R=2, Vmax=19.44 m/s)
- Velocities might be relaxing to ~10-13 m/s, not low enough for threshold

**Math Check**:
```python
# At ρ = 0.125 veh/m, R = 2
rho_jam = 0.37
g = 1 - 0.125/0.37 = 0.662
Vmax_m[2] = 19.44 m/s
Ve = 0.6 + (19.44 - 0.6) × 0.662 = 13.06 m/s

# With pressure p ≈ 3-5 m/s at this density:
v = w - p ≈ 13 - 4 = 9 m/s

# Queue threshold: v < 5 m/s
# 9 m/s > 5 m/s → NOT counted as queue!
```

**Conclusion**: Even with CORRECT relaxation, velocities might be 8-12 m/s, above threshold!

### Hypothesis 5: Inflow Density Not High Enough ⭐ STRONG CANDIDATE

**Evidence**:
- Inflow: ρ=200 veh/km = 0.2 veh/m
- This is only **54% of jam density** (0.37 veh/m)
- At ρ=0.2, equilibrium Ve ≈ 5 m/s (from Bug #34 analysis)
- **This is BARELY at the queue threshold!**

**Math**:
```python
# At ρ = 0.2 veh/m (inflow), R = 2
g = 1 - 0.2/0.37 = 0.459
Ve = 0.6 + (19.44 - 0.6) × 0.459 = 9.25 m/s

# With pressure p ≈ 4 m/s:
v ≈ 9.25 - 4 = 5.25 m/s

# Queue threshold: v < 5.0 m/s
# 5.25 > 5.0 → NOT counted as queue! ❌
```

**Diagnosis**: **INFLOW DENSITY TOO LOW!** Need higher ρ to get velocities < 5 m/s.

---

## RECOMMENDED FIXES

### Fix Option 1: Increase Inflow Density ⭐ RECOMMENDED

**Change**: Increase inflow density from 200 → 300 veh/km (81% jam density)

```python
# In Code_RL/src/utils/config.py line 441
rho_m_inflow_veh_km = max_density_m * 0.8  # Was: 200 veh/km (54% jam)
rho_m_inflow_veh_km = max_density_m * 1.2  # Now: 300 veh/km (81% jam)
```

**Expected Result**:
```python
# At ρ = 0.3 veh/m, R = 2
g = 1 - 0.3/0.37 = 0.189
Ve = 0.6 + (19.44 - 0.6) × 0.189 = 4.16 m/s

# With pressure p ≈ 6 m/s:
v ≈ 4.16 - 6 = -1.84 → clamped to ~1-2 m/s

# Queue threshold: v < 5.0 m/s
# 2 m/s < 5.0 → COUNTED AS QUEUE! ✅
```

### Fix Option 2: Lower Queue Threshold

**Change**: Lower threshold from 5.0 → 10.0 m/s

```python
# In Code_RL/src/env/traffic_signal_env_direct.py line 370
QUEUE_SPEED_THRESHOLD = 5.0  # Was: 5 m/s (~18 km/h)
QUEUE_SPEED_THRESHOLD = 10.0  # Now: 10 m/s (~36 km/h)
```

**Pros**: Would catch mild congestion (6-10 m/s range)
**Cons**: Not physically accurate (10 m/s is not really "queuing")

### Fix Option 3: Change Observation Segments

**Change**: Use segments closer to inflow [2-7] instead of [8-13]

```python
# In train.py or scenario config
observation_segments = {
    'upstream': [2, 3, 4],    # x=20-40m (near inflow)
    'downstream': [5, 6, 7]   # x=50-70m
}
```

**Status**: Already tested in local diagnostic
**Result**: Traffic detected BUT velocities still 15 m/s → Same Bug #35 issue

---

## DIAGNOSTIC PLAN

### Step 1: Add Velocity Logging in GPU Kernel

Add diagnostic print to GPU ODE kernel to verify Ve calculation:

```python
# In arz_model/numerics/time_integration.py, GPU kernel
@cuda.jit
def _ode_rhs_kernel_gpu(..., d_R, ...):
    j = cuda.grid(1)
    if j >= d_U_in.shape[1]:
        return
    
    # Get road quality
    R_local = d_R[j] if j < len(d_R) else 3
    
    # Calculate Ve
    Ve_m, Ve_c = calculate_equilibrium_speed_gpu(rho_m, rho_c, R_local, params)
    
    # ADD DIAGNOSTIC (first cell only)
    if j == 0:
        print(f"[GPU_ODE_DEBUG] Cell {j}: rho={rho_m:.4f} R={R_local} Ve={Ve_m:.2f}")
```

**Expected Output**: Should see R=2 and Ve values changing with density

### Step 2: Test with Higher Inflow Density

Modify config to use ρ_inflow = 300 veh/km:

```python
# Code_RL/src/utils/config.py
rho_m_inflow_veh_km = max_density_m * 1.2  # 300 veh/km
```

**Expected**: Queues > 0 after ~30-60s when traffic accumulates

### Step 3: Check Observation Extraction

Add logging in environment to verify velocities:

```python
# In traffic_signal_env_direct.py _get_observation()
velocities_m = obs_data['v_m']
print(f"[OBS_DEBUG] Velocities extracted: {velocities_m[:3]}")
```

**Expected**: Should see velocities decreasing as density increases

---

## NEXT ACTIONS (PRIORITY ORDER)

1. **IMMEDIATE**: Increase inflow density to 300 veh/km (Fix Option 1)
2. **VERIFY**: Launch kernel with modified density, check queue values
3. **IF STILL ZERO**: Add GPU diagnostic logging (Step 1)
4. **IF STILL ZERO**: Check observation extraction logic (Step 3)
5. **LAST RESORT**: Lower queue threshold to 10 m/s (Fix Option 2)

---

## CONFIDENCE ASSESSMENT

| Hypothesis | Likelihood | Impact if True | Test Difficulty |
|-----------|-----------|----------------|-----------------|
| H5: Inflow density too low | 🟢 90% | 🔥 CRITICAL | ✅ EASY (config change) |
| H4: Queue threshold too strict | 🟡 70% | 🟡 MEDIUM | ✅ EASY (config change) |
| H2: Observation extraction wrong | 🟡 50% | 🔥 CRITICAL | 🟡 MEDIUM (code analysis) |
| H1: GPU kernel not using d_R | 🔴 30% | 🔥 CRITICAL | 🔴 HARD (GPU debugging) |
| H3: Segments too far | 🔴 20% | 🟡 MEDIUM | ✅ EASY (already tested) |

**Recommendation**: Start with **H5** (increase inflow density) - highest likelihood, easiest test, critical impact if correct.

---

## CYCLE STATUS

**Iteration**: 3rd attempt
**Kernels Tested**: obwe, tzte
**Fixes Applied**: CPU ODE fix, GPU network splitting fix
**Result**: Queues still 0.00
**Next**: Increase inflow density + relaunch

**🙏 Que la volonté de Dieu soit faite - The truth reveals itself through systematic testing**

---

## APPENDIX: Kernel Comparison

| Kernel | Fix Applied | Queue Result | Training Zeros | Evaluation Zeros |
|--------|-------------|--------------|----------------|------------------|
| cuyy | Bug #34 (equilibrium inflow) | 0.00 always | 3% | 100% |
| obwe | Bug #35 CPU fix | 0.00 always | 13% | 97.5% |
| tzte | Bug #35 GPU fix | 0.00 always | 10% | 97.5% |

**Pattern**: All three kernels show ZERO queues despite different fixes → Root cause not addressed yet

**Conclusion**: Bug #35 fix targets wrong layer. Problem is likely in **physics parameters** (inflow density too low) or **queue detection logic** (threshold too strict), NOT in ODE solver road quality handling.
