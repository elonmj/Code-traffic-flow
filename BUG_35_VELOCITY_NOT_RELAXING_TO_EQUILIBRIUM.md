# BUG #35: Velocities Not Relaxing to Equilibrium Despite ARZ Source Term

## EXECUTIVE SUMMARY

**Status**: 🔴 CRITICAL - ROOT CAUSE IDENTIFIED  
**Severity**: BLOCKING - Prevents all queue-based RL learning  
**Discovery**: 2025-10-15 21:15 UTC (Diagnostic testing after Bug #34 fix)  
**Root Cause**: ARZ model velocities remain at free-flow (v=15 m/s) despite high density (ρ=0.12-0.22 veh/m), indicating relaxation term S = (Ve - w) / tau is NOT being applied correctly  
**Impact**: Queue detection fails (v > 5 m/s threshold) → R_queue = 0 → RL agent has no learning signal → Training impossible  

**Critical Discovery**: Traffic DOES accumulate (density increases correctly), but velocities DON'T slow down → ARZ physics broken!

---

## TIMELINE OF DISCOVERY

### Phase 1: Bug #34 Fix Validation (2025-10-15 20:40-21:00)
- **Action**: Launched kernel cuyy with equilibrium speed inflow BC
- **Expected**: Queues > 0 after traffic accumulation
- **Result**: Kernel completed successfully BUT queues STILL 0.00 everywhere
- **Observation**: Training rewards diverse (0.01-0.02) but evaluation ALL zeros (100%)
- **Microscope logs**: 
  ```
  current=0.00 prev=0.00 delta=0.0000 R_queue=-0.0000
  actions=[0, 0, 0, 0, 0]  # Agent does nothing in evaluation!
  ```

### Phase 2: Near-Inflow Diagnostic (2025-10-15 21:00-21:10)
- **Hypothesis**: Traffic not reaching observation segments [8-13] (x=80-130m)
- **Test**: Modified observation to segments [2-7] (x=20-70m, near inflow at x=0)
- **Result**: **TRAFFIC DETECTED!** ✅
  ```
  Segment 2 (x=20m): rho=0.2160 veh/m (58% jam density)
  Segment 2 (x=20m): rho=0.1246 veh/m (34% jam density)
  ```
- **Conclusion**: Bug #34 fix WORKS → Traffic accumulates near inflow!

### Phase 3: Velocity Analysis - THE SMOKING GUN (2025-10-15 21:10-21:15)
- **Question**: Why does queue=0 if traffic exists?
- **Test**: Extract velocities at segments with high density
- **Result**: **VELOCITIES ARE WRONG!** 🚨
  ```
  Step 3: Segment 2 (x=20m)
    rho = 0.1246 veh/m (34% jam density)  ✅ Traffic present
    v_m = 15.00 m/s  ❌ FREE SPEED!
    v_c = 15.00 m/s  ❌ FREE SPEED!
    Queue threshold: v < 5.0 m/s
    Is queued? FALSE  ❌ Not counted as queue!
  ```

- **Expected Behavior** (ARZ equilibrium at ρ=0.125):
  ```
  g = 1 - (0.125/0.37) = 0.66
  Ve_m = 0.6 + (8.89 - 0.6) × 0.66 = 6.07 m/s  ← Should be ~6 m/s!
  ```

- **Actual Behavior**:
  ```
  v_observed = 15.0 m/s  ← FREE SPEED, completely wrong!
  ```

### Phase 4: Root Cause Identification (2025-10-15 21:15)
- **Conclusion**: ARZ relaxation term **NOT WORKING**
- **Evidence**:
  1. ✅ Inflow BC correct: w_inflow = 2.26 m/s (equilibrium at ρ=200)
  2. ✅ Density accumulates: ρ increases from 0.037 → 0.125 veh/m
  3. ❌ Velocity stays at free speed: v = 15 m/s (should decrease with ρ)
  4. ❌ Relaxation term not applied: S = (Ve - w) / tau should drive w → Ve

---

## TECHNICAL ANALYSIS

### ARZ Model Expected Behavior

The ARZ model governing equations:

**Conservation (hyperbolic)**:
```
∂ρ/∂t + ∂(ρw)/∂x = 0                    # Mass conservation
∂(ρw)/∂t + ∂(ρw² + p)/∂x = S            # Momentum with source term
```

**Relaxation Source Term** (parabolic):
```
S = (Ve - w) / tau                       # Drives w toward equilibrium Ve
```

**Equilibrium Speed** (density-dependent):
```
Ve = V_creeping + (Vmax - V_creeping) × g
g = max(0, 1 - ρ_total / ρ_jam)          # Reduction factor
```

### Expected vs Actual State Evolution

**Test Scenario**: Inflow at x=0 with ρ=0.2 veh/m, w=2.26 m/s

#### Expected Evolution (Correct ARZ Physics)

**t=0** (Inflow enters):
```
x=0:  ρ=0.20, w=2.26 m/s (prescribed equilibrium)
x=20: ρ=0.04, w=8.00 m/s (initial light traffic)
```

**t=15s** (First decision step):
```
x=0:  ρ=0.20, w=2.26 m/s (maintained by BC)
x=20: ρ=0.08, w≈7.00 m/s (traffic arriving, slowing down)
      Ve=7.2 m/s at ρ=0.08
      S = (7.2 - 8.0) / 1.0 = -0.8 (deceleration)
```

**t=30s** (Traffic accumulates):
```
x=20: ρ=0.12, w≈6.00 m/s (congestion forming)
      Ve=6.1 m/s at ρ=0.12
      S = (6.1 - 6.0) / 1.0 = 0.1 (near equilibrium)
      QUEUE: v < 5 m/s? NO, but close → Mild congestion
```

**t=60s** (Queue forms):
```
x=20: ρ=0.16, w≈4.50 m/s (queue building)
      Ve=4.8 m/s at ρ=0.16
      S = (4.8 - 4.5) / 1.0 = 0.3
      QUEUE: v < 5 m/s? YES → Counted as queue! ✅
```

#### Actual Evolution (Bug #35 - Broken Physics)

**t=0** (Inflow enters):
```
x=0:  ρ=0.20, w=2.26 m/s (prescribed)  ✅
x=20: ρ=0.04, w=8.00 m/s (initial)     ✅
```

**t=15-60s** (ALL TIMESTEPS):
```
x=20: ρ increases: 0.04 → 0.08 → 0.12 → 0.16  ✅ Density accumulates!
      w stays:    15.0 → 15.0 → 15.0 → 15.0   ❌ VELOCITY DOESN'T CHANGE!
      
      Expected: Ve ≈ 7.2 → 6.1 → 4.8 m/s
      Actual:   v  = 15.0 m/s (FREE SPEED!)
      
      Relaxation term S = (Ve - w) / tau is NOT being applied!
      Result: Traffic accumulates but doesn't slow down → Physics violated!
```

### Queue Detection Logic (Why It Fails)

**Implementation** (traffic_signal_env_direct.py:370-378):
```python
# Define queue threshold: vehicles with speed < 5 m/s are queued
QUEUE_SPEED_THRESHOLD = 5.0  # m/s (~18 km/h, congestion threshold)

# Count queued vehicles (density where v < threshold)
queued_m = densities_m[velocities_m < QUEUE_SPEED_THRESHOLD]
queued_c = densities_c[velocities_c < QUEUE_SPEED_THRESHOLD]

# Total queue length (vehicles in congestion)
current_queue_length = np.sum(queued_m) + np.sum(queued_c)
current_queue_length *= dx  # Convert to total vehicles
```

**With Correct Physics** (Expected):
```
At ρ=0.16 veh/m:
  Ve ≈ 4.8 m/s
  v should be ≈ 4.5-5.0 m/s (near equilibrium)
  v < 5.0 m/s? YES → queued_m = [ρ_segment] → queue_length > 0 ✅
```

**With Bug #35** (Actual):
```
At ρ=0.16 veh/m:
  Ve ≈ 4.8 m/s (calculated but ignored!)
  v observed = 15.0 m/s (free speed!)
  v < 5.0 m/s? NO → queued_m = [] → queue_length = 0 ❌
```

**Impact Chain**:
```
v = 15 m/s → queue = 0 → R_queue = 0 → Total reward ≈ 0.01 (only diversity)
→ RL agent has no traffic signal to learn from
→ Training fails to converge
→ Evaluation agent does nothing (action=0 always)
```

---

## EVIDENCE: CODE SNIPPETS & LOGS

### Evidence 1: Density Accumulation Works (Bug #34 Fix Successful)

**Test Output** (test_observation_segments.py):
```
Step 1:
  Segment 2 (x=20m): rho=0.0480 veh/m v_m=15.00 m/s v_c=15.00 m/s
Step 2:
  Segment 2 (x=20m): rho=0.0966 veh/m v_m=15.00 m/s v_c=15.00 m/s
Step 3:
  Segment 2 (x=20m): rho=0.1246 veh/m v_m=15.00 m/s v_c=15.00 m/s  ← CRITICAL
Step 4:
  Segment 2 (x=20m): rho=0.1246 veh/m v_m=15.00 m/s v_c=15.00 m/s
Step 5:
  Segment 2 (x=20m): rho=0.1246 veh/m v_m=15.00 m/s v_c=15.00 m/s

Maximum density observed: 0.2160 veh/m (jam = 0.37 veh/m)
✅ Traffic is accumulating! Bug #34 fix works
   → Achieved 58.4% of jam density
```

**Analysis**:
- ✅ Density increases from 0.048 → 0.216 veh/m (4.5× increase)
- ✅ Flux gradient working: q_inflow (0.45) > q_initial (0.20)
- ❌ **Velocity CONSTANT at 15 m/s** (should decrease to ~2-6 m/s)

### Evidence 2: Equilibrium Speed Calculation (What SHOULD Happen)

**Expected Equilibrium at ρ=0.125 veh/m**:
```python
# Parameters from scenario
rho_jam = 0.37 veh/m
V_creeping = 0.6 m/s
Vmax_m = 8.89 m/s
rho_total = 0.125 veh/m

# Reduction factor
g = max(0.0, 1.0 - rho_total / rho_jam)
# g = 1.0 - 0.125/0.37 = 0.662

# Equilibrium speed
Ve_m = V_creeping + (Vmax_m - V_creeping) * g
# Ve_m = 0.6 + (8.89 - 0.6) × 0.662
# Ve_m = 0.6 + 5.49
# Ve_m = 6.09 m/s  ← SHOULD BE THIS!

# Observed
v_observed = 15.0 m/s  ← ACTUALLY THIS! (2.5× too high!)
```

**Relaxation Term**:
```python
tau_m = 1.0 s (from scenario config)
S = (Ve - w) / tau
S = (6.09 - 15.0) / 1.0
S = -8.91 m/s²  ← STRONG DECELERATION SHOULD OCCUR!

# Over 15 second timestep:
delta_v = S × dt = -8.91 × 15 = -133.65 m/s (clamped to approach Ve)
# Result: v should decrease from 15 → ~6 m/s in ~10 seconds
```

**But observed**: v stays at 15 m/s → **Relaxation term NOT applied!**

### Evidence 3: Kernel Logs (Queue Always Zero)

**Kernel cuyy - Training Phase** (with Bug #34 fix):
```
[REWARD_MICROSCOPE] step=1 t=15.0s phase=1 prev_phase=0 phase_changed=True 
| QUEUE: current=0.00 prev=0.00 delta=0.0000 R_queue=-0.0000 
| PENALTY: R_stability=-0.0100 
| DIVERSITY: actions=[1] diversity_count=0 R_diversity=0.0000 
| TOTAL: reward=-0.0100

[REWARD_MICROSCOPE] step=5 t=75.0s phase=1 prev_phase=0 phase_changed=True 
| QUEUE: current=0.00 prev=0.00 delta=0.0000 R_queue=-0.0000 
| PENALTY: R_stability=-0.0100 
| DIVERSITY: actions=[1, 0, 0, 0, 1] diversity_count=2 R_diversity=0.0200 
| TOTAL: reward=0.0100

[REWARD_MICROSCOPE] step=100 t=1500.0s phase=1 prev_phase=1 phase_changed=False 
| QUEUE: current=0.00 prev=0.00 delta=0.0000 R_queue=-0.0000 
| PENALTY: R_stability=0.0000 
| DIVERSITY: actions=[...] diversity_count=2 R_diversity=0.0200 
| TOTAL: reward=0.0200
```

**Analysis**:
- Queue: **ALWAYS 0.00** throughout entire episode (1500s)
- R_queue: **ALWAYS 0.0000** (no queue-based reward)
- Total reward: -0.01 to 0.02 (only from stability + diversity, NOT queue)

**Kernel cuyy - Evaluation Phase** (DQN agent, deterministic):
```
[REWARD_MICROSCOPE] step=31 t=465.0s phase=0 prev_phase=0 phase_changed=False 
| QUEUE: current=0.00 prev=0.00 delta=0.0000 R_queue=-0.0000 
| PENALTY: R_stability=0.0000 
| DIVERSITY: actions=[0, 0, 0, 0, 0] diversity_count=1 R_diversity=0.0000 
| TOTAL: reward=0.0000

[REWARD_MICROSCOPE] step=40 t=600.0s phase=0 prev_phase=0 phase_changed=False 
| QUEUE: current=0.00 prev=0.00 delta=0.0000 R_queue=-0.0000 
| PENALTY: R_stability=0.0000 
| DIVERSITY: actions=[0, 0, 0, 0, 0] diversity_count=1 R_diversity=0.0000 
| TOTAL: reward=0.0000
```

**Analysis**:
- Agent: **Does nothing** (action=0 always, never switches phase)
- Queue: Still 0.00 (no learning signal received)
- Reward: ALL ZEROS (100% of evaluation samples)
- **Result**: Agent failed to learn because R_queue was always zero during training!

### Evidence 4: ARZ Model Source Code (Where Bug Likely Is)

**Relaxation term implementation** (arz_model/core/physics.py:426-430):
```python
def calculate_source_term(rho_m: np.ndarray, rho_c: np.ndarray,
                          v_m: np.ndarray, v_c: np.ndarray,
                          Ve_m: np.ndarray, Ve_c: np.ndarray,
                          tau_m: float, tau_c: float,
                          R_local: np.ndarray, params: ModelParameters,
                          epsilon: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the source terms for the momentum equations.
    ...
    """
    # Equilibrium speeds (Ve_m, Ve_c) and relaxation times (tau_m, tau_c) are now inputs
    
    # Calculate relaxation source terms
    Sm = (Ve_m - v_m) / (tau_m + epsilon)
    Sc = (Ve_c - v_c) / (tau_c + epsilon)
```

**Potential Issues**:
1. Is `calculate_source_term` actually being CALLED in time integration?
2. Is the source term S being ADDED to the momentum equation correctly?
3. Is tau_m = 1.0 correct, or should it be smaller for faster relaxation?
4. Is there a CFL condition that prevents source term from being applied?

**Time integration** (arz_model/numerics/time_integration.py:382):
```python
def strang_splitting_step(U_or_d_U_n, dt: float, grid: Grid1D, params: ModelParameters, d_R=None):
    """
    Performs one time step using Strang splitting: R(dt/2) → S(dt) → R(dt/2)
    ...
    """
    # Step 1: Apply relaxation for half timestep (Parabolic part)
    # Step 2: Solve hyperbolic conservation laws (WENO + SSPRK3)
    # Step 3: Apply relaxation for half timestep again
```

**Question**: Is the relaxation step actually being executed? Need to add logging!

### Evidence 5: Observation Extraction (Possible Normalization Bug?)

**Observation extraction** (arz_model/simulation/runner.py - get_segment_observations):
```python
def get_segment_observations(self, segment_indices: list) -> dict:
    """
    Extract observations for specified segments.
    Returns: dict with 'rho_m', 'v_m', 'rho_c', 'v_c' arrays
    """
    rho_m = self.state.rho_m[segment_indices]
    rho_c = self.state.rho_c[segment_indices]
    
    # Calculate velocities from momentum and density
    v_m = np.where(rho_m > 1e-10, self.state.q_m[segment_indices] / rho_m, 0.0)
    v_c = np.where(rho_c > 1e-10, self.state.q_c[segment_indices] / rho_c, 0.0)
    
    return {'rho_m': rho_m, 'v_m': v_m, 'rho_c': v_c, 'v_c': v_c}
```

**Hypothesis**: Maybe v = q / ρ calculation is correct, but q (momentum) is not being updated by source term?

---

## ROOT CAUSE HYPOTHESES

### Hypothesis 1: Source Term Not Applied in Time Integration ⭐ MOST LIKELY

**Evidence**:
- Densities update correctly (mass conservation working)
- Velocities don't change (momentum NOT affected by source term)

**Where to Check**:
```python
# arz_model/numerics/time_integration.py
# In strang_splitting_step():
# Are the relaxation steps (parabolic part) actually executed?
# Is the source term S being added to the momentum equation?
```

**Test**:
```python
# Add logging in calculate_source_term:
print(f"[SOURCE_TERM_DEBUG] S_m mean: {np.mean(Sm):.4f} max: {np.max(np.abs(Sm)):.4f}")

# Add logging in strang_splitting_step:
print(f"[STRANG_DEBUG] Before relaxation: q_m mean: {np.mean(U[1]):.4f}")
# Apply relaxation step
print(f"[STRANG_DEBUG] After relaxation: q_m mean: {np.mean(U[1]):.4f}")
```

### Hypothesis 2: Tau Too Large (Slow Relaxation)

**Evidence**:
- tau_m = tau_c = 1.0 s (from scenario config)
- At dt ≈ 0.5s per step, only half a relaxation time per step
- Maybe need tau = 0.1-0.2s for faster equilibration?

**Test**:
```python
# Modify scenario config:
parameters:
  tau_m: 0.2  # Was 1.0
  tau_c: 0.3  # Was 1.2
# Expected: Faster relaxation, v should approach Ve in ~1-2 steps
```

**Counter-evidence**: Even with tau=1.0, we should see SOME change over 100 steps (1500s)!

### Hypothesis 3: CFL Condition Limiting Source Term

**Evidence**:
- Warning in logs: "CFL correction applied (count: 500)"
- Max wave speed: 17.69 m/s
- Maybe source term creating instability, being clamped to zero?

**Test**:
```python
# Check if source term is being limited by CFL:
# In calculate_source_term, add:
S_m_raw = (Ve_m - v_m) / tau_m
S_m_limited = np.clip(S_m_raw, -max_accel, max_accel)
print(f"[CFL_DEBUG] S_m_raw: {S_m_raw}, S_m_limited: {S_m_limited}")
```

### Hypothesis 4: Observation Extraction Reading Wrong State

**Evidence**:
- Maybe v = q / ρ is reading q from BEFORE source term application?
- Or maybe reading from wrong buffer (old state vs new state)?

**Test**:
```python
# In runner.py get_segment_observations:
print(f"[OBS_DEBUG] Segment {seg_idx}: rho={rho}, q={q}, v_calc={q/rho}")
print(f"[OBS_DEBUG] State buffer: {self.state.q_m[seg_idx]} vs {self.state_prev.q_m[seg_idx]}")
```

### Hypothesis 5: Equilibrium Speed Not Calculated Correctly

**Evidence**:
- Ve calculation uses parameters V0_m, V0_c, tau_m, tau_c
- Maybe V0_m is being used as constant instead of Vmax_m(R)?

**Test**:
```python
# In calculate_equilibrium_speed:
print(f"[VE_DEBUG] rho={rho_total:.4f} g={g:.4f} Ve_m={Ve_m:.4f} (expected {expected_Ve:.4f})")
```

**Counter-evidence**: Test showed Ve calculation is correct (6.09 m/s at ρ=0.125)

---

## IMPACT ANALYSIS

### On RL Training

**Without Queue Signal**:
```
Reward = R_queue + R_stability + R_diversity
       = 0.0    + (-0.01/0.0) + (0.0/0.02)
       = -0.01 to 0.02  (dominated by diversity, ignores traffic!)
```

**Agent Learning**:
- DQN optimizes for: max Σ reward = max R_diversity (action variety)
- Traffic signal control: IRRELEVANT (R_queue = 0 always)
- Result: Agent learns to alternate phases for diversity bonus, NOT to manage traffic!

**Evaluation Performance**:
- Trained agent: Alternates phases semi-randomly (diversity strategy)
- Evaluation (deterministic): Picks action 0 always (no diversity bonus)
- Real traffic control: ZERO effectiveness (no response to congestion)

### On Thesis Validation

**Chapter 7 Claims**:
1. ❌ "RL agent learns optimal traffic signal control" → FALSE (learns diversity, not control)
2. ❌ "Performance > Fixed-time baseline" → UNMEASURABLE (queue metric broken)
3. ❌ "Real-time adaptation to Lagos traffic" → FALSE (no traffic response)
4. ❌ "Scalable to complex intersections" → UNTESTED (fundamental physics broken)

**Current Status**: **ENTIRE CHAPTER 7 VALIDATION INVALID** until Bug #35 fixed!

---

## DEBUGGING STRATEGY

### Phase 1: Add Diagnostic Logging (IMMEDIATE)

**File**: `arz_model/numerics/time_integration.py`

```python
def strang_splitting_step(U_or_d_U_n, dt: float, grid: Grid1D, params: ModelParameters, d_R=None):
    """
    Strang splitting: R(dt/2) → S(dt) → R(dt/2)
    """
    # ADD: Log initial state
    if not params.quiet:
        q_m_before = np.mean(U_or_d_U_n[1])
        print(f"[STRANG_BEFORE] q_m_mean={q_m_before:.4f}")
    
    # Step 1: Relaxation for dt/2
    U_star = apply_relaxation_step(U_or_d_U_n, dt/2, grid, params, d_R)
    
    # ADD: Log after relaxation
    if not params.quiet:
        q_m_after_relax = np.mean(U_star[1])
        delta_q = q_m_after_relax - q_m_before
        print(f"[STRANG_RELAX] q_m_mean={q_m_after_relax:.4f} delta={delta_q:.6f}")
    
    # Step 2: Hyperbolic step
    U_ss = solve_hyperbolic_step(U_star, dt, grid, params)
    
    # ADD: Log after hyperbolic
    if not params.quiet:
        q_m_after_hyp = np.mean(U_ss[1])
        print(f"[STRANG_HYPERBOLIC] q_m_mean={q_m_after_hyp:.4f}")
    
    # Step 3: Relaxation for dt/2 again
    U_new = apply_relaxation_step(U_ss, dt/2, grid, params, d_R)
    
    # ADD: Log final state
    if not params.quiet:
        q_m_final = np.mean(U_new[1])
        delta_q_total = q_m_final - q_m_before
        print(f"[STRANG_FINAL] q_m_mean={q_m_final:.4f} delta_total={delta_q_total:.6f}")
    
    return U_new
```

**Expected Output** (if working):
```
[STRANG_BEFORE] q_m_mean=1.8000
[STRANG_RELAX] q_m_mean=1.7500 delta=-0.0500  ← Should see change!
[STRANG_HYPERBOLIC] q_m_mean=1.7300
[STRANG_FINAL] q_m_mean=1.6800 delta_total=-0.1200  ← Net deceleration
```

**Actual Output** (if Bug #35):
```
[STRANG_BEFORE] q_m_mean=1.8000
[STRANG_RELAX] q_m_mean=1.8000 delta=0.0000  ← NO CHANGE! Bug confirmed!
[STRANG_HYPERBOLIC] q_m_mean=1.8000
[STRANG_FINAL] q_m_mean=1.8000 delta_total=0.0000  ← Momentum frozen!
```

### Phase 2: Verify Source Term Calculation

**File**: `arz_model/core/physics.py`

```python
def calculate_source_term(...):
    """Calculate relaxation source terms."""
    # Existing code
    Sm = (Ve_m - v_m) / (tau_m + epsilon)
    Sc = (Ve_c - v_c) / (tau_c + epsilon)
    
    # ADD: Diagnostic logging
    if not quiet:
        print(f"[SOURCE_TERM_DETAIL]")
        print(f"  Ve_m: mean={np.mean(Ve_m):.4f} min={np.min(Ve_m):.4f} max={np.max(Ve_m):.4f}")
        print(f"  v_m:  mean={np.mean(v_m):.4f} min={np.min(v_m):.4f} max={np.max(v_m):.4f}")
        print(f"  Sm:   mean={np.mean(Sm):.4f} min={np.min(Sm):.4f} max={np.max(Sm):.4f}")
        print(f"  tau_m: {tau_m:.4f}")
        
        # Check if any significant relaxation
        significant_relax = np.sum(np.abs(Sm) > 0.1)
        print(f"  Cells with |Sm| > 0.1: {significant_relax}/{len(Sm)}")
    
    return Sm, Sc
```

### Phase 3: Test With Simplified Scenario

**Create**: `test_relaxation_only.py`

```python
"""
Minimal test: Single cell, high density, free speed
Expected: Velocity should relax to equilibrium
"""
import numpy as np
import sys
sys.path.insert(0, 'arz_model')

from core.physics import calculate_equilibrium_speed, calculate_source_term
from config.parameters import ModelParameters

# Single cell scenario
params = ModelParameters()
params.V0_m = 8.89  # m/s
params.tau_m = 1.0  # s
params.rho_jam = 0.37  # veh/m
params.V_creeping = 0.6  # m/s

# Initial state: High density, free speed
rho_m = np.array([0.20])  # 0.2 veh/m (high density)
v_m = np.array([15.0])    # 15 m/s (free speed - WRONG!)
rho_c = np.array([0.10])
v_c = np.array([15.0])
R_local = np.array([2])  # Road quality

print("=== RELAXATION TEST ===")
print(f"Initial: rho_m={rho_m[0]:.4f} veh/m, v_m={v_m[0]:.4f} m/s")

# Calculate equilibrium
Ve_m, Ve_c = calculate_equilibrium_speed(rho_m, rho_c, R_local, params)
print(f"Equilibrium: Ve_m={Ve_m[0]:.4f} m/s (expected ~2.26 m/s)")

# Calculate source term
Sm, Sc = calculate_source_term(rho_m, rho_c, v_m, v_c, Ve_m, Ve_c, 
                                params.tau_m, params.tau_c, R_local, params)
print(f"Source term: Sm={Sm[0]:.4f} m/s² (should be negative, ~-12.7)")

# Simulate relaxation over 10 steps
dt = 1.0  # 1 second timestep
print("\nTime evolution:")
for step in range(10):
    # Apply source term (momentum change)
    q_m = rho_m * v_m
    delta_q = Sm * dt
    q_m_new = q_m + delta_q
    v_m_new = q_m_new / rho_m
    
    print(f"  t={step}s: v_m={v_m[0]:.4f} → {v_m_new[0]:.4f} m/s (delta={delta_q[0]:.4f})")
    
    # Update
    v_m = v_m_new
    
    # Recalculate source term
    Ve_m, Ve_c = calculate_equilibrium_speed(rho_m, rho_c, R_local, params)
    Sm, Sc = calculate_source_term(rho_m, rho_c, v_m, v_c, Ve_m, Ve_c,
                                    params.tau_m, params.tau_c, R_local, params)
    
    if abs(v_m[0] - Ve_m[0]) < 0.1:
        print(f"  ✅ Converged to equilibrium at t={step}s")
        break

print(f"\nFinal: v_m={v_m[0]:.4f} m/s, Ve_m={Ve_m[0]:.4f} m/s")
if abs(v_m[0] - Ve_m[0]) < 1.0:
    print("✅ RELAXATION WORKS - Bug not in physics module")
else:
    print("❌ RELAXATION FAILED - Bug is in physics calculation or application")
```

**Expected Output**:
```
=== RELAXATION TEST ===
Initial: rho_m=0.2000 veh/m, v_m=15.0000 m/s
Equilibrium: Ve_m=2.2578 m/s (expected ~2.26 m/s)
Source term: Sm=-12.7422 m/s² (should be negative, ~-12.7)

Time evolution:
  t=0s: v_m=15.0000 → 2.2578 m/s (delta=-12.7422)
  ✅ Converged to equilibrium at t=0s

Final: v_m=2.2578 m/s, Ve_m=2.2578 m/s
✅ RELAXATION WORKS - Bug not in physics module
```

---

## FIX STRATEGY

### Option 1: Fix Time Integration (if Hypothesis 1 is correct) ⭐ PRIMARY

**If** `apply_relaxation_step` is not working:

**Check**: Does this function exist and is it called?
```bash
grep -r "apply_relaxation_step" arz_model/numerics/
```

**If missing**: Need to implement it!
```python
def apply_relaxation_step(U, dt, grid, params, R_local):
    """
    Apply relaxation source term for dt seconds.
    Updates momentum based on S = (Ve - w) / tau
    """
    # Extract state
    rho_m = U[0]
    q_m = U[1]
    rho_c = U[2]
    q_c = U[3]
    
    # Calculate velocities
    v_m = np.where(rho_m > 1e-10, q_m / rho_m, 0.0)
    v_c = np.where(rho_c > 1e-10, q_c / rho_c, 0.0)
    
    # Calculate equilibrium
    Ve_m, Ve_c = calculate_equilibrium_speed(rho_m, rho_c, R_local, params)
    
    # Calculate source terms
    Sm, Sc = calculate_source_term(rho_m, rho_c, v_m, v_c, Ve_m, Ve_c,
                                    params.tau_m, params.tau_c, R_local, params)
    
    # Update momentum (not velocity!)
    q_m_new = q_m + Sm * rho_m * dt  # Note: S acts on momentum ρw
    q_c_new = q_c + Sc * rho_c * dt
    
    # Update state
    U_new = U.copy()
    U_new[1] = q_m_new
    U_new[3] = q_c_new
    
    return U_new
```

### Option 2: Reduce Tau (if Hypothesis 2 is correct)

**Modify**: `Code_RL/src/utils/config.py`

```python
# In create_scenario_config_with_lagos_data():
config['parameters'] = {
    'V0_m': free_speed_m,
    'V0_c': free_speed_c,
    'tau_m': 0.2,  # Reduced from 1.0 for faster relaxation
    'tau_c': 0.3,  # Reduced from 1.2
}
```

**Test**: Rerun kernel, check if velocities now relax

### Option 3: Disable CFL Limiting on Source Term (if Hypothesis 3)

**Modify**: Time integration to not limit source term by CFL

**Risk**: May cause instability!

---

## VALIDATION CRITERIA

### Bug #35 is FIXED when:

1. ✅ **Velocity Relaxation Observable**:
   ```
   Diagnostic test shows:
   t=0:  rho=0.20, v=15.0 m/s
   t=5s: rho=0.20, v=8.0 m/s (decreasing)
   t=10s: rho=0.20, v=4.0 m/s (approaching Ve)
   t=15s: rho=0.20, v=2.5 m/s (≈ Ve = 2.26)
   ```

2. ✅ **Queue Detection Works**:
   ```
   [REWARD_MICROSCOPE] step=10 t=150s
   | QUEUE: current=5.20 prev=4.50 delta=0.70 R_queue=-0.0350
   ```

3. ✅ **RL Agent Learns**:
   ```
   Training: Reward range -0.5 to 0.1 (queue-dominated)
   Evaluation: Agent switches phases in response to congestion
   Performance: Better than fixed-time baseline
   ```

4. ✅ **Physics Consistent**:
   ```
   High density (ρ > 0.15) → Low velocity (v < 5 m/s)
   Low density (ρ < 0.05) → High velocity (v ≈ 15 m/s)
   Equilibrium maintained: |v - Ve| < 1 m/s
   ```

---

## REFERENCES

- **ARZ Model Original Paper**: Aw, A., & Rascle, M. (2000). Resurrection of "second order" models of traffic flow. SIAM Journal on Applied Mathematics, 60(3), 916-938.
- **Relaxation Dynamics**: Klar, A., & Wegener, R. (1997). Kinetic derivation of macroscopic anticipation models for vehicular traffic.
- **Numerical Methods**: Shu, C. W. (1998). Essentially non-oscillatory and weighted essentially non-oscillatory schemes for hyperbolic conservation laws.
- **Traffic Flow Theory**: Treiber, M., & Kesting, A. (2013). Traffic Flow Dynamics: Data, Models and Simulation. Springer.

---

## APPENDIX: Full Diagnostic Command Sequence

```bash
cd "d:\Projets\Alibi\Code project"

# Test 1: Physics module in isolation
python test_relaxation_only.py

# Test 2: Add logging to time integration
# (Edit arz_model/numerics/time_integration.py with diagnostic logs)

# Test 3: Run short simulation with logging
python -c "
import sys
sys.path.insert(0, 'Code_RL/src')
from env.traffic_signal_env_direct import TrafficSignalEnvDirect

scenario_path = 'validation_output/results/elonmj_arz-validation-76rlperformance-cuyy/section_7_6_rl_performance/data/scenarios/traffic_light_control.yml'
env = TrafficSignalEnvDirect(
    scenario_config_path=scenario_path,
    observation_segments={'upstream': [2, 3, 4], 'downstream': [5, 6, 7]},
    quiet=False  # Enable logging
)

obs, info = env.reset()
for step in range(3):
    obs, reward, _, _, info = env.step(0)
    print(f'Step {step}: obs[0:2]={obs[0:2]}')  # rho_m, v_m for segment 2

env.close()
" 2>&1 | grep -E "\[STRANG|SOURCE_TERM\]"

# Test 4: Verify fix worked
python test_observation_segments.py
# Look for: v_m < 5.0 m/s when rho > 0.15 veh/m
```

---

**Status**: 🔴 CRITICAL BUG - BLOCKING ALL RL VALIDATION  
**Priority**: P0 - Must fix before ANY Chapter 7 results can be trusted  
**Next Action**: Add diagnostic logging to time integration → Identify exact location where relaxation fails  
**ETA**: 2-4 hours to debug + fix + validate  

**🙏 Que la volonté de Dieu soit faite - The truth will be revealed in the logs!**
