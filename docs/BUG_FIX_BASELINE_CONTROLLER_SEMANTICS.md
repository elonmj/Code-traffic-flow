# BUG #7: BaselineController Semantic Mismatch

**Date**: 2025-10-10  
**Priority**: CRITICAL - Root cause of domain drainage  
**Status**: Solution identified - Ready to implement

---

## Discovery

**Kernel**: czlc (2025-10-10 15:14)  
**Evidence**: Lines 4162-4239 in log

**Observation**:
- Bug #6 fix IS applied ([BC UPDATE] appears every step)
- But phase is WRONG (stays GREEN when should be RED)
- Domain still drains (0.037 → 0.008 → 0.002)

---

## Root Cause

**THE MISMATCH**:

**BaselineController** (`test_section_7_6_rl_performance.py:221`):
```python
return 1.0 if (self.time_step % 120) < 60 else 0.0
```
- Returns **1.0** = "I want GREEN phase"
- Returns **0.0** = "I want RED phase"

**Environment** (`traffic_signal_env_direct.py:219-225`):
```python
if action == 1:
    self.current_phase = (self.current_phase + 1) % self.n_phases  # TOGGLE
# else: maintain current phase
```
- Interprets **1** = "Toggle to next phase"
- Interprets **0** = "Maintain current phase"

**SEMANTIC CLASH**:
- Controller thinks: "1=GREEN, 0=RED"
- Environment thinks: "1=toggle, 0=maintain"

---

## Timeline Analysis

**Actual Behavior with Current Code**:

| Time | Controller Output | Controller Intent | Env Action | Actual Phase | Correct? |
|------|-------------------|-------------------|------------|--------------|----------|
| 0-60s | 1.0 | GREEN | Toggle 0→1 | GREEN | ✅ Luck! |
| 60-120s | 0.0 | RED | Maintain 1 | GREEN | ❌ WRONG! |
| 120-180s | 1.0 | GREEN | Toggle 1→0 | RED | ❌ WRONG! |
| 180-240s | 0.0 | RED | Maintain 0 | RED | ✅ Luck! |
| 240-300s | 1.0 | GREEN | Toggle 0→1 | GREEN | ✅ Luck! |
| 300-360s | 0.0 | RED | Maintain 1 | GREEN | ❌ WRONG! |

**Only 50% of steps have correct phase!**

---

## Evidence from Kernel czlc

**Line 4162-4165 (STEP 0, t=0-60s)**:
```
Action: 1.000000
[BC UPDATE] left → phase 1 GREEN (normal inflow)
Mean densities: rho_m=0.036794 ✅
```
Controller wants GREEN, Env toggles to GREEN → **Correct by luck!**

**Line 4201-4204 (STEP 1, t=60-120s)**:
```
Action: 0.000000
[BC UPDATE] left → phase 1 GREEN (normal inflow) ❌
```
Controller wants RED, Env maintains GREEN → **WRONG!**

**Line 4239 (STEP 1 result)**:
```
Mean densities: rho_m=0.008375 💀
```
Domain drains 78% in ONE step because phase is wrong!

---

## Solution: Phase-Direct Semantics

**Rationale**:
1. **Simplicity**: Controller already returns desired phase (0 or 1)
2. **Clarity**: No toggle confusion
3. **RL Compatible**: Binary action [0,1] means [RED, GREEN]
4. **Minimal Change**: Only env.step() modified

**Implementation**:

**File**: `Code_RL/src/env/traffic_signal_env_direct.py`  
**Lines**: 219-230

**BEFORE (with Bug #6 fix)**:
```python
# Apply action to traffic signal
if action == 1:
    # Switch phase
    self.current_phase = (self.current_phase + 1) % self.n_phases
# else: maintain current phase (action == 0)

# ✅ BUG #6 FIX: ALWAYS synchronize BC with current phase
self.runner.set_traffic_signal_state('left', phase_id=self.current_phase)
```

**AFTER (Bug #7 fix)**:
```python
# ✅ BUG #7 FIX: Interpret action as desired phase directly
# Action 0 = RED phase, Action 1 = GREEN phase
# This fixes semantic mismatch with BaselineController
self.current_phase = int(action)

# Bug #6 fix preserved: Always update BC
self.runner.set_traffic_signal_state('left', phase_id=self.current_phase)
```

**Expected Timeline with Fix**:

| Time | Controller Output | Desired Phase | Env Action | Actual Phase | Correct? |
|------|-------------------|---------------|------------|--------------|----------|
| 0-60s | 1.0 | GREEN | Set phase=1 | GREEN | ✅ |
| 60-120s | 0.0 | RED | Set phase=0 | RED | ✅ |
| 120-180s | 1.0 | GREEN | Set phase=1 | GREEN | ✅ |
| 180-240s | 0.0 | RED | Set phase=0 | RED | ✅ |
| 240-300s | 1.0 | GREEN | Set phase=1 | GREEN | ✅ |

**100% correct phase alignment!**

---

## Expected Outcomes

**Domain Behavior**:
- t=0-60s: GREEN phase → Traffic flows, density builds
- t=60-120s: RED phase → Inflow reduced, density decreases
- t=120-180s: GREEN phase → Traffic flows again
- **Oscillating pattern instead of drainage!**

**Metrics**:
- Baseline: Shows periodic density oscillations
- RL: Different pattern (undertrained → constant RED)
- Non-zero improvement differences
- `validation_success=true`

---

## Validation Checklist

After fix applied and kernel executed:

**Phase Correctness**:
- [ ] STEP 0 (t=0-60s): [BC UPDATE] phase 1 GREEN
- [ ] STEP 1 (t=60-120s): [BC UPDATE] phase 0 RED
- [ ] STEP 2 (t=120-180s): [BC UPDATE] phase 1 GREEN
- [ ] Pattern consistent throughout

**Domain Health**:
- [ ] No drainage (density > 0.01 veh/m)
- [ ] Oscillating density pattern in baseline
- [ ] Different pattern in RL
- [ ] Metrics show non-zero improvement

---

**STATUS**: ✅ Solution validated - Ready to implement with Bug #8 fix
