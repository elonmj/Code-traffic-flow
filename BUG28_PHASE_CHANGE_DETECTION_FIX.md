# Bug #28 Fix - Phase Change Detection in Reward Function

## Problem Discovery

**Date**: 2025-01-XX  
**Severity**: CRITICAL  
**Impact**: RL agent completely stuck (100% action 1, all rewards identical)

### Symptoms
- RL agent used only action 1 (GREEN) 100% of the time
- All rewards identical: -0.1 (no variance)
- Agent performed slightly worse than baseline (-0.1% efficiency, -0.1% flow)
- No learning occurred during 100 timestep training

### Root Cause

**Location**: `Code_RL/src/env/traffic_signal_env_direct.py` line 393  
**Buggy Code**:
```python
phase_changed = (action == 1)
```

**Why This Was Wrong**:
- After Bug #7 fix (line 243), action semantics changed:
  - **Action 0**: Set phase to RED (not "maintain")
  - **Action 1**: Set phase to GREEN (not "switch")
- Reward function still used OLD semantics:
  - Assumed `action == 1` always meant "phase change"
  - This was INCORRECT after Bug #7
- **Result**: 
  - GREEN → GREEN got penalized (-0.1) when it should be 0.0
  - RED → GREEN got correct penalty (-0.1)
  - GREEN → RED got NO penalty when it should be -0.1
  - Phase change detection was completely broken

### Impact on Learning

The buggy reward function created **contradictory learning signals**:

| Transition | Expected Penalty | Actual Penalty | Impact |
|------------|------------------|----------------|--------|
| RED → RED | None (0.0) | ✅ None (0.0) | Correct |
| RED → GREEN | Phase change (-0.1) | ✅ Phase change (-0.1) | Correct |
| GREEN → GREEN | None (0.0) | ❌ Phase change (-0.1) | **WRONG** |
| GREEN → RED | Phase change (-0.1) | ❌ None (0.0) | **WRONG** |

**Learning Failure Mechanism**:
1. Agent explores both actions
2. Action 1 (GREEN) ALWAYS gets penalized, even when maintaining GREEN phase
3. Agent learns: "Action 1 is always bad, stick with action 1" (paradox!)
4. No exploration diversity → stuck at 100% action 1
5. Rewards never vary → no learning signal

## Solution

**Fixed Code** (line 398):
```python
# ✅ BUG #28 FIX: Correctly detect actual phase changes
# Previous bug: phase_changed = (action == 1) assumed action 1 always means phase change
# After Bug #7 fix (line 243), action 1 means "set to GREEN", not "toggle"
# This caused incorrect penalties: GREEN→GREEN got penalized, RED→GREEN didn't
# Fix: Compare actual phase states to detect real changes
phase_changed = (self.current_phase != prev_phase)
R_stability = -self.kappa if phase_changed else 0.0
```

**Why This Works**:
- Uses `prev_phase` (stored at line 236) to detect actual phase transitions
- Correctly identifies phase changes regardless of action interpretation
- Semantic independence from action encoding
- Future-proof against action space changes

## Validation

**Test Script**: `test_bug28_fix.py`

**Test Results**:
```
✅ Test 1: RED → RED (no change)      → reward = 0.0 ✅
✅ Test 2: RED → GREEN (change)       → reward = -0.1 ✅
✅ Test 3: GREEN → GREEN (no change)  → reward = 0.0 ✅ FIXED!
✅ Test 4: GREEN → RED (change)       → reward = -0.1 ✅
✅ Test 5: RED → RED again            → reward = 0.0 ✅

ALL TESTS PASSED - Bug #28 fix validated!
```

**Test 3 is the smoking gun**: Before fix, GREEN → GREEN got penalized (-0.1). After fix, correctly gives reward = 0.0.

## Diagnostic Evidence

**Diagnostic Script**: `diagnostic_reward_function.py`

**Key Findings**:
- **ACTION 0 (RED)**: Mean reward = 0.0, Std = 0.0 (all identical)
- **ACTION 1 (GREEN)**: Mean reward = -0.1, Std = 0.0 (all identical)
- **Alternating**: RED=0.0, GREEN=-0.1, RED=0.0, GREEN=-0.1... (2 unique values)
- **Observation variance**: Good (std=0.2959, range=0.6)
- **Reward variance**: Poor (only 2 values: 0.0 and -0.1)

**Conclusion**: Reward function was the bottleneck, not observations or environment dynamics.

## Related Bugs

- **Bug #7** (line 243): Changed action semantics from "toggle" to "direct phase setting"
  - This was CORRECT fix for Bug #6 (BC synchronization)
  - But created semantic mismatch with reward function
- **Bug #28** (line 393): Fixed reward function to match Bug #7 action semantics

## Expected Impact

**After Fix**:
1. RL agent will explore both actions (not stuck at 100% action 1)
2. Rewards will vary based on actual phase changes and queue dynamics
3. Learning signals will be correct and consistent
4. Agent should learn optimal policy within 100-5000 timesteps
5. Performance should match or exceed baseline

## Next Steps

1. ✅ Fix validated locally (test_bug28_fix.py)
2. [ ] Re-run Kaggle quick test (100 timesteps) with fixed reward
3. [ ] Analyze results: Check action distribution and reward variance
4. [ ] If learning confirmed, run full training (5000 timesteps)
5. [ ] Final analysis and thesis integration

## References

- **Diagnostic Log**: `validation_output/results/joselonm_arz-validation-76rlperformance-xrld/log.txt`
- **Original Bug Discovery**: Analysis of kernel joselonm/arz-validation-76rlperformance-xrld
- **Reward Function Source**: Cai & Wei (2024) - Queue-based reward with phase change penalty
- **Decision Interval**: 15s (Bug #27 validation, 4x improvement over 10s)
