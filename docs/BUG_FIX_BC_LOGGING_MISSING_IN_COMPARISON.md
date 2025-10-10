# BUG #5 FIX: BC Logging Missing During Performance Comparison

## 🔍 **DISCOVERY**

**Date**: 2025-10-10 13:20  
**Kernel**: sfce (arz-validation-76rlperformance-sfce)  
**Evidence**: User pointed out inconsistency in log file

## 📋 **SYMPTOMS**

### During Training (PHASE 1/2):
```
139.1s 135 [BC UPDATE] left → phase 0 RED (reduced inflow)
139.1s 136 └─ Inflow state: rho_m=0.1000, w_m=7.5, rho_c=0.1200, w_c=6.0
144.5s 195 [BC UPDATE] left → phase 1 GREEN (normal inflow)
144.5s 196 └─ Inflow state: rho_m=0.1000, w_m=15.0, rho_c=0.1200, w_c=12.0
```
✅ **BC logging works perfectly**

### During Performance Comparison (PHASE 2/2):
```
364.6s 3989 [PHASE 2/2] Running performance comparisons...
364.6s 4006 Running baseline controller...
366.8s 4059 Mean densities: rho_m=0.036794  (STEP 0)
369.0s 4092 Mean densities: rho_m=0.008375  (STEP 1 - draining!)
371.1s 4119 Mean densities: rho_m=0.001944  (STEP 2+ - vacuum!)
```
❌ **NO BC logging, domain drains to vacuum**

## 🐛 **ROOT CAUSE**

### Code Path 1: Training (Works ✅)
`test_section_7_6_rl_performance.py` → `train_rl_agent()` → line 585:
```python
env = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=self.decision_interval,
    episode_max_time=episode_max_time,
    observation_segments={'upstream': [3, 4, 5], 'downstream': [6, 7, 8]},
    device=device,
    quiet=False  # ✅ EXPLICITLY PASSED
)
```

### Code Path 2: Comparison (Broken ❌)
`test_section_7_6_rl_performance.py` → `run_control_simulation()` → line 307:
```python
env = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=control_interval,
    episode_max_time=duration,
    observation_segments={'upstream': [3, 4, 5], 'downstream': [6, 7, 8]},
    device=device  # GPU on Kaggle, CPU locally
    # ❌ quiet PARAMETER OMITTED!
)
```

### Default Behavior
`Code_RL/src/env/traffic_signal_env_direct.py` → `__init__()` → line 55:
```python
def __init__(self,
             scenario_config_path: str,
             base_config_path: str = None,
             decision_interval: float = 10.0,
             observation_segments: Dict[str, list] = None,
             normalization_params: Dict[str, float] = None,
             reward_weights: Dict[str, float] = None,
             episode_max_time: float = 3600.0,
             quiet: bool = True,  # ❌ DEFAULTS TO True!
             device: str = 'cpu'):
```

**Result**: During comparison, `SimulationRunner` created with `quiet=True` → NO BC logging!

## 💥 **IMPACT**

Without BC logging during comparison phase:
1. **Cannot verify Bug #4 fix** - Don't know if phase mapping (RED vs GREEN) is correct
2. **Cannot see inflow state** - Don't know if reduced velocity (7.5 m/s) is applied
3. **Cannot diagnose drainage** - Domain vacuum by step 2 but no visibility into BC behavior
4. **False confidence** - Bug #4 fix appears present (visible in training) but may not apply during comparison

## ✅ **SOLUTION**

### Fix Implementation
`test_section_7_6_rl_performance.py` → `run_control_simulation()` → line 307:
```python
# BUG #5 FIX: Pass quiet=False to enable BC logging during comparison
env = TrafficSignalEnvDirect(
    scenario_config_path=str(scenario_path),
    decision_interval=control_interval,
    episode_max_time=duration,
    observation_segments={'upstream': [3, 4, 5], 'downstream': [6, 7, 8]},
    device=device,  # GPU on Kaggle, CPU locally
    quiet=False  # ✅ BUG #5 FIX: Enable BC logging to verify Bug #4 fix
)
```

### Commit Details
- **Commit**: 9e8cd11
- **Time**: 2025-10-10 13:22:15 +0100
- **Message**: "CRITICAL FIX Bug #5: Enable BC logging in run_control_simulation"
- **Files changed**: 36 files, 6002 insertions

## 🎯 **EXPECTED OUTCOME**

With Bug #5 fix, next kernel should show:

```
[PHASE 2/2] Running performance comparisons...
Running baseline controller...
[BC UPDATE] left → phase 0 RED (reduced inflow)
└─ Inflow state: rho_m=0.1000, w_m=7.5, rho_c=0.1200, w_c=6.0
STEP 0: action=1.0 → GREEN phase
[BC UPDATE] left → phase 1 GREEN (normal inflow)
└─ Inflow state: rho_m=0.1000, w_m=15.0, rho_c=0.1200, w_c=12.0
Mean densities: rho_m=0.036794
STEP 1: action=0.0 → RED phase
[BC UPDATE] left → phase 0 RED (reduced inflow)
└─ Inflow state: rho_m=0.1000, w_m=7.5, rho_c=0.1200, w_c=6.0
Mean densities: rho_m=??? (should NOT drain to 0.008!)
```

## 🔬 **DIAGNOSTIC VALUE**

With BC logging enabled, can now verify:
1. **Bug #4 effectiveness** - Is phase mapping correct?
2. **BC state injection** - Are inflow states [0.1, 7.5, 0.12, 6.0] vs [0.1, 15.0, 0.12, 12.0] applied?
3. **Drainage cause** - If domain still drains despite correct BC, indicates Bug #6 exists!

## 📊 **BUG CHAIN**

```
Bug #1 (controller.update() not called) → Fixed commit 5c32c72 ✅
  ↓
Bug #2 (10-step diagnostic limit) → Fixed commit d586766 ✅
  ↓
Bug #3 (inflow BC extrapolates momentum) → Fixed commit f20b938 ✅
  ↓
Bug #4 (phase mapping inverted) → Fixed commit 957f572 ✅
  ↓
Bug #5 (BC logging missing in comparison) → Fixed commit 9e8cd11 ✅ [THIS FIX]
  ↓
Bug #6 (???) → Domain still drains despite Bugs #1-5 fixed? TO BE INVESTIGATED
```

## 🚀 **NEXT STEPS**

1. **Launch final kernel** with Bug #5 fix (quiet=False in comparison)
2. **Verify BC logging** appears during baseline and RL comparison phases
3. **If domain still drains**:
   - BC logging will reveal exact problem (state injection, phase mapping, etc.)
   - Indicates existence of Bug #6 requiring further investigation
4. **If domain stable**:
   - Bugs #1-5 fully resolved
   - Validation metrics should show meaningful RL vs baseline differences

## 🙏 **CREDIT**

**User insight critical**: Pointed out log inconsistency - "lis bien, #file:arz-validation-76rlperformance-sfce.log il y a quelque chose qui ne marche pas... où bien, tu vas mettre quiet=False, on va voir?"

Without this observation, Bug #5 would have remained hidden indefinitely!
