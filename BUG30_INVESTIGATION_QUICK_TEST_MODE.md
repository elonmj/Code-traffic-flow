# Bug #30: Kernel wncg Quick Test Mode - 2 Timesteps Investigation

## Status
🔴 **CRITICAL BUG** - RL Agent Not Training

## Discovery Date
2025-10-15 16:45 UTC

## Problem Summary
Kernel `joselonm/arz-validation-76rlperformance-wncg` completed successfully but produced **ALL ZERO REWARDS** because it trained with **ONLY 2 TIMESTEPS** instead of the intended 100 timesteps for quick test mode.

## Evidence

### Kernel Log Contradiction (Line 59-60)
```
Line 59: "2025-10-15 15:36:09 - INFO - train_rl_agent:1033 -   - Total timesteps: 100\n"
Line 37: "- Training: 2 timesteps only\n"
```

**Analysis:** The validation script CLAIMS it will train for 100 timesteps but the QUICK_TEST_MODE overrides this to 2 timesteps.

### Results Analysis
```bash
$ python analyze_bug29_results.py validation_output/results/joselonm_arz-validation-76rlperformance-wncg/section_7_6_rl_performance

📊 Total reward entries: 40
📊 Unique reward values: 1
📊 Min reward: 0.000000
📊 Max reward: 0.000000
📊 Mean reward: 0.000000
📊 Action: 0.000000 (100% of the time)
```

**All rewards are ZERO, all actions are ZERO** - This is worse than kernel wblw which had -0.1 rewards!

### Kernel Configuration (Lines 28-30)
```json
{"stream_name":"stdout","time":98.332827997,"data":"[STEP 3/4] Running validation tests...\n"}
{"stream_name":"stdout","time":98.333127273,"data":"[QUICK_TEST] Quick test mode enabled (100 timesteps for RL, reduced simulation time)\n"}
{"stream_name":"stdout","time":98.33321823,"data":"[SCENARIO] Single scenario mode: traffic_light_control\n"}
```

The message says "100 timesteps for RL" but internally the script uses 2 timesteps.

### Critical Discrepancy (Lines 36-40)
```json
Line 36: "QUICK TEST MODE ENABLED\n"
Line 37: "- Training: 2 timesteps only\n"          ← ACTUAL TRAINING
Line 38: "- Duration: 2 minutes simulated time\n"
Line 39: "- Scenarios: 1 scenario only\n"
Line 40: "- Expected runtime: ~5 minutes on GPU\n"
```

## Root Cause Analysis

### Hypothesis 1: Script Configuration Conflict ⭐ **MOST LIKELY**
The validation script has TWO DIFFERENT quick test configurations:
1. **CLI/Kaggle level**: `--quick-test` → 100 timesteps
2. **Internal test level**: `QUICK_TEST_MODE` → 2 timesteps

**Evidence:**
- Kernel log shows "Quick test mode enabled (100 timesteps for RL)"
- But then prints "Training: 2 timesteps only"
- This suggests an internal override in `test_section_7_6_rl_performance.py`

### Hypothesis 2: Environment Variable Override
Kaggle might set an internal `QUICK_TEST_MODE=True` that overrides CLI args.

### Hypothesis 3: Git Commit Issue
Kernel might have cloned an outdated commit with the 2-timestep configuration.

## Impact Assessment

### ❌ Kernel wncg Results: INVALID
- Cannot validate Bug #29 fix with 2 timesteps
- Zero rewards do NOT indicate Bug #29 failure - indicate no training occurred
- All previous analysis comparing to kernels wblw/xrld is INVALID

### ✅ Git Commit e004042: VERIFIED
Bug #29 code changes ARE correctly committed and pushed:
```bash
$ git show e004042 --stat
commit e004042...
Code_RL/src/env/traffic_signal_env_direct.py | 45 ++++++++++++++-----------
```

### 🟡 Bug #29 Status: UNVALIDATED
We STILL don't know if Bug #29 fix works because:
- Kernel wblw: Ran commit 71783a1 (WITHOUT Bug #29 fix)
- Kernel wncg: Ran commit e004042 (WITH Bug #29 fix) but ONLY 2 TIMESTEPS

## Next Steps

### Priority 1: Identify Script Bug 🚨 HIGH
**Action:** Search for where "2 timesteps" is configured
```bash
grep -r "2 timesteps" validation_ch7/
grep -r "Training: 2" validation_ch7/
grep -r "QUICK_TEST_MODE" validation_ch7/
```

**Files to check:**
1. `validation_ch7/scripts/test_section_7_6_rl_performance.py`
2. `validation_ch7/scripts/test_utils.py`
3. Environment variable handling in Kaggle deployment

### Priority 2: Deploy Proper Quick Test 🔥 CRITICAL
**Options:**

#### Option A: Fix Script + Re-deploy
1. Find and remove the "2 timestep" override
2. Verify script locally trains with 100 timesteps
3. Commit fix as Bug #30
4. Deploy new kernel (e.g., `wncg-v2` or `xyza`)

#### Option B: Use Full Training Mode
1. Deploy kernel WITHOUT `--quick-test` flag
2. Train for full 5000 timesteps (~4 hours)
3. Guaranteed to validate Bug #29 properly

#### Option C: Manual Override
1. Modify script to accept `--min-timesteps` CLI arg
2. Deploy with explicit `--min-timesteps=100`
3. Bypass internal quick test logic

**Recommendation:** **Option A** (Fix script properly) is best for long-term health.

### Priority 3: Literature Comparison (CAN PROCEED IN PARALLEL)
This task is independent of Bug #29/30 validation and can continue:
- Complete `THESIS_CONTRIBUTION_CACHE_AND_SCENARIO.md`
- Add literature comparison table
- Cite Flow, LibSignal, Open RL Benchmark, Cai 2024

## Timeline

### Completed (2025-10-15 16:00-16:40)
- ✅ Bug #29 properly committed (commit e004042)
- ✅ Kernel wncg deployed and completed
- ✅ Results downloaded and analyzed
- ✅ Bug #30 discovered (2-timestep issue)

### Next 30 minutes (16:45-17:15)
1. Search codebase for "2 timesteps" configuration
2. Identify exact location of quick test override
3. Design fix for Bug #30
4. Complete literature comparison table (parallel task)

### Next 2 hours (17:15-19:00)
1. Implement Bug #30 fix
2. Test locally (verify 100 timesteps executed)
3. Commit Bug #30 fix
4. Deploy new kernel (wncg-v2 or similar)
5. Monitor completion
6. Analyze Bug #29 validation results (finally!)

## Lessons Learned

### ❌ What Went Wrong
1. **Insufficient Quick Test Validation:** We didn't verify the "100 timestep" quick test worked before deploying
2. **Silent Override:** Script internally overrode CLI args without clear error message
3. **Result Analysis Assumptions:** Assumed zero rewards meant Bug #29 failure, not training failure

### ✅ What Went Right
1. **Comprehensive Logging:** Debug logs showed exactly what happened (2 timesteps)
2. **Analysis Script:** `analyze_bug29_results.py` caught the anomaly immediately
3. **Git Hygiene:** Bug #29 code properly committed despite validation failure

### 🎓 Process Improvements
1. **Add timestep verification:** Script should print "ACTUALLY TRAINING X TIMESTEPS" after all overrides
2. **Validate quick test locally:** Always test quick mode locally before Kaggle deployment
3. **Assertion checks:** Add assertions that actual training matches intended training

## Related Documents
- `BUG29_DEPLOYMENT_FIX.md` - Previous Git staging issue
- `BUG28_REWARD_FUNCTION_ANALYSIS.md` - Reward sensitivity analysis
- `KAGGLE_GPU_INTEGRATION_SUMMARY.md` - Kaggle deployment procedures

## Status Updates

### [2025-10-15 16:45] - Bug Discovered
- Kernel wncg completed with all zero rewards
- Discovered training only ran 2 timesteps
- Created this investigation document

### [2025-10-15 16:50] - Next: Root Cause Analysis
- About to search codebase for "2 timesteps" configuration
- Will identify exact override location
- Then proceed to fix implementation
