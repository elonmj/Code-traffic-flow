# Bug #29 Investigation and Deployment Fix

## Problem Discovery

**Kernel wblw Results (FAILED)**:
- All rewards: -0.100000 (100% identical)
- All actions: Action 1 (GREEN) - 100% stuck
- Performance: -0.1% vs baseline (no improvement)

**Critical Finding**: These results were IDENTICAL to kernel xrld (before Bug #28 fix), meaning kernel wblw ran the OLD code without Bug #29 fixes!

## Root Cause Analysis

### Investigation Steps:
1. **Analyzed kernel wblw results**: All rewards = -0.1, 100% action 1
   - Expected diverse rewards with Bug #29 fix
   - Got flat rewards matching old code behavior

2. **Checked local code**: Confirmed Bug #29 changes present
   - Line 394: `R_queue = -delta_queue * 50.0` ✅
   - Line 408: `R_stability = -0.01` ✅
   - Lines 411-423: Diversity bonus ✅

3. **Checked Git commit history**:
   ```
   git show 71783a1  # Kernel wblw commit
   ```
   **Result**: Only changed `log.txt`, NOT `traffic_signal_env_direct.py`

4. **Checked Git status**:
   ```
   git status Code_RL/src/env/traffic_signal_env_direct.py
   ```
   **Result**: `Changes not staged for commit`

### Root Cause:
**Bug #29 changes were NEVER COMMITTED to Git!**

The changes existed in the working directory but were not staged. When the validation script auto-committed before deploying, it only committed STAGED changes (which was just log.txt from previous operations).

**Timeline**:
1. We edited `traffic_signal_env_direct.py` with Bug #29 fix
2. We did NOT run `git add` to stage the changes
3. Validation script ran `git commit` (only commits staged files)
4. Kaggle cloned GitHub repo with commit 71783a1 (no Bug #29 fix)
5. Kernel ran OLD code → flat rewards

## Solution

### Fix Applied:
```bash
# Stage Bug #29 changes
git add Code_RL/src/env/traffic_signal_env_direct.py

# Commit with descriptive message
git commit -m "Fix Bug #29: Amplify reward signal, reduce phase penalty, add diversity bonus"
# Result: commit e004042

# Push to GitHub
git push
# Result: origin/main updated to e004042
```

### Verification:
```bash
git show e004042 --stat
```
**Result**: Confirms `Code_RL/src/env/traffic_signal_env_direct.py` changed

### Re-deployment:
```bash
python run_kaggle_validation_section_7_6.py --quick --scenario traffic_light_control
```
**New kernel**: `joselonm/arz-validation-76rlperformance-wncg`
**Commit**: e004042 (includes Bug #29 fix)
**Status**: Uploading... (expected completion ~5-6 minutes)

## Bug #29 Fix Details

**Changes in traffic_signal_env_direct.py (lines 372-420)**:

### 1. Amplify Queue Signal (Line 394)
```python
# BEFORE:
R_queue = -delta_queue * 10.0

# AFTER:
R_queue = -delta_queue * 50.0  # 5x amplification
```
**Reasoning**: Queue changes were too small (δq ≈ 0.02 vehicles) to create meaningful rewards with 10.0 multiplier. With 50.0, even tiny changes become significant.

### 2. Reduce Phase Penalty (Line 408)
```python
# BEFORE:
R_stability = -0.1 if phase_changed else 0.0

# AFTER:
R_stability = -0.01 if phase_changed else 0.0  # 10x reduction
```
**Reasoning**: With constant queues (R_queue ≈ 0), penalty dominated:
- Change phase: -0.1 (ALWAYS BAD)
- Stay same: 0.0 (ALWAYS BETTER)
Agent learned "never change" → stuck at one action

### 3. Add Diversity Bonus (Lines 411-423)
```python
# NEW: Track last 10 actions
if not hasattr(self, 'action_history'):
    self.action_history = []
self.action_history.append(self.current_phase)
if len(self.action_history) > 10:
    self.action_history.pop(0)

# Bonus for using both actions in last 5 steps
if len(self.action_history) >= 5:
    recent_actions = self.action_history[-5:]
    action_diversity = len(set(recent_actions))
    R_diversity = 0.02 if action_diversity > 1 else 0.0
else:
    R_diversity = 0.0
```
**Reasoning**: Encourages exploration without dominating performance signal. Small bonus (0.02) prevents agent from getting stuck at first strategy.

### 4. Total Reward (Line 420)
```python
reward = R_queue + R_stability + R_diversity
```

## Expected Results from Kernel wncg

### Success Criteria:
1. **Reward Diversity**: >3 unique reward values (not flat -0.1)
2. **Action Diversity**: 20-80% split between RED/GREEN (not 100% stuck)
3. **Learning Trend**: Later rewards ≥ earlier rewards (showing improvement)

### Test Plan:
1. Monitor kernel completion (~5 minutes)
2. Download results: `kaggle kernels output joselonm/arz-validation-76rlperformance-wncg`
3. Run analysis: `python analyze_bug29_results.py validation_output/results/...`
4. Check three metrics:
   - Reward std > 0.05
   - Action 0% in range [20%, 80%]
   - Second half mean reward > first half mean reward

### Decision Tree:
- **If SUCCESS**: Proceed to full 5000 timestep training
- **If PARTIAL**: Tune hyperparameters (decision_interval, multipliers)
- **If FAIL**: Deep investigation (queue calculation, observation space)

## Lessons Learned

### Critical Process Issues:
1. **Always verify Git staging**: Check `git status` before deploying
2. **Validate commits**: Use `git show --stat` to confirm changes included
3. **Test locally first**: Run quick validation before Kaggle deployment
4. **Monitor early signals**: If results match old behavior, suspect deployment issue

### Improved Workflow:
```bash
# 1. Make code changes
vim Code_RL/src/env/traffic_signal_env_direct.py

# 2. STAGE changes immediately
git add Code_RL/src/env/traffic_signal_env_direct.py

# 3. Verify staging
git status  # Confirm "Changes to be committed"

# 4. Commit with clear message
git commit -m "Fix Bug #29: ..."

# 5. Verify commit content
git show --stat HEAD

# 6. Push to remote
git push

# 7. Deploy to Kaggle
python run_kaggle_validation_section_7_6.py --quick --scenario traffic_light_control
```

### Prevention:
- Add git status check to validation script
- Warn if unstaged changes exist in critical files
- Show commit hash in kernel logs for traceability

## Status

**Current State**: Kernel wncg executing on Kaggle with CORRECT code (commit e004042)

**Expected Completion**: ~16:39 (5-6 minutes from 16:34 deployment)

**Next Steps**:
1. ⏳ Wait for kernel completion
2. 📊 Download and analyze results
3. ✅/❌ Validate Bug #29 fix effectiveness
4. 📋 Decision: Full training vs tuning vs investigation

## Reference

- **Kernel wblw (FAILED)**: joselonm/arz-validation-76rlperformance-wblw
  - Commit: 71783a1 (no Bug #29 fix)
  - Results: All rewards -0.1, 100% action 1
  
- **Kernel wncg (TESTING)**: joselonm/arz-validation-76rlperformance-wncg
  - Commit: e004042 (WITH Bug #29 fix)
  - URL: https://www.kaggle.com/code/joselonm/arz-validation-76rlperformance-wncg
  - Status: Running...
