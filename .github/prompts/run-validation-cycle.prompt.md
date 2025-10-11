---
mode: 'agent'
tools: ['codebase', 'terminalCommand']
description: 'Executes one full ARZ-RL validation cycle: launch, download logs, deep analysis, diagnose Bug #N, implement fix, commit, push, and relaunch. Continues until validation_success: true.'
---

# Execute One ARZ-RL Validation Cycle

Your goal is to execute **one complete iteration** of our proven development workflow. The ultimate objective is to achieve `validation_success: true` with non-zero performance improvements (avg_flow_improvement > 0.0).

## Critical Context

**This is a proven system:**
- 75.1% success rate over 369 cycles
- Average 1.8 iterations before success
- Based on 14,114 lines of real development history

**Your responsibility:** Execute SOP-1 (Standard Operating Procedure) without deviation.

---

## Starting Point

The last Kaggle kernel execution has finished. You have access to logs.

**Input methods:**
1. User provides `#file:path/to/log` → Start analysis immediately
2. No file provided → Find latest in `validation_output/results/`

**Quick identification:**
```bash
# Find latest kernel results
ls -lt validation_output/results/ | head -n 1
```

---

## Execution Steps (SOP-1: Core Validation Cycle)

### Step 1: Launch & Wait (if this is first cycle)
```bash
python validation_ch7/scripts/run_kaggle_validation_section_7_6.py --quick-test
```

**CRITICAL RULES:**
- ✅ Wait for COMPLETE execution
- ✅ Monitor until terminal prompt returns
- ✅ DO NOT interrupt the process
- ✅ Kernel runs for ~15 minutes (quick test) or ~3-4 hours (full test)

**Why this matters:** Interrupting = incomplete logs = impossible to debug

---

### Step 2: Secure the Log (TOP PRIORITY)

**Your FIRST priority is successfully downloading the kernel log.**

**Primary log:** `arz-validation-76rlperformance-XXXX.log`
**Secondary log:** `section_7_6_rl_performance/debug.log`

**If download fails:**
- This becomes **Bug #0** (highest priority)
- Common cause: `UnicodeEncodeError` from Unicode chars (`→`, `≈`, `🚀`)
- **Fix immediately:** Update `validation_kaggle_manager.py` encoding
- Test encoding fix locally before pushing

**Pattern to use:**
```
grep_search("download.*log") → 
read_file(validation_kaggle_manager.py) → 
replace_string(fix encoding) → 
run_terminal(test locally) → 
commit + push
```

**Non-negotiable:** Cannot proceed without logs.

---

### Step 3: Analyze High-Level Results

**Check session summary:**
```bash
cat validation_output/results/{kernel_slug}/validation_results/session_summary.json
```

**Decision tree:**
```python
if validation_success == True and avg_flow_improvement > 0.0:
    print("🎉 SUCCESS! Validation complete.")
    # Document success, commit final results
    exit(0)
else:
    print("❌ FAILURE. Proceeding to deep diagnosis.")
    # Go to Step 4
```

**Success markers:**
- `validation_success: true`
- `avg_flow_improvement > 0.0` (e.g., 2.5%, 8.3%)
- Figures generated (PNG files exist)
- LaTeX content created
- No errors in validation_log.txt

---

### Step 4: Deep Diagnosis (The Log is the Fuel)

**This is where you EARN your success. Be systematic.**

#### 4.1 Open Both Logs
```bash
# Primary log (full kernel output)
cat arz-validation-76rlperformance-XXXX.log

# Secondary log (debug output)
cat section_7_6_rl_performance/debug.log
```

#### 4.2 Search for Critical Patterns

**Pattern #1: Domain Drainage Check**
```bash
grep "Mean densities:" arz-*.log
```
**What to look for:**
- Values approaching 0.0 → Domain draining to vacuum (BUG!)
- Values stable at reasonable levels (0.1-0.4) → OK

**Pattern #2: Boundary Condition Verification**
```bash
grep "\[BC UPDATE\]" arz-*.log
```
**What to verify:**
- Timing: Updates happening at correct simulation time?
- Values: Inflow/outflow values reasonable?
- Frequency: Updates at expected intervals?

**Pattern #3: Controller Divergence**
```bash
grep "State hash:" arz-*.log
```
**What to compare:**
- BaselineController state hash at step N
- RLController state hash at step N
- Should be IDENTICAL initially, then diverge due to different actions

**Pattern #4: Action Sequences**
```bash
grep "Action:" arz-*.log | head -n 50
```
**What to check:**
- BaselineController: Consistent, predictable actions?
- RLController: Learning (actions change over time)?
- Both: Actions within valid range?

#### 4.3 Compare Baseline vs RL

**Critical analysis:**
```
1. Do both controllers start with identical initial conditions?
   → Search for: "Initial state:" in logs
   
2. Do they diverge ONLY due to actions (not due to bugs)?
   → Compare state progression step-by-step
   
3. Are metrics calculated from same simulation runs?
   → Verify no desynchronization in logging
```

#### 4.4 Identify Root Cause

**Root cause is NOT:**
- ❌ "Metrics are zero" (symptom)
- ❌ "RL doesn't work" (vague)
- ❌ "Need more training" (maybe, but why?)

**Root cause IS:**
- ✅ "Baseline controller state not logged, causing comparison failure"
- ✅ "Boundary conditions update after RL action, creating desynchronization"
- ✅ "Inflow BC uses wrong momentum equation (missing ρv² term)"

**Use the 5 Whys:**
```
Why are metrics 0.0%?
→ Because flow improvement calculation returns 0

Why does calculation return 0?
→ Because baseline_flow == rl_flow

Why are they equal?
→ Because both controllers see identical states

Why do they see identical states?
→ Because state hash logging is missing for BaselineController

ROOT CAUSE: State logging not implemented for BaselineController
```

---

### Step 5: Document and Fix the Bug

#### 5.1 Assign Bug Number

**Next sequential bug:** Bug #N (e.g., Bug #12, Bug #13, Bug #14...)

**Naming convention:**
```
Bug #12: Baseline controller state not logged
Bug #13: Inflow BC momentum equation missing ρv² term
Bug #14: Log download fails on Unicode characters
```

#### 5.2 Create Bug Documentation

**File:** `docs/BUG_FIX_<DESCRIPTIVE_NAME>.md`

**Template:**
```markdown
# BUG #N: [One-line summary]

## Symptom
What was observed in the logs?
- Example: "All metrics show 0.0% improvement"
- Example: "Domain drains to vacuum after 10 steps"

## Evidence
Paste key log snippets with line numbers:

```
Line 423: Mean densities: [0.0, 0.0, 0.0, 0.0]
Line 445: [BC UPDATE] t=15.2, inflow=0.0 (WRONG - should be >0)
Line 502: State hash BaselineController: NOT FOUND
```

Explain how evidence points to root cause:
- Densities=0.0 indicates domain drainage
- BC inflow=0.0 when it should be >0 → BC bug
- Missing state hash → logging not implemented

## Root Cause
Explain the FUNDAMENTAL problem:
- Not surface symptom
- Not "doesn't work"
- Specific code or configuration issue

Example: "BaselineController.step() does not call self.log_state(), 
so state hash is never logged, making comparison impossible."

## Solution
Describe the fix:
- What code changed
- Why this fixes the root cause
- How to verify the fix

**Before:**
```python
def step(self, action):
    new_state = self.simulator.step(action)
    return new_state
```

**After:**
```python
def step(self, action):
    new_state = self.simulator.step(action)
    self.log_state(new_state)  # FIX: Log state for comparison
    return new_state
```

**Justification:** Now both controllers log states, enabling valid comparison.

## Verification
How to confirm fix works:
- [ ] Quick test passes
- [ ] Logs show state hashes for both controllers
- [ ] Metrics are non-zero
- [ ] No regressions in other tests
```

#### 5.3 Implement the Fix

**Use proven Pattern #2 (read_file → replace_string → run_terminal):**

```
1. Read relevant files (3+ for context)
   read_file(main_file)
   read_file(dependent_file1)
   read_file(dependent_file2)

2. Implement fix with replace_string_in_file
   - Include 3-5 lines context before/after
   - Verify exact match of oldString
   - Ensure newString is complete and correct

3. Test locally if possible
   run_terminal("pytest tests/test_specific.py")
   
4. If no local test, prepare for Kaggle validation
```

**Quality checklist:**
- [ ] Fix addresses root cause (not symptom)
- [ ] Code follows project style
- [ ] No unintended side effects
- [ ] Comments explain WHY (not just WHAT)
- [ ] Error handling added if needed

---

### Step 6: Commit and Push

**Create detailed, multi-line commit message:**

```bash
git add <files>
git commit -m "Fix Bug #N: <one-line summary>

<Detailed explanation>
- Root cause: <fundamental problem>
- Solution: <what was changed>
- Evidence: <log lines that showed the bug>
- Verification: <how to confirm fix>

Closes #<issue_number> (if applicable)
Ref: docs/BUG_FIX_<NAME>.md"

git push origin main
```

**Example commit message:**
```
Fix Bug #12: Baseline controller state not logged

Root cause: BaselineController.step() did not call self.log_state(),
making it impossible to compare state hashes between Baseline and RL.

Solution: Added self.log_state(new_state) after simulator step in
BaselineController.step() method.

Evidence: Log line 502 showed "State hash BaselineController: NOT FOUND"
while RLController logged state hash correctly.

Verification: After fix, both controllers log state hashes, enabling
valid comparison and non-zero metrics.

Ref: docs/BUG_FIX_BASELINE_CONTROLLER_STATE_LOGGING.md
```

**CRITICAL: Verify push succeeds**
```bash
git push origin main
# Wait for confirmation
# Kaggle kernel will clone this version
```

**If push fails:**
- Check network
- Verify credentials  
- Resolve merge conflicts
- DO NOT launch Kaggle until push succeeds

---

### Step 7: Relaunch and Monitor

**Launch new quick test:**
```bash
python validation_ch7/scripts/run_kaggle_validation_section_7_6.py --quick-test
```

**Monitor progress:**
- Watch terminal output
- Note kernel slug
- Confirm upload success
- Wait for completion (~15 minutes)

**Kernel URL:**
```
https://www.kaggle.com/code/{username}/{kernel_slug}
```

**While waiting:**
- Review bug documentation
- Plan next steps if this fails
- Update project notes

---

### Step 8: Report Results

**Once kernel completes, provide structured summary:**

```markdown
## Cycle N Results

**Bug Addressed:** Bug #N - <summary>

**Fix Applied:** <one-line description>

**Outcome:**
- Validation success: [true/false]
- Avg flow improvement: [X.X%]
- Key metrics: [list]

**Log Analysis:**
- [Critical log findings]
- [Comparison to previous cycle]

**Next Actions:**
- If success: Document final results, commit, celebrate 🎉
- If failure: Diagnose Bug #N+1, repeat cycle

**Confidence Level:** [High/Medium/Low]
- High (>90%): All indicators positive
- Medium (70-90%): Some issues but progress made  
- Low (<70%): Major problems persist, new strategy needed
```

**If still failing after 3 cycles:**
- Stop and reflect
- Review all bug docs
- Consider alternative approach
- May need to revisit earlier assumptions

---

## Tool Sequences for This Workflow

### Analysis Phase
**Pattern:** `grep_search` → `read_file` × 3
```
grep_search("Mean densities")  # Find issue
read_file(main_log)            # Context
read_file(simulator_code)      # Source
read_file(boundary_conditions) # Related
```

### Fix Implementation  
**Pattern:** `read_file` × 3 → `replace_string` → `run_terminal`
```
read_file(file_to_fix)        # Target
read_file(related_file1)      # Dependencies
read_file(related_file2)      # Usage patterns
replace_string(implement_fix) # Apply fix
run_terminal(test)            # Validate
```

### Deployment
**Pattern:** `run_terminal` × 3
```
run_terminal("git add .")
run_terminal("git commit -m '...'")
run_terminal("git push origin main")
```

---

## Success Indicators

### High Confidence (>90% success probability)
- ✅ Logs show expected patterns
- ✅ State hashes present for both controllers
- ✅ Boundary conditions update correctly
- ✅ Domain densities remain stable
- ✅ Actions are within valid ranges
- ✅ Metrics calculation succeeds

### Medium Confidence (70-90%)
- ⚠️ Some unexpected outputs but minor
- ⚠️ Most tests pass, few failures
- ⚠️ Metrics non-zero but lower than expected
- ⚠️ Need additional validation round

### Low Confidence (<70%)
- ❌ Major log discrepancies
- ❌ Zero metrics persist
- ❌ Domain drainage continues
- ❌ Controllers show identical behavior
- 🔄 **Action:** Return to Step 4 (Deep Diagnosis)

---

## Failure Recovery

### If Cycle Fails (Normal - 25% failure rate)

**Don't panic:**
- 92 out of 369 cycles failed in training data
- Failure provides crucial learning data
- Average 1.8 iterations to success

**Extract learning:**
1. What new information does log reveal?
2. Was hypothesis incorrect?
3. What assumptions were wrong?

**Update strategy:**
- Refine root cause understanding
- Try different tool sequence
- Read additional context files
- Consider alternative explanations

### If >3 Cycles Fail (Red Flag)

**Stop and reassess:**
1. Review all bug documentation
2. Check if fixing symptoms vs. root cause
3. Consider fundamentally different approach
4. May need to revisit architecture

**Escalation:**
- Document persistent issue
- Create detailed analysis
- Request user guidance on strategy

---

## Communication Style

**Be direct and evidence-based:**
✅ "Log line 423 shows domain drainage: `Mean density: 0.0`. This indicates boundary condition bug."
❌ "I think there might be a problem."

**Document decisions:**
✅ "Using Pattern #2 (read → replace → test) because bug location is known from logs."
❌ "Let me try fixing this."

**Announce bugs clearly:**
✅ "🎯 **BUG #12 DISCOVERED!** Baseline controller state not logged in step() method."
❌ "Found an issue."

**Report progress:**
- State current step (1-8)
- Show evidence for decisions
- Estimate confidence level
- List next 2-3 actions

---

## Remember

1. **Log-driven:** Evidence before action
2. **Systematic:** Follow steps 1-8 in order
3. **Patient:** Wait for kernel completion
4. **Persistent:** 1.8 iterations average
5. **Thorough:** Deep diagnosis is key
6. **Documented:** Bug #N format
7. **Tested:** Quick test before full
8. **Verified:** Push before launch

**Your mission:** Execute one complete cycle. If it fails, the log provides data for the next cycle. If it succeeds, announce victory! 🎉

---

**Reference:** This prompt implements SOP-1 (Standard Operating Procedure) from `arz-rl-validator.instructions.md`, enhanced with proven patterns from 369 analyzed development cycles (75.1% success rate, DEVELOPMENT_CYCLE.md).
