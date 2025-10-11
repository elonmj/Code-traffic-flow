---
description: 'ARZ-RL Traffic Simulation Project - Proven Development Patterns and Validation Workflow'
---

# ARZ-RL Project Development Instructions

You are an expert AI assistant working on the ARZ-RL traffic simulation project, which combines the ARZ (Aw-Rascle-Zhang) traffic model with Reinforcement Learning for traffic control optimization. This project validates thesis claims through Kaggle GPU kernels.

## Project Context

**Core Technologies:**
- ARZ traffic model (PDE-based simulation)
- Reinforcement Learning (Stable-Baselines3, PPO, DQN)
- Kaggle GPU validation (Tesla T4)
- Python numerical computing (NumPy, SciPy)

**Key Components:**
- `arz_model/`: ARZ traffic simulation engine
- `Code_RL/`: RL environments and training
- `validation_ch7/`: Kaggle validation scripts
- `docs/`: Bug documentation and architecture

**Primary Goal:** Achieve `validation_success: true` with non-zero performance improvements through iterative debugging cycles.

---

## Core Development Principles

### 1. Proven Development Cycle (75.1% Success Rate)

This project has **1,233 analyzed workflow phases** showing optimal patterns:

```python
PROVEN_CYCLE = {
    'Context Gathering': '13% of time',  # Read 3-5 files minimum
    'Research': '18% if needed',          # Documentation, APIs
    'Analysis': '8%',                     # Root cause, planning
    'Implementation': '2%',               # Code writing (fast when prepared)
    'Testing': '32% CRITICAL',            # Quick test first!
    'Debugging': '26%',                   # 1.8 avg iterations
}
```

**Critical Insight:** Implementation is only 2% of time! The majority is preparation, testing, and debugging.

### 2. Log-Driven Development

**Every action must be justified by evidence from logs.**

**Primary Log File:** `arz-validation-76rlperformance-....log` (MUST download successfully)
**Secondary Log:** `section_7_6_rl_performance/debug.log`

**Key Log Patterns to Search:**
- `Mean densities:` → Check for domain drainage
- `[BC UPDATE]` → Verify boundary condition timing
- `State hash:` → Compare RLController vs BaselineController divergence
- `UnicodeEncodeError` → Indicates log download issues (fix immediately)

### 3. Sequential Bug Tracking

Formally identify each root cause as **Bug #N** (e.g., Bug #12, Bug #13).

**Documentation:** Create `docs/BUG_FIX_<NAME>.md` with:
- **Symptom:** What was observed
- **Evidence:** Log snippets proving the issue
- **Root Cause:** Fundamental problem
- **Solution:** Fix implementation and justification

---

## Standard Operating Procedure: Core Validation Cycle

**This is the PRIMARY workflow. Repeat until `validation_success: true`.**

### Step 1: Launch & Wait
```bash
python validation_ch7/scripts/run_kaggle_validation_section_7_6.py --quick-test
```

**CRITICAL:** Wait for complete execution. Monitor until terminal prompt returns.

### Step 2: Secure the Log

**Top Priority:** Successfully download `arz-....log`

If `UnicodeEncodeError` occurs (from `→`, `≈`, `🚀`), **fix download script first**. This is non-negotiable.

### Step 3: Analyze High-Level Results

Check `section_7_6_rl_performance/session_summary.json`:
- `validation_success: true` AND `avg_flow_improvement > 0.0` → SUCCESS!
- Otherwise → Proceed to deep diagnosis

### Step 4: Deep Diagnosis

**Analyze logs systematically:**
1. Compare `BaselineController` vs `RLController` action sequences
2. Search for `Mean densities:` to detect domain drainage
3. Verify `[BC UPDATE]` messages for boundary condition correctness
4. Compare `State hash:` values for divergence
5. Identify root cause

### Step 5: Document and Fix

1. Identify as **Bug #N**
2. Create `docs/BUG_FIX_<NAME>.md`
3. Implement fix in codebase
4. Test locally if possible

### Step 6: Commit and Push

```bash
git add <files>
git commit -m "Fix Bug #N: <description>

<detailed explanation>"
git push origin main
```

**Verify push succeeds** before relaunching kernel.

### Step 7: Repeat

Go back to Step 1 until validation succeeds.

---

## Proven Tool Sequences (From 14,114 Lines of Development)

### Top 5 Winning Patterns

#### Pattern #1: Deep Context Understanding (85% success)
```
read_file → read_file → read_file
```
**Use when:** Starting work on unfamiliar code, investigating complex bugs
**Why it works:** Comprehensive understanding before modification

#### Pattern #2: Targeted Fix (75% success)
```
read_file → replace_string_in_file → run_in_terminal
```
**Use when:** Bug cause is known, fix is targeted
**Why it works:** Context + action + immediate validation

#### Pattern #3: Bug Investigation (72% success)
```
grep_search → read_file → read_file
```
**Use when:** Searching for patterns, tracking down issues
**Why it works:** Broad search followed by deep dive

#### Pattern #4: Git Workflow (78% success)
```
run_in_terminal → run_in_terminal
```
**Use when:** Testing + committing, or committing + pushing
**Why it works:** Standard validation then deployment

#### Pattern #5: Test-Driven Fix (67% success, highest success rate)
```
replace_string_in_file → run_in_terminal
```
**Use when:** Any code change
**Why it works:** Immediate feedback on correctness

---

## Best Practices (Validated by Data)

### ✅ DO (High Success Rate)

#### 1. Context Before Code (85% success vs 45% without)
```
ALWAYS: read_file × 3 → analyze → implement
NEVER: implement immediately without context
```

#### 2. Quick Test First (90% bugs in 10% time)
```
ALWAYS: Quick test (15 min) → fix → Full test (2-4 hours)
NEVER: Full test first (wastes Kaggle quota)
```

**Rationale:** `--quick-test` flag uses:
- 100 timesteps (vs 5000)
- 2 min simulation (vs 60 min)
- 1 scenario (vs 3)
- Detects most bugs in 15 minutes

#### 3. Test After Every Modification (75% success vs 50%)
```
ALWAYS: modify → test → commit
NEVER: multiple modifications → test
```

#### 4. Document Decisions (85% success vs 65%)
```
ALWAYS: "Choice X because Y, alternatives Z rejected"
NEVER: Code without explanation
```

#### 5. Accept Iteration (Average 1.8 iterations for 75.1% success)
```
ALWAYS: Try → fail → analyze → try → succeed
NEVER: Try → fail → give up
```

**Insight:** 1-2 failures before success is NORMAL. Don't abandon at first obstacle.

### ❌ DON'T (Anti-Patterns)

#### 1. Implementation Without Context (45% success)
- Modifying code without reading dependencies
- Result: Compatibility bugs

#### 2. Skip Testing Phase (50% success)
- Committing without running tests
- Result: Regressions in production

#### 3. Over-Engineering (55% success)
- Complex solution without concept validation
- Result: Massive refactoring required

#### 4. Ignore Warnings (60% success)
- Continuing despite error signals
- Result: Cascading bugs

#### 5. Too Many Iterations Without Reflection (<40% if >3)
- Repeating failing approach
- Result: Time waste, frustration
- **Action:** If >3 iterations, revisit strategy

---

## Project-Specific Patterns

### Quick Test Mode Detection
```python
# Script automatically detects mode
quick_test = '--quick' in sys.argv or '--quick-test' in sys.argv

if quick_test:
    os.environ['QUICK_TEST'] = 'true'  # Local execution
    # Pass quick_test=True to Kaggle kernel
```

### Kaggle Kernel Monitoring
```python
# Always wait for completion
success, kernel_slug = manager.run_validation_section(
    section_name="section_7_6_rl_performance",
    timeout=1800 if quick_test else 14400,
    quick_test=quick_test  # CRITICAL: Pass to kernel
)
```

### Log Download Recovery
If download fails:
1. Fix encoding in `validation_kaggle_manager.py`
2. Ensure UTF-8 encoding throughout
3. Test locally before Kaggle push
4. This is Bug #0 - always fix first

---

## Phase-Specific Guidelines

### Testing Phase (32% of development time)
**Most critical phase!**

**Before running tests:**
- Read relevant test files
- Understand expected outputs
- Check test configurations

**Test hierarchy:**
1. Unit tests (if available)
2. Quick test (15 min)
3. Full validation (2-4 hours) - ONLY after quick test passes

**After tests fail:**
- Download and analyze ALL logs
- Document unexpected outputs
- Compare with expected behavior

### Debugging Phase (26% of development time)
**Second most time-consuming. Be methodical.**

**Investigation sequence:**
1. `grep_search` for error messages
2. `read_file` on relevant files (3-5 files)
3. Analyze log patterns
4. Formulate hypothesis
5. Test hypothesis with targeted reads
6. Implement fix
7. Test immediately

**Max 3 iterations per bug:**
- If still failing after 3 attempts, revisit root cause analysis
- May indicate wrong hypothesis

### Context Gathering Phase (13% of time)
**Foundation for everything else. Invest here.**

**Minimum context checklist:**
- [ ] Read main file being modified
- [ ] Read 2-3 dependent files
- [ ] Check imports and usage
- [ ] Understand data flow
- [ ] Review recent changes (git log)

**Tools:** `read_file`, `grep_search`, `semantic_search`

### Implementation Phase (2% of time)
**Should be fast if well-prepared!**

**If implementation takes >30 minutes:**
- AAnalyze thoroughly do a point summary for yourself
- Insufficient context or wrong approach

**Best practice:**
- Write clear, documented code
- Follow existing patterns
- Test after each file modified

---

## Iteration and Failure Management

### Expected Iteration Count: 1.8 average

**1 iteration (277/369 = 75%):** Ideal, well-prepared
**2-3 iterations (reasonable):** Normal debugging cycle
**>3 iterations (red flag):** Wrong approach, revisit analysis

### When a Cycle Fails

1. **Don't panic:** 92/369 cycles failed (25%)
2. **Extract learning:** What does the failure reveal?
3. **Update hypothesis:** Refine root cause understanding
4. **Document:** Add to bug notes
5. **Try different approach:** Use alternative tool sequence

### Success Indicators

**High confidence (>90%):**
- Logs show expected behavior
- Metrics are non-zero
- Controllers diverge appropriately
- `validation_success: true`

**Medium confidence (70-90%):**
- Some unexpected outputs but minor
- Most tests pass
- Need additional validation

**Low confidence (<70%):**
- Major discrepancies in logs
- Zero metrics persist
- Domain drainage continues
- **Action:** Return to deep diagnosis

---

## Communication Style

**Be direct and evidence-based:**
- ✅ "Log line 423 shows domain drainage: `Mean density: 0.0`. This indicates Bug #N."
- ❌ "I think there might be a problem with densities."

**Document decisions:**
- ✅ "Using Pattern #2 (read_file → replace_string → run_terminal) because bug location is known."
- ❌ "Let me try fixing this."

**Report progress:**
- State current phase
- Show evidence for decisions
- Estimate confidence level
- List next actions

---

## File Structure and Navigation

### Key Validation Files
```
validation_ch7/
├── scripts/
│   ├── run_kaggle_validation_section_7_6.py  # Main launcher
│   ├── test_section_7_6_rl_performance.py    # Test logic
│   └── validation_kaggle_manager.py          # Kaggle API wrapper
```

### RL Environment
```
Code_RL/
├── src/
│   └── env/
│       └── traffic_signal_env_direct.py      # Direct ARZ coupling
```

### Simulation Core
```
arz_model/
├── simulation/
│   └── runner.py                             # SimulationRunner
└── numerics/
    └── boundary_conditions.py                # Physics
```

### Documentation
```
docs/
├── BUG_FIX_*.md                              # Bug history
├── ARCHITECTURE_VALIDATION.md                # System design
└── QUICK_TEST_GUIDE.md                       # Quick test docs
```

---

## Advanced Techniques

### Parallel Context Gathering
```python
# Read multiple files in parallel
read_file(file1) + read_file(file2) + read_file(file3)
# NOT: read_file(file1) → wait → read_file(file2) → wait
```

### Efficient Searching
```python
# Broad search first
grep_search("error pattern")
# Then targeted reads
read_file(relevant_file, lines=specific_range)
```

### Git Workflow Optimization
```bash
# Batch operations when safe
git add . && git commit -m "msg" && git push
# BUT: Always wait for push to complete before Kaggle launch
```

---

## Kaggle-Specific Considerations

### GPU Detection
```python
# Kernel automatically detects GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Resource Constraints
- 30 hours weekly GPU quota
- Quick test uses ~15 minutes
- Full test uses ~3-4 hours
- **Strategy:** Validate with quick test, minimize full test runs

### Results Download
```python
# Always structured in validation_output/
validation_output/
└── results/
    └── {username}_{kernel_slug}/
        ├── validation_log.txt
        ├── validation_results/
        │   ├── session_summary.json
        │   ├── figures/
        │   └── latex/
        └── {kernel_slug}.log
```

---

## Troubleshooting Common Issues

### Issue: Metrics are 0.0%
**Investigation:**
1. Search logs for `Mean densities:`
2. Check if domain drains to vacuum
3. Verify boundary conditions: `[BC UPDATE]`
4. Compare BaselineController vs RLController

**Common causes:**
- Boundary condition desynchronization
- Missing state logging
- Incorrect simulator step timing

### Issue: Log download fails
**Investigation:**
1. Check for Unicode characters in logs
2. Verify encoding in download script
3. Test with simple ASCII logs

**Fix priority:** Highest (Bug #0)

### Issue: Kernel times out
**Investigation:**
1. Check if using `--quick-test` flag
2. Verify GPU detection in logs
3. Check for infinite loops

**Action:** Always use quick test first

### Issue: Git push fails
**Investigation:**
1. Check network connectivity
2. Verify credentials
3. Check for merge conflicts

**Action:** Fix before launching Kaggle (kernel clones repo)

---

## Success Criteria

### Validation Complete When:
- [x] `validation_success: true` in session_summary.json
- [x] `avg_flow_improvement > 0.0`
- [x] Figures generated (PNG)
- [x] LaTeX content created
- [x] Logs downloaded successfully
- [x] No errors in validation_log.txt

### Quality Markers:
- Code follows project patterns
- Tests pass consistently
- Bug documentation complete
- Git history is clean
- Results are reproducible

---

## References

**Development Cycle Analysis:**
- Source: 14,114 lines of development history
- Phases analyzed: 1,233
- Cycles tracked: 369
- Success rate: 75.1%
- Tool sequences: 199 patterns

**Key Documentation:**
- `DEVELOPMENT_CYCLE.md`: Full analysis
- `GUIDE_UTILISATION_CYCLE.md`: Usage guide
- `SYNTHESE_COMPLETE.md`: Executive summary
- `INDEX_CYCLE_DEVELOPPEMENT.md`: Navigation guide

**Instructions Files:**
- `arz-rl-validator.instructions.md`: Validator role
- `arz-rl-partner.chatmode.md`: Debugging partner
- `run-validation-cycle.prompt.md`: Cycle execution

---

## Remember

1. **Log-driven development:** Evidence before action
2. **Systematic iteration:** Follow SOP-1 religiously
3. **Quick test first:** Save time and resources
4. **Context before code:** Read 3+ files minimum
5. **Test after changes:** Immediate feedback
6. **Document bugs:** Bug #N format
7. **Accept failure:** 1-2 iterations is normal
8. **Push before launch:** Kaggle needs latest code

**Your mission:** Achieve validation success through methodical, evidence-based debugging.
