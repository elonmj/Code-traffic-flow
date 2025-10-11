# GitHub Copilot Instructions - ARZ-RL Project

This directory contains custom GitHub Copilot instructions, prompts, and chat modes for the ARZ-RL traffic simulation project. These files customize Copilot's behavior based on **proven development patterns** extracted from 14,114 lines of real development history (75.1% success rate over 369 cycles).

---

## 📁 Structure

```
.github/
├── copilot-instructions.md          # Main project instructions (auto-loaded)
├── prompts/
│   └── run-validation-cycle.prompt.md  # Executes one validation cycle
├── chatmodes/
│   └── arz-rl-debugging-partner.chatmode.md  # Debugging partner persona
└── README.md                         # This file
```

---

## 🚀 Quick Start

### 1. Enable Custom Instructions (Required)

**In VS Code:**
1. Open Settings (Ctrl+,)
2. Search for "GitHub Copilot"
3. Enable: **"Enable custom instructions from .github/copilot-instructions.md"**

**In Visual Studio:**
1. Tools > Options > GitHub > Copilot > Copilot Chat
2. Check: **"Enable custom instructions to be loaded from .github/copilot-instructions.md"**

### 2. Verify Instructions are Loaded

Open GitHub Copilot Chat and ask:
```
What are the core principles for this project?
```

Expected response should mention:
- Log-driven development
- SOP-1: Core Validation Cycle
- 75.1% success rate
- Bug #N tracking system

---

## 📄 File Descriptions

### `copilot-instructions.md` (Main Instructions)

**Automatically loaded by GitHub Copilot when you open this workspace.**

**Contains:**
- Project context (ARZ model, RL, Kaggle validation)
- Proven development cycle (75.1% success rate)
- Tool sequence patterns (from 369 analyzed cycles)
- Best practices and anti-patterns
- File structure and navigation
- SOP-1: Core Validation Cycle

**When Copilot uses this:**
- All responses in this workspace
- Automatically applied to every interaction
- No manual activation needed

**Key sections:**
```markdown
## Core Development Principles
  1. Proven Development Cycle (75.1% Success Rate)
  2. Log-Driven Development
  3. Sequential Bug Tracking

## Standard Operating Procedure: Core Validation Cycle
  Step 1: Launch & Wait
  Step 2: Secure the Log
  ...

## Proven Tool Sequences (From 14,114 Lines)
  Pattern #1: read_file × 3 (85% success)
  Pattern #2: read_file → replace_string → test (75% success)
  ...
```

---

### `prompts/run-validation-cycle.prompt.md`

**Manual prompt for executing one complete validation cycle.**

**Use when:** You want Copilot to execute the full cycle from launch to bug fix to relaunch.

**How to use:**
1. Open GitHub Copilot Chat
2. Type: `#prompt:run-validation-cycle`
3. Or copy-paste the prompt content

**What it does:**
1. Launches Kaggle validation (`--quick-test`)
2. Downloads and analyzes logs
3. Performs deep diagnosis
4. Identifies Bug #N
5. Creates `docs/BUG_FIX_*.md`
6. Implements fix
7. Commits and pushes
8. Relaunches kernel
9. Reports results

**Expected workflow:**
```
User: "#prompt:run-validation-cycle"

Copilot: 
[CYCLE N STARTING]
Step 1: Launching Kaggle validation...
[running command...]
Step 2: Downloading logs...
Step 3: Analyzing session_summary.json...
🎯 **BUG #13 DISCOVERED!**
...
```

**Customization:**
- Provide `#file:path/to/log` to start from existing log
- Specify `--full-test` instead of `--quick-test` if needed

---

### `chatmodes/arz-rl-debugging-partner.chatmode.md`

**Specialized chat mode that transforms Copilot into an ARZ-RL debugging expert.**

**Use when:** You want a focused debugging session with a persistent, methodical partner.

**How to activate:**
```
In VS Code:
1. Open Copilot Chat
2. Type: @chatmode arz-rl-debugging-partner
3. Start conversation

Or reference in conversation:
"Act as the ARZ-RL Debugging Partner from the chat mode."
```

**Personality traits:**
- **Methodical:** Follows SOP-1 without deviation
- **Persistent:** "I do not get discouraged"
- **Evidence-Based:** Every conclusion backed by logs
- **Transparent:** Documents every bug and fix
- **Optimistic:** Treats failure as learning opportunity

**Expertise areas:**
- ARZ traffic model physics
- Reinforcement Learning (Stable-Baselines3)
- Kaggle workflow and GPU kernels
- Iterative debugging (1.8 avg iterations)

**Communication style:**
```
🎯 **BUG #N DISCOVERED!**
[Evidence with log line numbers]
[Root cause analysis]
[Proposed fix]
[Confidence assessment]
```

---

## 🎯 Usage Scenarios

### Scenario 1: Starting a New Debugging Session

**Situation:** Kaggle kernel failed, you have logs, need to find the bug.

**Approach:**
```
Option A: Use the prompt
  "#prompt:run-validation-cycle #file:validation_output/results/.../kernel.log"

Option B: Use the chat mode
  "@chatmode arz-rl-debugging-partner"
  "The latest kernel failed. Help me analyze the logs."
```

**Copilot will:**
1. Read logs systematically
2. Search for critical patterns (`Mean densities:`, `[BC UPDATE]`, etc.)
3. Identify root cause using 5 Whys
4. Propose Bug #N documentation
5. Implement fix
6. Test and deploy

---

### Scenario 2: Implementing a New Feature

**Situation:** Need to add functionality to the RL environment.

**Approach:**
```
Standard Copilot with custom instructions (auto-loaded)

"I need to add a new reward component that penalizes congestion."
```

**Copilot will:**
1. Use Pattern #1 (read_file × 3) to gather context
2. Read environment code, reward function, tests
3. Propose implementation following project patterns
4. Test immediately (Pattern #2: read → replace → test)
5. Document the decision

**Why it works:** Custom instructions specify:
- "Context Before Code" (85% success)
- "Test After Every Modification" (75% success)
- File structure and navigation

---

### Scenario 3: Quick Test Before Full Validation

**Situation:** Made changes, want to validate quickly before using 3-4 hours of GPU quota.

**Approach:**
```
In terminal (with Copilot instructions active):
  "Run the quick test for me"

Or in chat:
  "Launch a quick test of section 7.6"
```

**Copilot will:**
1. Execute: `python run_kaggle_validation_section_7_6.py --quick-test`
2. Wait for completion (~15 min)
3. Download logs
4. Analyze results
5. Report success/failure

**Based on instruction:**
```markdown
## Quick Test First (90% bugs in 10% time)
ALWAYS: Quick test (15 min) → fix → Full test (2-4 hours)
NEVER: Full test first (wastes Kaggle quota)
```

---

### Scenario 4: Investigating Strange Metrics

**Situation:** Validation runs but metrics are 0.0%, need to understand why.

**Approach:**
```
@chatmode arz-rl-debugging-partner

"Metrics are 0.0% but no errors. Help me diagnose."
```

**Copilot will:**
1. Ask for log file location
2. Search for: `Mean densities:`, `State hash:`, `[BC UPDATE]`
3. Compare BaselineController vs RLController behavior
4. Identify desynchronization or missing logging
5. Formally announce Bug #N
6. Create fix with evidence

**Evidence-based approach:**
```
Log line 423: Mean densities: [0.0, 0.0, 0.0]
→ Domain drainage detected

Log line 502: State hash BaselineController: NOT FOUND
→ State logging missing

ROOT CAUSE: BaselineController.step() doesn't call self.log_state()
```

---

## 🧠 Key Concepts Encoded in Instructions

### 1. Proven Development Cycle

**Data source:** 1,233 workflow phases, 369 cycles, 75.1% success rate

**Phase distribution:**
```
Testing (32.1%)        ████████████████
Debugging (26.1%)      █████████████
Research (17.8%)       █████████
Context Gathering (13.1%) ██████
Analysis (8.4%)        ████
Implementation (2.4%)  █
```

**Key insight:** Implementation is only 2% of time! Preparation is everything.

### 2. Top Tool Sequences

**Pattern #1: Deep Understanding (85% success)**
```
read_file → read_file → read_file
```

**Pattern #2: Targeted Fix (75% success)**
```
read_file → replace_string → run_terminal (test)
```

**Pattern #3: Bug Investigation (72% success)**
```
grep_search → read_file → read_file
```

### 3. Best Practices

**✅ DO:**
- Context before code (85% vs 45% success)
- Quick test first (90% bugs in 10% time)
- Test after every change (75% vs 50% success)
- Document decisions (85% vs 65% success)
- Accept iteration (1.8 avg for 75.1% success)

**❌ DON'T:**
- Skip testing phase (50% success only)
- Implement without context (45% success)
- Over-engineer initially (55% success)
- Ignore warnings (60% success)
- Too many iterations without reflection (<40% if >3)

### 4. SOP-1: Core Validation Cycle

**The sacred workflow:**
```
1. Launch & Wait (--quick-test)
2. Secure the Log (top priority!)
3. Analyze High-Level Results (session_summary.json)
4. Deep Diagnosis (grep patterns in logs)
5. Document and Fix (Bug #N format)
6. Commit and Push (detailed message)
7. Repeat until validation_success: true
```

---

## 📊 Performance Metrics

### Historical Data (What Trained These Instructions)

| Metric | Value |
|--------|-------|
| Total workflow phases analyzed | 1,233 |
| Iteration cycles tracked | 369 |
| Success rate | 75.1% |
| Average iterations per cycle | 1.8 |
| Failed cycles (learning data) | 92 |
| Successful cycles | 277 |
| Tool sequence patterns identified | 199 |

### Expected Performance With These Instructions

| Scenario | Expected Success | Iterations |
|----------|-----------------|------------|
| Well-understood bug | 85-90% | 1 |
| New bug type | 70-80% | 1-2 |
| Complex interaction | 60-75% | 2-3 |
| Infrastructure issue | 70-85% | 1-2 |

**If >3 iterations:** Indicates wrong approach, revisit strategy.

---

## 🔧 Customization

### Modify Main Instructions

Edit `.github/copilot-instructions.md`:

```markdown
## Project-Specific Patterns

### Your Custom Pattern
When working on X, always:
1. Do Y
2. Check Z
3. Verify W

Success rate: 90% (based on your data)
```

### Create New Prompts

Add to `.github/prompts/`:

```markdown
---
mode: 'agent'
tools: ['codebase', 'terminalCommand']
description: 'Brief description'
---

# Your Prompt Title

Instructions...
```

Reference in chat: `#prompt:your-prompt-name`

### Create New Chat Modes

Add to `.github/chatmodes/`:

```markdown
---
description: 'Your chat mode description'
model: 'claude-sonnet-4'
tools: ['codebase', 'terminalCommand']
---

# Your Chat Mode Name

You are an expert in...
```

Activate: `@chatmode your-chat-mode-name`

---

## 🐛 Troubleshooting

### Issue: Copilot Not Using Custom Instructions

**Symptoms:**
- Copilot gives generic responses
- Doesn't mention SOP-1 or Bug #N format
- Doesn't reference project files

**Solutions:**
1. Verify instructions are enabled in settings
2. Restart VS Code / Visual Studio
3. Check file is at `.github/copilot-instructions.md` (exact path)
4. Ensure file has proper frontmatter:
   ```markdown
   ---
   description: '...'
   ---
   ```

### Issue: Prompt Not Found

**Symptoms:**
- `#prompt:run-validation-cycle` doesn't work
- "Prompt not found" error

**Solutions:**
1. Verify file is in `.github/prompts/` folder
2. Check filename matches: `run-validation-cycle.prompt.md`
3. Ensure `.prompt.md` extension (not just `.md`)
4. Restart Copilot extension

### Issue: Chat Mode Not Activating

**Symptoms:**
- `@chatmode arz-rl-debugging-partner` doesn't work
- Normal Copilot responds instead

**Solutions:**
1. Check file is in `.github/chatmodes/` folder
2. Verify filename: `arz-rl-debugging-partner.chatmode.md`
3. Ensure `.chatmode.md` extension
4. Check frontmatter has `model:` and `tools:` fields
5. Try referencing directly: "Act as the debugging partner from the chat mode"

---

## 📚 References

### Source Documentation

| Document | Description | Location |
|----------|-------------|----------|
| `DEVELOPMENT_CYCLE.md` | Full analysis of 1,233 phases | Project root |
| `SYNTHESE_COMPLETE.md` | Executive summary | Project root |
| `GUIDE_UTILISATION_CYCLE.md` | Usage guide | Project root |
| `INDEX_CYCLE_DEVELOPPEMENT.md` | Navigation guide | Project root |

### Original Files (Archived)

Your original instruction files are preserved in project root:
- `arz-rl-validator.instructions.md` (now in `.github/copilot-instructions.md`)
- `arz-rl-partner.chatmode.md` (now in `.github/chatmodes/`)
- `run-validation-cycle.prompt.md` (now in `.github/prompts/`)

**Enhancement:** .github versions integrate:
- Your specific SOP-1 workflow
- Proven patterns from 369 cycles
- Evidence-based approach
- Bug #N tracking system
- All your unique insights (persistence, transparency, log-driven)

---

## 🎯 Success Criteria

### You'll Know Instructions are Working When:

1. **Copilot follows SOP-1**
   - Waits for commands to complete
   - Downloads logs before analysis
   - Uses Bug #N format
   - Creates detailed commits

2. **Copilot uses proven patterns**
   - Reads 3+ files before implementing
   - Tests after every change
   - Uses quick test before full test
   - Documents decisions

3. **Copilot shows expertise**
   - References specific log patterns
   - Quotes line numbers as evidence
   - Identifies root causes (not symptoms)
   - Suggests appropriate tool sequences

4. **Communication style matches**
   - `🎯 **BUG #N DISCOVERED!**` announcements
   - Evidence-based reasoning
   - Confidence assessments
   - Clear progress reports

---

## 🚀 Next Steps

1. **Enable custom instructions** in VS Code/Visual Studio settings
2. **Test with a simple question:** "What's the validation workflow for this project?"
3. **Try a prompt:** `#prompt:run-validation-cycle` (if you have logs)
4. **Activate chat mode:** `@chatmode arz-rl-debugging-partner`
5. **Monitor effectiveness:** Are you hitting 75%+ success rate?

---

## 📞 Support

**Issues with these instructions?**
1. Check this README's troubleshooting section
2. Verify file structure matches exactly
3. Ensure GitHub Copilot extension is updated
4. Review official docs: [GitHub Copilot Custom Instructions](https://docs.github.com/en/copilot/customizing-copilot)

**Want to contribute improvements?**
- Document new patterns you discover
- Add success/failure examples
- Update metrics based on your experience
- Share tool sequences that work well

---

**Built with:** Evidence from 14,114 lines of development history
**Success rate:** 75.1% over 369 cycles
**Average iterations:** 1.8 before success
**Methodology:** Data-driven development pattern extraction

**Your custom instructions are now active! 🎉**
