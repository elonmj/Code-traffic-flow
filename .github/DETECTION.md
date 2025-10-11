# How GitHub Copilot Recognizes These Instructions

This document explains how GitHub Copilot automatically discovers and applies custom instructions, prompts, and chat modes from the `.github` folder.

---

## 🔍 Auto-Discovery Mechanism

### File Location Requirements

GitHub Copilot looks for custom configuration files in **specific locations**:

```
your-project/
└── .github/                                    ← Must be at repository root
    ├── copilot-instructions.md                 ← Exact filename required
    ├── prompts/                                ← Exact folder name
    │   └── *.prompt.md                         ← Files ending in .prompt.md
    └── chatmodes/                              ← Exact folder name
        └── *.chatmode.md                       ← Files ending in .chatmode.md
```

**Critical rules:**
1. ✅ `.github` folder must be at **repository root** (not in subdirectory)
2. ✅ `copilot-instructions.md` must have **exact filename** (case-sensitive on Linux/Mac)
3. ✅ Prompts must be in `prompts/` subfolder with `.prompt.md` extension
4. ✅ Chat modes must be in `chatmodes/` subfolder with `.chatmode.md` extension

---

## 📄 File Format Requirements

### copilot-instructions.md

**Format:**
```markdown
---
description: 'Your project description'
---

# Your Project Instructions

Content in Markdown format...
```

**Requirements:**
- ✅ YAML frontmatter with `description` field (optional but recommended)
- ✅ Markdown content
- ✅ Must be at `.github/copilot-instructions.md` (exact path)

**How Copilot uses it:**
- Automatically loaded when workspace is opened
- Applied to ALL Copilot interactions in this workspace
- No manual activation needed
- Appears in "References" when Copilot uses it

---

### Prompt Files (*.prompt.md)

**Format:**
```markdown
---
mode: 'agent'
tools: ['codebase', 'terminalCommand']
description: 'What this prompt does'
---

# Prompt Title

Detailed instructions...
```

**Requirements:**
- ✅ YAML frontmatter with:
  - `mode` (optional): 'agent' or 'chat'
  - `tools` (optional): Array of tool names
  - `description` (required): Brief description
- ✅ File in `.github/prompts/` folder
- ✅ Filename ending in `.prompt.md`
- ✅ Markdown content

**How to use:**
```
In Copilot Chat:
  #prompt:filename-without-extension

Example:
  #prompt:run-validation-cycle
```

**What happens:**
1. User types `#prompt:run-validation-cycle`
2. Copilot loads `.github/prompts/run-validation-cycle.prompt.md`
3. Uses content as context for conversation
4. Executes instructions in the prompt

---

### Chat Mode Files (*.chatmode.md)

**Format:**
```markdown
---
description: 'Chat mode description'
model: 'claude-sonnet-4'
tools: ['codebase', 'terminalCommand']
---

# Chat Mode Name

You are an expert...

## Your Expertise
...

## Your Approach
...
```

**Requirements:**
- ✅ YAML frontmatter with:
  - `description` (required): Brief description
  - `model` (optional): Specific model to use
  - `tools` (optional): Array of tool names
- ✅ File in `.github/chatmodes/` folder
- ✅ Filename ending in `.chatmode.md`
- ✅ Markdown content defining persona

**How to use:**
```
In Copilot Chat (VS Code):
  @chatmode filename-without-extension

Example:
  @chatmode arz-rl-debugging-partner
```

**What happens:**
1. User types `@chatmode arz-rl-debugging-partner`
2. Copilot loads `.github/chatmodes/arz-rl-debugging-partner.chatmode.md`
3. Adopts the persona/expertise defined in file
4. Maintains that persona for the conversation

**Alternative activation:**
```
"Act as the [Chat Mode Name] defined in the chat mode file."
```

---

## ⚙️ Configuration Setup

### VS Code

**Enable custom instructions:**
1. Open Settings (Ctrl+,)
2. Search: `github copilot instructions`
3. Find: **"GitHub Copilot: Enable Custom Instructions"**
4. Check the box: ✅

**Or via settings.json:**
```json
{
  "github.copilot.enable": {
    "*": true
  },
  "github.copilot.advanced": {
    "customInstructions": true
  }
}
```

**Verify it's working:**
```
1. Open Copilot Chat
2. Ask: "What are the custom instructions for this project?"
3. Should mention your project-specific instructions
```

---

### Visual Studio 2022

**Enable custom instructions:**
1. Tools > Options
2. GitHub > Copilot > Copilot Chat
3. Check: **"Enable custom instructions to be loaded from .github/copilot-instructions.md files and added to requests"**

**Verify it's working:**
```
1. Open Copilot Chat window
2. Ask a project-specific question
3. Check "References" section - should show copilot-instructions.md
```

---

## 🔄 How Instructions are Applied

### Loading Priority

**Copilot loads instructions in this order:**

1. **Global settings** (from Copilot configuration)
2. **Workspace instructions** (`.github/copilot-instructions.md`)
3. **Prompt-specific context** (`#prompt:...`)
4. **Chat mode persona** (`@chatmode ...`)

**Example combination:**
```
Conversation context:
  1. General Copilot knowledge (base)
  2. + .github/copilot-instructions.md (ARZ-RL project context)
  3. + #prompt:run-validation-cycle (specific task instructions)
  
Result: Copilot knows project + executes specific validation cycle
```

---

### Scope of Application

**copilot-instructions.md:**
- ✅ Applied to ALL interactions in workspace
- ✅ Active for inline completions
- ✅ Active for chat conversations
- ✅ Active for refactoring suggestions
- ⚠️ Must be enabled in settings

**Prompt files:**
- ✅ Applied when explicitly referenced (`#prompt:name`)
- ❌ Not automatically loaded
- ✅ Can reference instructions file
- ✅ Can stack with other prompts

**Chat mode files:**
- ✅ Applied when activated (`@chatmode name`)
- ❌ Not automatically loaded
- ✅ Overrides default Copilot persona
- ✅ Can reference instructions and prompts

---

## 📚 Context References

### How to Reference Other Files

**In copilot-instructions.md:**
```markdown
See `.github/prompts/run-validation-cycle.prompt.md` for cycle execution.
See `DEVELOPMENT_CYCLE.md` for detailed analysis.
```

**In prompt files:**
```markdown
Follow the `SOP-1: Core Validation Cycle` defined in 
`.github/copilot-instructions.md`.
```

**In chat mode files:**
```markdown
I follow our established `SOP-1: The Core Validation Cycle` without deviation.

Reference: `.github/copilot-instructions.md`, `DEVELOPMENT_CYCLE.md`
```

**Cross-referencing benefits:**
- Single source of truth
- Consistent behavior
- Easy updates (change in one place)

---

## 🧪 Testing Your Setup

### Test 1: Instructions Loaded

**Ask Copilot:**
```
"What are the core development principles for this project?"
```

**Expected response should include:**
- Log-driven development
- SOP-1: Core Validation Cycle
- 75.1% success rate
- Bug #N tracking

**If not working:**
- Check settings enabled
- Verify file path: `.github/copilot-instructions.md`
- Restart VS Code
- Check frontmatter format

---

### Test 2: Prompt Works

**In Copilot Chat:**
```
#prompt:run-validation-cycle
```

**Expected response:**
```
[CYCLE N STARTING]

Starting point: ...
Goal: Achieve validation_success: true

Step 1: Analyzing session_summary.json...
```

**If not working:**
- Check file in `.github/prompts/` folder
- Verify `.prompt.md` extension
- Check frontmatter has `description` field
- Try full path: `.github/prompts/run-validation-cycle.prompt.md`

---

### Test 3: Chat Mode Activates

**In Copilot Chat:**
```
@chatmode arz-rl-debugging-partner
```

**Expected response:**
```
I'm your ARZ-RL Debugging Partner. I'm methodical, persistent, 
and evidence-based. Let's systematically identify and eliminate 
bugs until we achieve validation_success: true.

How can I help you today?
```

**If not working:**
- Check file in `.github/chatmodes/` folder
- Verify `.chatmode.md` extension
- Check frontmatter has required fields
- Try: "Act as the ARZ-RL Debugging Partner"

---

## 📊 Monitoring Usage

### Check if Instructions are Being Used

**VS Code:**
1. Open Copilot Chat
2. Ask a question
3. Look at response - should show "References" section
4. Should list: `.github/copilot-instructions.md`

**Visual Studio:**
1. Open Copilot Chat
2. After response, check "References" tab
3. Should show custom instructions file

---

### Debugging Not Working

**Checklist:**
```
[ ] Settings enabled for custom instructions
[ ] File at correct path: .github/copilot-instructions.md
[ ] Frontmatter format correct (YAML with description)
[ ] File is valid Markdown
[ ] VS Code / Visual Studio restarted after creating files
[ ] Workspace is open (not just a single file)
[ ] File committed to repository (if using remote)
```

**Common issues:**

**Issue:** "Instructions not loading"
```
Cause: Settings not enabled
Fix: Enable "GitHub Copilot: Enable Custom Instructions" in settings
```

**Issue:** "Prompt not found"
```
Cause: Wrong path or filename
Fix: Ensure .github/prompts/name.prompt.md (exact structure)
```

**Issue:** "Chat mode not working"
```
Cause: Missing frontmatter fields
Fix: Add description, model, tools to frontmatter
```

---

## 🎯 Best Practices

### 1. Single Source of Truth

**DO:** Keep core patterns in `copilot-instructions.md`
```markdown
## Proven Tool Sequences

Pattern #1: read_file × 3 → analyze → implement
Success rate: 85%
```

**Reference in prompts:**
```markdown
Use Pattern #1 from copilot-instructions.md for context gathering.
```

**Benefits:**
- Update once, affects all prompts/chat modes
- Consistency across all interactions
- Easy to maintain

---

### 2. Hierarchical Information

**copilot-instructions.md:**
- Project overview
- Core principles
- File structure
- Common patterns

**Prompts:**
- Specific tasks
- Step-by-step procedures
- When to use what tool

**Chat modes:**
- Persona/expertise
- Communication style
- Domain knowledge

---

### 3. Evidence-Based Content

**DO:** Include metrics and data
```markdown
## Quick Test First (90% bugs in 10% time)
Based on 369 analyzed cycles, quick tests detect 90% of bugs 
while using only 10% of the time/resources.
```

**DON'T:** Make vague claims
```markdown
## Quick Test First
Quick tests are better than full tests.
```

---

### 4. Clear Examples

**DO:** Show concrete examples
```markdown
## Bug Announcement Format

Example:
🎯 **BUG #13 DISCOVERED!**

Summary: Inflow BC missing momentum term
Evidence: Line 445: [BC UPDATE] w_in=10.0, calculated without ρv²
Root Cause: boundary_conditions.py:287 uses w = ρv instead of w = ρv + ρv²/(2γ)
```

**DON'T:** Just describe
```markdown
## Bug Announcement Format

Announce bugs with number, summary, evidence, and root cause.
```

---

## 📖 Additional Resources

### Official Documentation

- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [Custom Instructions](https://docs.github.com/en/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot)
- [VS Code Copilot Extension](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot)

### Community Examples

- [Awesome GitHub Copilot](https://github.com/github/awesome-copilot)
- [Copilot Instructions Samples](https://github.com/github/awesome-copilot/blob/main/README.instructions.md)
- [Copilot Prompt Examples](https://github.com/github/awesome-copilot/blob/main/README.prompts.md)

---

## 🔧 Advanced Configuration

### Using applyTo for Conditional Instructions

**In copilot-instructions.md:**
```markdown
---
description: 'ARZ-RL Project Instructions'
applyTo: ['**/*.py', '**/*.md']
---

These instructions apply only to Python and Markdown files.
```

**Scope:**
- `'**'` - All files
- `'**/*.py'` - Only Python files
- `'src/**'` - Only files in src/ folder

---

### Multiple Instruction Files

**Structure:**
```
.github/
├── copilot-instructions.md      # Main instructions
├── python.instructions.md        # Python-specific (if needed)
└── docs.instructions.md          # Documentation-specific
```

**In main instructions:**
```markdown
For Python development, see `.github/python.instructions.md`
For documentation, see `.github/docs.instructions.md`
```

---

### Environment-Specific Instructions

**In copilot-instructions.md:**
```markdown
## Development Environment

### Local Development
- Use `--quick-test` flag
- Test locally when possible

### Kaggle Validation
- Always push before launching kernel
- Monitor for 15+ minutes
- Download logs immediately
```

---

## ✅ Verification Checklist

Before considering your setup complete:

```
[ ] copilot-instructions.md exists at .github/copilot-instructions.md
[ ] File has valid YAML frontmatter
[ ] Custom instructions enabled in IDE settings
[ ] Copilot mentions instructions when asked about project
[ ] Prompt files in .github/prompts/ with .prompt.md extension
[ ] Prompts load when referenced with #prompt:name
[ ] Chat mode files in .github/chatmodes/ with .chatmode.md extension
[ ] Chat modes activate with @chatmode name
[ ] All files committed to repository
[ ] Team members can access same instructions
[ ] Documentation (README.md) explains usage
[ ] Examples tested and working
```

---

**Your GitHub Copilot custom instructions are now properly configured! 🎉**

Copilot will automatically use:
- ✅ `.github/copilot-instructions.md` for all interactions
- ✅ Prompts when you reference them with `#prompt:name`
- ✅ Chat modes when you activate them with `@chatmode name`

**Next:** Test with a simple question to verify everything works!
