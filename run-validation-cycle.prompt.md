---
mode: 'agent'
tools: ['codebase', 'terminalCommand']
description: 'Executes one full ARZ-RL validation cycle: launch, download, analyze, diagnose Bug #N, fix, commit, and prepare for the next loop.'
---

# Execute One ARZ-RL Validation Cycle

Your goal is to execute one complete iteration of our established development workflow. The ultimate objective is to achieve `validation_success: true` with non-zero performance improvements.

## Instructions

1.  **Follow the `SOP-1: The Core Validation Cycle`** defined in your `arz-rl-validator.instructions.md`.

2.  **Start Point**: The last kernel execution has finished. You have access to its log file. If I provide a `#file:path/to/log`, use that as your starting point for analysis. If not, find the latest execution results in the `validation_output` directory.

3.  **Execute the Cycle**:
    -   **Analyze**: Perform a deep analysis of the log to find the root cause of the failure.
    -   **Document**: Identify the issue as the next sequential bug (`Bug #N`) and create the corresponding documentation file (`docs/BUG_FIX_....md`).
    -   **Fix**: Correct the bug in the codebase.
    -   **Deploy**: Commit and push the fix to GitHub with a detailed message.
    -   **Relaunch**: Launch a new Kaggle quick test.
    -   **Wait**: Monitor the kernel until it completes, ensuring you successfully download the new log for the next cycle.

4.  **Report**: Once the new kernel is complete, provide a summary of its outcome. If it still fails, be prepared to start the next cycle immediately.