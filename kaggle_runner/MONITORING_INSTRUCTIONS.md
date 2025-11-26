# IMPORTANT: MONITORING AND RECOVERY INSTRUCTIONS

## 🛑 CRITICAL: DO NOT INTERRUPT MONITORING
When running `executor.py`, you **MUST NOT** interrupt the process. The script monitors the kernel and **automatically downloads** the results upon completion. Interrupting it breaks this cycle and leaves you without data.

## 🔄 Manual Recovery Command
If monitoring WAS interrupted (or failed), you must manually download the results using this command:

```powershell
# Set config dir if needed (usually d:\Projets\Alibi\Code project)
$env:KAGGLE_CONFIG_DIR = "d:\Projets\Alibi\Code project"

# Download output to the specific results folder
kaggle kernels output elonmj/generic-test-runner-kernel -p "kaggle/results/generic-test-runner-kernel" --force
```

## 📋 Workflow Checklist
1. Launch test: `python kaggle_runner/executor.py ...`
2. **WAIT** for "Monitor finished" message.
3. If interrupted, run the **Manual Recovery Command** above immediately.
