# Bug #36 Kaggle Deployment and Testing Guide

## Quick Reference

**Bug**: GPU inflow BC injects only 14.7% of configured density (0.044 vs 0.3 veh/m)  
**Fix**: Thread `current_bc_params` through GPU call stack (10 modifications, 3 files)  
**Status**: ✅ Code complete, ready for GPU testing  
**Test Environment**: Kaggle GPU kernels (CUDA toolkit required)

---

## Deployment Steps

### 1. Commit Fixed Code to Repository

```bash
cd "d:\Projets\Alibi\Code project"

# Stage modified files
git add arz_model/numerics/reconstruction/weno_gpu.py
git add arz_model/numerics/time_integration.py
git add arz_model/simulation/runner.py

# Commit with descriptive message
git commit -m "Fix Bug #36: GPU inflow boundary condition parameter propagation

Root cause: GPU spatial discretization bypassed dispatcher,
used static params.boundary_conditions instead of dynamic
current_bc_params updated during simulation.

Fix: Thread current_bc_params through entire GPU call stack
- weno_gpu.py: Use dispatcher with current_bc_params (2 changes)
- time_integration.py: Add parameter to 9 functions
- runner.py: Pass self.current_bc_params at 4 call sites

Impact: Fixes inflow density (was 14.7% of target), enables
queue detection, allows RL training to converge.

Validation: Ready for Kaggle GPU testing."

# Push to remote
git push origin main
```

### 2. Update Kaggle Kernel

#### Option A: Auto-Sync (If GitHub Integration Enabled)
1. Open Kaggle kernel in browser
2. Click "File" → "Add or Upload Data"
3. Select "Dataset from GitHub"
4. Select your repository
5. Kaggle auto-pulls latest commit
6. Click "Save Version" → "Save & Run All"

#### Option B: Manual Upload
1. Create dataset archive:
   ```bash
   cd "d:\Projets\Alibi\Code project"
   # Zip only the necessary files
   tar -czf arz_model_bug36_fix.tar.gz arz_model/ Code_RL/
   ```
2. Upload to Kaggle:
   - Go to kaggle.com/datasets
   - Click "New Dataset"
   - Upload `arz_model_bug36_fix.tar.gz`
   - Title: "ARZ Model - Bug #36 Fix"
3. Update kernel data sources:
   - Open your kernel
   - Click "Add data" → Select uploaded dataset
   - Remove old dataset version
   - Click "Save Version"

---

## Testing Procedure

### Select Test Kernel

Choose one of the 6 failed kernels showing Bug #36 symptoms:
- `rl-kernel-*-single-intersection-heavy-inflow-*`
- Any kernel with ρ_observed ≈ 0.044 veh/m (14.7% ratio)

### Monitoring During Run

Watch console output for these diagnostic logs:

```
[BC UPDATE] left à phase 0 RED (reduced inflow)
 └─ Inflow state: rho_m=0.XXXX, w_m=Y.Y, ...
```

**Before Fix (Bug #36)**:
```
[QUEUE_DIAGNOSTIC] densities_m (veh/m): [0.000044 0.000044 0.000044 ...]
[QUEUE_DIAGNOSTIC] velocities_m (m/s): [11.111 11.111 11.111 ...]  # Constant!
[QUEUE_DIAGNOSTIC] queue_length=0.00 vehicles  # Always zero!
[REWARD_MICROSCOPE] R_queue=-0.0000  # Always zero!
```

**After Fix (Expected)**:
```
[QUEUE_DIAGNOSTIC] densities_m (veh/m): [0.000200 0.000180 0.000160 ...]  # Increasing!
[QUEUE_DIAGNOSTIC] velocities_m (m/s): [8.5 7.2 5.8 4.1 ...]  # Dropping!
[QUEUE_DIAGNOSTIC] queue_length=12.50 vehicles  # Non-zero!
[REWARD_MICROSCOPE] R_queue=-0.0234  # Non-zero!
```

### Key Metrics to Track

| Metric | Before Fix | After Fix (Target) | Success Threshold |
|--------|------------|-------------------|-------------------|
| Upstream Density | 0.044 veh/m | 0.2-0.3 veh/m | ≥ 0.15 veh/m |
| Density Ratio | 14.7% | 80-100% | ≥ 50% |
| Queue Length | Always 0 | > 0 vehicles | > 0 |
| Min Velocity | 11.11 m/s (constant) | < 5 m/s | < 8 m/s |
| R_queue Reward | Always 0 | Non-zero | != 0 |
| RL Learning | Stuck/diverging | Converging | Loss ↓, Reward ↑ |

### Collect Evidence

#### 1. Console Logs
Save the first 200 lines showing initialization and first few steps:
```python
# In Kaggle cell
!python train_rl_agent.py 2>&1 | head -n 200 > bug36_test_log.txt
```

#### 2. Density Evolution Plot
```python
import matplotlib.pyplot as plt
import numpy as np

# Extract from runner output
densities = [step_data['upstream_density'] for step_data in history]
times = [step_data['time'] for step_data in history]

plt.figure(figsize=(10, 5))
plt.plot(times, densities, label='Observed Density')
plt.axhline(y=0.3, color='r', linestyle='--', label='Configured Inflow (0.3 veh/m)')
plt.axhline(y=0.044, color='orange', linestyle='--', label='Bug #36 Level (0.044 veh/m)')
plt.xlabel('Time (s)')
plt.ylabel('Upstream Density (veh/m)')
plt.title('Bug #36 Fix Verification: Inflow Density Propagation')
plt.legend()
plt.grid(True)
plt.savefig('bug36_density_verification.png')
plt.show()
```

#### 3. Reward Components
```python
# Plot R_queue over time
queue_rewards = [step_data['R_queue'] for step_data in history]
plt.figure(figsize=(10, 5))
plt.plot(times, queue_rewards)
plt.xlabel('Time (s)')
plt.ylabel('R_queue Reward Component')
plt.title('Queue Reward Component (Should be Non-Zero After Fix)')
plt.axhline(y=0, color='r', linestyle='--', label='Bug #36 (Always Zero)')
plt.legend()
plt.grid(True)
plt.savefig('bug36_reward_verification.png')
plt.show()
```

---

## Quick Diagnostic Script

Add this cell to your Kaggle kernel for instant verification:

```python
"""Quick Bug #36 Fix Verification"""

import numpy as np
from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect

# Create environment with heavy inflow
env = TrafficSignalEnvDirect(
    scenario_config_path='scenario_heavy_inflow.yml',
    device='gpu',  # MUST use GPU to test Bug #36 fix
    quiet=False
)

# Run 10 steps
densities = []
queues = []
for step in range(10):
    obs, reward, done, truncated, info = env.step(1)  # Always GREEN
    
    # Extract density from runner state
    U = env.runner.d_U.copy_to_host()  # GPU → CPU
    rho_upstream = np.mean(U[0, 2:7])  # First 5 physical cells, motorcycles
    queue = env.previous_queue_length
    
    densities.append(rho_upstream)
    queues.append(queue)
    
    print(f"Step {step:2d}: ρ={rho_upstream:.6f} veh/m, queue={queue:.2f} veh")

# Verdict
mean_density = np.mean(densities[3:])  # Ignore first 3 for transient
max_queue = np.max(queues)

print("\n" + "="*60)
print("BUG #36 FIX VERIFICATION")
print("="*60)

if mean_density >= 0.15 and max_queue > 0:
    print("✅ PASS: Bug #36 is FIXED!")
    print(f"  • Mean density: {mean_density:.6f} veh/m ({mean_density/0.3*100:.1f}% of target)")
    print(f"  • Max queue: {max_queue:.2f} veh")
else:
    print("❌ FAIL: Bug #36 still present")
    print(f"  • Mean density: {mean_density:.6f} veh/m (expected ≥ 0.15)")
    print(f"  • Max queue: {max_queue:.2f} veh (expected > 0)")
```

---

## Troubleshooting

### Issue: Density still ~0.044 veh/m after fix

**Possible causes:**
1. Code not updated - Check git commit hash matches
2. Cache not cleared - Restart Kaggle kernel runtime
3. Wrong device mode - Verify `device='gpu'` (not 'cpu')
4. BC schedule overriding - Check `scenario_config['schedule']['enabled'] = False` for test

**Debug steps:**
```python
# Add temporary debug prints
# In weno_gpu.py line 302 (after fix):
if current_bc_params is not None:
    print(f"[DEBUG] weno_gpu: current_bc_params passed: {current_bc_params}")
else:
    print(f"[DEBUG] weno_gpu: WARNING - current_bc_params is None!")
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size or grid resolution:
```python
scenario_config['N'] = 30  # Reduce from 50
scenario_config['xmax'] = 300.0  # Reduce from 500m
```

### Issue: Kernel timeout

**Solution**: Reduce simulation duration:
```python
scenario_config['t_final'] = 60.0  # 1 minute instead of 3
```

---

## Success Checklist

Before declaring Bug #36 fixed, verify ALL:

- [ ] Code uploaded to Kaggle with latest commit
- [ ] Kernel runs without errors on GPU
- [ ] Mean upstream density ≥ 0.15 veh/m (50% of target)
- [ ] Density ratio ≥ 50% (not 14.7%)
- [ ] Max queue length > 0 vehicles (not always 0)
- [ ] Min velocity < 8 m/s (congestion forming)
- [ ] R_queue reward component non-zero
- [ ] RL training loss decreasing (convergence)
- [ ] Density evolution plot shows increase over time
- [ ] Console logs show BC values updating correctly

---

## Reporting Results

### Success Report Template

```markdown
# Bug #36 Fix Verification - SUCCESS ✅

**Kernel**: [kernel-name]  
**Device**: GPU (CUDA 11.x)  
**Commit**: [git hash]  
**Date**: [YYYY-MM-DD]

## Metrics

| Metric | Before Fix | After Fix | Improvement |
|--------|------------|-----------|-------------|
| Upstream Density | 0.044 veh/m | 0.XXX veh/m | XXX% |
| Density Ratio | 14.7% | XX.X% | XXX% |
| Queue Length | 0.00 veh | XX.XX veh | Queue forms |
| Min Velocity | 11.11 m/s | X.XX m/s | Congestion |
| R_queue | 0.0000 | -X.XXXX | Non-zero |

## Evidence

- Console log: [link to bug36_test_log.txt]
- Density plot: ![Density Evolution](bug36_density_verification.png)
- Reward plot: ![Reward Components](bug36_reward_verification.png)

## Conclusion

Bug #36 is FIXED. Inflow boundary condition now correctly propagates
configured density into GPU simulation domain, enabling queue detection
and RL training convergence.

**Ready for full performance benchmark (Section 7.6).**
```

### Failure Report Template

```markdown
# Bug #36 Fix Verification - NEEDS DEBUG ⚠️

**Kernel**: [kernel-name]  
**Device**: GPU (CUDA 11.x)  
**Commit**: [git hash]  
**Date**: [YYYY-MM-DD]

## Observed Behavior

- Upstream density: X.XXXX veh/m (expected ≥ 0.15)
- Queue length: X.XX veh (expected > 0)
- R_queue: X.XXXX (expected != 0)

## Debug Information Needed

- [ ] Verify git commit matches uploaded code
- [ ] Add debug prints to weno_gpu.py line 302
- [ ] Check runner.current_bc_params values
- [ ] Verify GPU kernel input parameters
- [ ] Test on different kernel/scenario

## Next Steps

1. [Action item 1]
2. [Action item 2]
...
```

---

## Related Documentation

- **Bug Report**: `BUG_36_INFLOW_BOUNDARY_CONDITION_FAILURE.md`
- **Fix Summary**: `BUG_36_FIX_SUMMARY.md`
- **Code Changes**: See commit history in git

---

**Last Updated**: 2025-01-XX  
**Author**: AI Assistant  
**Status**: Ready for Kaggle Deployment
