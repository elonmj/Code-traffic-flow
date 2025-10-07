# Quick Test Guide - Section 7.6 RL Performance Validation

## Problem Solved ✅

**Your constraint:** Only 30 hours of Kaggle GPU quota available  
**Solution:** Quick test mode runs in ~15 minutes, validating setup without consuming quota

---

## Quick Test vs Full Test

| Mode | Training Steps | Scenarios | Duration | Kaggle GPU Time |
|------|---------------|-----------|----------|-----------------|
| **Quick Test** | 10 steps | 1 scenario | 10 min sim | **~15 min** ✅ |
| **Full Test** | 20,000 steps | 3 scenarios | 60 min sim | ~2 hours |

---

## How to Run Quick Test

### Option 1: Simple Runner (Recommended)
```bash
cd "d:\Projets\Alibi\Code project"
python validation_ch7/scripts/quick_test_section_7_6.py
```

### Option 2: Direct with Flag
```bash
cd "d:\Projets\Alibi\Code project"
python validation_ch7/scripts/run_kaggle_validation_section_7_6.py --quick
```

### Option 3: Environment Variable
```bash
cd "d:\Projets\Alibi\Code project"
set QUICK_TEST=true
python validation_ch7/scripts/run_kaggle_validation_section_7_6.py
```

---

## What Quick Test Validates

### ✅ Integration Checks
- TrafficSignalEnvDirect initialization works
- GPU device detection (CUDA availability)
- SimulationRunner accepts `device='gpu'` parameter
- Direct coupling (no HTTP/mock) functional

### ✅ Performance Checks
- Step time measurement works
- GPU acceleration active (vs CPU baseline)
- Performance metrics collection
- Wallclock time vs simulated time ratio

### ✅ RL Pipeline Checks
- PPO agent initialization
- Environment step() method works
- Observation building correct
- Reward calculation functional
- Model save/load works

### ✅ Output Generation
- Figures generated (even with minimal data)
- Metrics CSV created
- LaTeX content produced
- Session summary written

---

## What Quick Test Skips

### ⚠️ Not Validated in Quick Mode
- ❌ Training convergence (only 10 steps)
- ❌ Multi-scenario comparison (only 1 scenario)
- ❌ Statistical significance (too few samples)
- ❌ Long-term stability (short episodes)

**For thesis:** You'll need to run **full test** eventually, but quick test confirms everything works first!

---

## Expected Output

### Console Output
```
[QUICK TEST MODE ENABLED]
- Training: 10 timesteps only
- Duration: 10 minutes simulated time
- Scenarios: 1 scenario only
- Expected runtime: ~15 minutes on GPU

[DEVICE] Detected: GPU
[GPU INFO] Tesla T4

[PHASE 1/2] Training RL agents...
[QUICK TEST] Training only: traffic_light_control
[TRAINING] Starting RL training for scenario: traffic_light_control
  Device: gpu
  Total timesteps: 10
  
[PHASE 2/2] Running performance comparisons...
  [PERFORMANCE] Simulation completed:
    - Total steps: 10
    - Avg step time: 0.243s (device=gpu)
    - Speed ratio: 41.15x real-time

[SUCCESS] VALIDATION KAGGLE 7.6 TERMINÉE
```

### Performance Metrics to Look For
- **Step time < 1s:** Confirms GPU acceleration working
- **Speed ratio > 10x:** Simulation faster than real-time
- **No crashes:** Integration stable
- **Figures generated:** Output pipeline works

---

## Interpreting Results

### ✅ Success Indicators
```
Avg step time: 0.2-0.5s (device=gpu)     ← GPU working!
Speed ratio: 20-50x real-time            ← Excellent performance
Environment initialized successfully      ← Integration OK
Model saved to rl_agent_*.zip            ← Training pipeline OK
```

### ❌ Failure Indicators
```
Avg step time: > 5s                      ← GPU not used (CPU fallback)
CUDA not available                       ← GPU allocation failed
Failed to initialize TrafficSignalEnvDirect  ← Integration broken
ImportError: No module named 'Code_RL'   ← Path issue
```

---

## Debugging Quick Test

### If GPU not detected:
```python
# Check in Kaggle kernel:
from numba import cuda
print(f"CUDA available: {cuda.is_available()}")
print(f"GPU: {cuda.get_current_device().name.decode() if cuda.is_available() else 'None'}")
```

### If step time too slow:
- Verify `device='gpu'` passed to TrafficSignalEnvDirect
- Check runner.device is 'gpu' not 'cpu'
- Confirm Kaggle kernel has GPU accelerator enabled

### If import errors:
- Check sys.path includes Code_RL directory
- Verify GitHub repo cloned correctly
- Check file structure matches expected layout

---

## Next Steps After Quick Test

### ✅ If Quick Test PASSES (15 min):
1. **Celebrate!** Integration works correctly
2. **Plan full test:** Schedule 2-hour window in your 30h quota
3. **Run full validation:**
   ```bash
   python validation_ch7/scripts/run_kaggle_validation_section_7_6.py
   # (no --quick flag)
   ```
4. **Get thesis results:** Full training convergence + 3 scenarios

### ❌ If Quick Test FAILS:
1. **Check Kaggle logs:** Identify exact error
2. **Fix locally:** Iterate on code
3. **Re-run quick test:** Validate fix (another 15 min)
4. **Repeat:** Until quick test passes

---

## Time Budget Planning

| Activity | Duration | Cumulative |
|----------|----------|------------|
| Quick test #1 (initial) | 15 min | 15 min |
| Debug + fixes (if needed) | 1-2 hours | - |
| Quick test #2 (verify fix) | 15 min | 30 min |
| **Full test (thesis results)** | 120 min | **2.5 hours** |
| **Total Kaggle GPU quota used** | - | **~3 hours** ✅ |

**Remaining quota:** 27 hours for other sections or re-runs

---

## Pro Tips

### 🎯 Maximize Efficiency
- Run quick test **first** before touching full test
- Fix all issues in quick mode (saves quota)
- Only run full test when quick test passes cleanly
- Use local CPU for code development (even if slow)

### 🔍 Monitor Progress
- Watch Kaggle kernel live URL
- Check session_summary.json appearance
- Monitor step time trends
- Verify GPU utilization in logs

### 💾 Save Results
- Download quick test outputs (practice for full test)
- Compare GPU step times with CPU baseline
- Document performance improvements
- Keep logs for debugging reference

---

## Summary

**Quick test is your safety net:**
- ✅ Validates integration in 15 minutes
- ✅ Uses minimal Kaggle quota
- ✅ Catches issues early
- ✅ Proves GPU acceleration works
- ✅ Confirms pipeline end-to-end

**Full test gets thesis data:**
- 📊 Real training convergence
- 📊 Multi-scenario comparison  
- 📊 Statistical significance
- 📊 Publication-ready figures

**Strategy:** Quick test now → Fix any issues → Full test once → Thesis done! 🎓
