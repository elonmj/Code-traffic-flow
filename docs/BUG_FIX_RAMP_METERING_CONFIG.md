# Bug #21: Ramp Metering Configuration - Variable Undefined

**Status**: 🔴 **ACTIVE** - Discovered during validation run  
**Severity**: **HIGH** - Blocks completion of 2 out of 3 validation scenarios  
**Date Discovered**: October 11, 2025  
**Session**: arz-validation-76rlperformance-xwvi  
**Affects**: Ramp metering and adaptive speed control scenarios

---

## 🔍 **Problem Description**

During validation run, after successfully completing the traffic_light_control scenario, the test attempted to run the ramp_metering scenario and crashed with a `NameError`.

### **Error Message**:
```python
NameError: name 'rho_m_high_si' is not defined

Traceback (most recent call last):
  File "test_section_7_6_rl_performance.py", line 1179, in main
    success = test.run_all_tests()
  File "test_section_7_6_rl_performance.py", line 888, in run_all_tests
    self.train_rl_agent(scenario, device=device)
  File "test_section_7_6_rl_performance.py", line 600, in train_rl_agent
    scenario_path = self._create_scenario_config(scenario_type)
  File "test_section_7_6_rl_performance.py", line 196, in _create_scenario_config
    'U_L': [rho_m_high_si*0.8, w_m_high, rho_c_high_si*0.8, w_c_high],
            ^^^^^^^^^^^^^
NameError: name 'rho_m_high_si' is not defined
```

---

## 📂 **Location**

**File**: `validation_ch7/scripts/test_section_7_6_rl_performance.py`  
**Method**: `_create_scenario_config(self, scenario_type: str)`  
**Line**: 196 (in ramp_metering configuration block)

---

## 🔬 **Root Cause Analysis**

### **Problem**:
The `_create_scenario_config` method defines scenario-specific variables within conditional blocks. For the `ramp_metering` scenario, the code references variables that were only defined in the `traffic_light_control` block.

### **Expected Behavior**:
Each scenario type should have its own complete set of variable definitions:
- Density parameters: `rho_m_high_si`, `rho_c_high_si`
- Velocity parameters: `w_m_high`, `w_c_high`
- Boundary conditions: `U_L`, `U_R`

### **Actual Behavior**:
Variables defined in `traffic_light_control` block are not accessible in `ramp_metering` block due to Python scoping rules. Each scenario block needs its own definitions.

---

## 📊 **Impact Assessment**

### **Scenarios Affected**:

| Scenario | Status | Reason |
|----------|--------|--------|
| Traffic Light Control | ✅ **COMPLETED** | Variables defined correctly |
| Ramp Metering | ❌ **BLOCKED** | Missing variable definitions |
| Adaptive Speed Control | ⏸️ **PENDING** | Not reached due to ramp_metering failure |

### **Validation Completeness**:
- **Current**: 1 / 3 scenarios (33%)
- **Target**: 3 / 3 scenarios (100%)
- **Blocking**: Bug #21 prevents 67% of validation

### **Thesis Impact**:
- Cannot claim comprehensive validation of RL approach
- Missing ramp metering results (important for highway control)
- Incomplete performance comparison data

---

## 🔧 **Solution Design**

### **Strategy**: Define scenario-specific variables in each block

### **Implementation**:

1. **Locate the `_create_scenario_config` method** (line ~160-250)

2. **For `ramp_metering` scenario block**, add complete variable definitions:

```python
elif scenario_type == 'ramp_metering':
    # ===================================================================
    # RAMP METERING SCENARIO - Highway On-Ramp Control
    # ===================================================================
    
    # Define scenario-specific parameters (SI units: veh/m, m/s)
    # High density for congested highway conditions
    rho_m_high_si = 0.18  # veh/m (motorcycles, high density)
    rho_c_high_si = 0.12  # veh/m (cars, high density)
    
    # Reduced speeds for congested conditions
    w_m_high = 8.0   # m/s (motorcycles, ~29 km/h congested)
    w_c_high = 6.0   # m/s (cars, ~22 km/h congested)
    
    # Boundary conditions for ramp metering
    # Left (upstream): Highway mainline traffic (moderate density)
    # Right (downstream): Merge zone (controlled ramp flow)
    config['boundary_conditions'] = {
        'left': {
            'type': 'traffic_signal',
            'signal_id': 'ramp_meter',
            'phases': [
                {
                    'phase_id': 0,
                    'name': 'METER_ON',
                    'state': 'U_L': [rho_m_high_si*0.6, w_m_high*1.2, rho_c_high_si*0.6, w_c_high*1.2],
                },
                {
                    'phase_id': 1,
                    'name': 'METER_OFF',
                    'state': 'U_L': [rho_m_high_si*0.3, w_m_high*1.5, rho_c_high_si*0.3, w_c_high*1.5],
                }
            ]
        },
        'right': {
            'type': 'free_flow',
            'state': 'U_R': [rho_m_high_si*0.8, w_m_high, rho_c_high_si*0.8, w_c_high]
        }
    }
```

3. **Similarly, for `adaptive_speed` scenario block**, add:

```python
elif scenario_type == 'adaptive_speed':
    # ===================================================================
    # ADAPTIVE SPEED CONTROL SCENARIO - Variable Speed Limits
    # ===================================================================
    
    # Define scenario-specific parameters
    rho_m_high_si = 0.15  # veh/m (motorcycles, moderate density)
    rho_c_high_si = 0.10  # veh/m (cars, moderate density)
    
    # Variable speeds based on control strategy
    w_m_high = 12.0  # m/s (motorcycles, ~43 km/h)
    w_c_high = 10.0  # m/s (cars, ~36 km/h)
    
    # Rest of adaptive_speed configuration...
```

---

## ✅ **Validation Plan**

### **Step 1: Local Testing**
```bash
# Test scenario creation independently
python -c "
from validation_ch7.scripts.test_section_7_6_rl_performance import TestSection76RLPerformance
test = TestSection76RLPerformance()
path = test._create_scenario_config('ramp_metering')
print(f'✅ Ramp metering config created: {path}')
"
```

### **Step 2: Dry-Run Validation**
```bash
# Run with quick-test mode to verify all scenarios initialize
python validation_ch7/scripts/run_kaggle_validation_section_7_6.py --quick-test
```

### **Step 3: Full Validation**
```bash
# Full run after local validation
python validation_ch7/scripts/run_kaggle_validation_section_7_6.py
```

---

## 📝 **Implementation Checklist**

- [ ] Read current `_create_scenario_config` method (lines 160-250)
- [ ] Identify exact line numbers for ramp_metering and adaptive_speed blocks
- [ ] Add variable definitions to ramp_metering block
- [ ] Add variable definitions to adaptive_speed block
- [ ] Verify Python syntax with `py_compile`
- [ ] Test scenario creation locally
- [ ] Commit fix with descriptive message
- [ ] Push to GitHub
- [ ] Rerun full validation
- [ ] Verify all 3 scenarios complete
- [ ] Update VALIDATION_SUCCESS report

---

## 🎯 **Success Criteria**

- [ ] All 3 scenarios create configuration files without errors
- [ ] Training completes for traffic_light_control (already done ✅)
- [ ] Training completes for ramp_metering
- [ ] Training completes for adaptive_speed
- [ ] `rl_performance_comparison.csv` contains all 3 scenarios
- [ ] Figures show comparison for all scenarios
- [ ] `validation_success: true` in session_summary.json

---

## 📊 **Expected Impact After Fix**

### **Before Fix**:
- 1 scenario completed (traffic_light_control)
- Empty comparison CSV
- Incomplete thesis validation

### **After Fix**:
- 3 scenarios completed (100% coverage)
- Full comparison metrics:
  - RL vs Baseline for traffic signals
  - RL vs Baseline for ramp metering
  - RL vs Baseline for adaptive speed
- Complete thesis validation ready
- Comprehensive performance analysis

---

## 🔗 **Related Issues**

- ✅ Bug #19 (Timeout) - **RESOLVED** (commit 02996ec)
- ✅ Bug #20 (Decision Interval) - **RESOLVED** (commit 1df1960)
- 🔴 Bug #21 (Ramp Metering Config) - **ACTIVE** (this document)

---

## 📚 **References**

- File: `validation_ch7/scripts/test_section_7_6_rl_performance.py`
- Method: `_create_scenario_config(scenario_type: str)`
- Previous success: Traffic light control scenario (lines ~170-195)
- Failed scenario: Ramp metering (line 196 - undefined variable)

---

## 🚀 **Next Session Objectives**

1. **Read** `_create_scenario_config` method completely
2. **Add** variable definitions to ramp_metering block
3. **Add** variable definitions to adaptive_speed block
4. **Test** scenario creation for all types
5. **Commit** and push fix
6. **Rerun** full validation
7. **Verify** all scenarios complete
8. **Generate** final thesis-ready results

---

**Bug Report Created**: October 11, 2025  
**Priority**: HIGH - Blocks 67% of validation  
**Estimated Fix Time**: 30 minutes  
**Estimated Test Time**: 2-3 hours (full validation run)

---

**Status**: 🔴 **AWAITING FIX** - Ready for implementation in next session
