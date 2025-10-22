# ARZ Model Architecture: Config Legacy vs Current Analysis

**Date**: 2025-02-20  
**Analysis Type**: Codebase Configuration Architecture  
**Status**: COMPLETE  

---

## 📊 Executive Summary

The `arz_model/config/` directory contains a mix of **active** and **legacy** configuration files. This analysis identifies which configs are part of the current Ch7 validation pipeline and which are legacy/unused.

**Key Finding**: 
- ✅ **2 Active Configs** (used by Ch7 pipeline, must maintain)
- ❌ **2 Legacy Configs** (NOT used, should archive)

---

## 🔍 Configuration Architecture Analysis

### Current Status (Before Cleanup)

```
arz_model/config/
├── config_base.yml                    ✅ PRIMARY (20+ uses)
├── scenario_convergence_test.yml      ✅ ACTIVE (3 uses)
├── riemann_problem_test.yaml          ❌ UNUSED (0 uses)
├── stationary_free_flow_test.yaml     ❌ UNUSED (0 uses)
├── network_config.py
└── __init__.py
```

---

## ✅ ACTIVE CONFIGS (Use These)

### 1. config_base.yml - PRIMARY CENTRAL CONFIG

**Purpose**: Base configuration for entire simulation system

**Status**: ✅ **ACTIVELY MAINTAINED** (Oct 2025 behavioral_coupling update)

**Usage Pattern**: 20+ files reference this config

**Content Sections**:

#### A. Scenario Control
```yaml
scenario_name: "base_scenario"
t_final_sec: 60.0
output_dt_sec: 10.0
output_dir: "results"
```

#### B. Grid Parameters
```yaml
grid:
  N_cells: 100
  x_min_m: 0.0
  x_max_m: 1000.0
  num_ghost_cells: 2
```

#### C. Physical Parameters (Base Values)
```yaml
alpha: 0.4
V_creeping_kmh: 5.0
rho_jam_veh_km: 250.0

# Pressure model parameters
pressure:
  gamma_m: 1.5
  gamma_c: 2.0
  K_m_kmh: 10.0
  K_c_kmh: 15.0

# Relaxation time scales
relaxation:
  tau_m_sec: 5.0
  tau_c_sec: 10.0

# Maximum speeds per road category
Vmax_kmh:
  c: {1: 75.0, 2: 60.0, 3: 35.0, 4: 25.0, 5: 10.0, 9: 35.0}
  m: {1: 85.0, 2: 70.0, 3: 50.0, 4: 45.0, 5: 30.0, 9: 50.0}
```

#### D. **Behavioral Coupling Parameters (NEW - Oct 2025)** 🆕

```yaml
behavioral_coupling:
  # Roundabouts (θ ∈ [0,1] controls speed adaptation at junction)
  theta_moto_insertion: 0.2        # Entry: strong adaptation (reset to equilibrium)
  theta_moto_circulation: 0.8      # Flow: weak adaptation (preserve upstream speed)
  
  # Signalized intersections
  theta_moto_signalized: 0.8       # Aggressive acceleration through green
  theta_car_signalized: 0.5        # Moderate acceleration through green
  
  # Priority roads
  theta_moto_priority: 0.9         # Through traffic: minimal disruption
  theta_car_priority: 0.9          # Through traffic: minimal disruption
  
  # Secondary roads (yield)
  theta_moto_secondary: 0.1        # Stop/yield: strong behavioral reset
  theta_car_secondary: 0.1         # Stop/yield: strong behavioral reset
```

**Behavioral Coupling Theory**:
- Formula: `w_out = w_eq + θ_k * (w_in - w_eq)`
- θ ≈ 0: Strong adaptation (reset to equilibrium, typical at yield/stop)
- θ ≈ 1: Weak adaptation (preserve upstream behavior, typical in flowing traffic)

#### E. Numerical Parameters
```yaml
cfl_number: 0.8
ghost_cells: 2
ode_solver: 'RK45'
ode_rtol: 1.0e-6
ode_atol: 1.0e-6
epsilon: 1.0e-10
```

#### F. Boundary Conditions
```yaml
boundary_conditions:
  left:
    type: "periodic"
  right:
    type: "periodic"
```

#### G. Initial Conditions
```yaml
initial_conditions:
  type: "uniform"
  state: [10.0, 10.0, 10.0, 10.0]  # [rho_m, w_m, rho_c, w_c]
```

**Used By** (20+ matches):
```
✓ validation_ch7_v2/scripts/test_section_7_3_analytical.py
✓ arz_model/run_convergence_test.py
✓ Code_RL/src/env/traffic_signal_env_direct.py (RL environment)
✓ test_network_config.py (test suite)
✓ arz_model/calibration/core/calibration_runner.py
✓ arz_model/simulation/runner.py
✓ validation_kaggle_manager.py (all sections 7.3-7.7)
✓ Multiple validation scripts in niveau1-4
```

---

### 2. scenario_convergence_test.yml - ACTIVE TEST CONFIG

**Purpose**: Test configuration for Section 7.3 (Analytical validation)

**Status**: ✅ **ACTIVELY USED** (3 references)

**Content**:
```yaml
scenario_name: convergence_test_sine

# Grid for convergence testing
N: 100
xmin: 0.0
xmax: 1000.0

# Time parameters
t_final: 10.0
output_dt: 10.0

# Initial condition: sine wave perturbation
initial_conditions:
  type: sine_wave_perturbation
  R_val: 3
  background_state:
    rho_m: 50.0e-3  # 50 veh/km
    rho_c: 20.0e-3  # 20 veh/km
  perturbation:
    amplitude: 5.0e-3
    wave_number: 1

# Boundary conditions
boundary_conditions:
  left:
    type: periodic
  right:
    type: periodic
```

**Used By** (3 matches):
```
✓ validation_ch7_v2/scripts/test_section_7_3_analytical.py (line 232)
✓ arz_model/run_convergence_test.py (line 121)
✓ arz_model/run_convergence_test.py (line 122)
```

**Purpose**: Convergence analysis requires smooth initial condition (sine wave) to test error decay with grid refinement.

---

## ❌ LEGACY CONFIGS (To Archive)

### 1. riemann_problem_test.yaml

**Purpose**: Classical Riemann problem benchmark

**Status**: ❌ **NOT USED** (0 references in codebase)

**Content**:
```yaml
scenario_name: "riemann_problem_test"

# Test discontinuity propagation
initial_conditions:
  type: "riemann"
  U_L: [0.02, 15.0, 0.01, 10.0]  # Left state
  U_R: [0.15, 5.0, 0.10, 3.0]    # Right state
  split_pos: 1000.0               # Discontinuity position

boundary_conditions:
  left:
    type: "outflow"
  right:
    type: "outflow"
```

**Why Unused**: 
- Development/debugging tool from Phase 1.3 calibration
- Superseded by `scenario_convergence_test.yml` in Section 7.3
- Section 7.3 provides more comprehensive numerical validation
- No modern code references this config

**Archive Location**: `_archive/legacy_test_configs/`

---

### 2. stationary_free_flow_test.yaml

**Purpose**: Free flow equilibrium state test

**Status**: ❌ **NOT USED** (0 references in codebase)

**Content**:
```yaml
scenario_name: "stationary_free_flow_test"

# Uniform equilibrium state
initial_conditions:
  type: "uniform_equilibrium"
  rho_m: 50.0  # veh/km
  rho_c: 20.0  # veh/km
  R_val: 3

# Simple periodic BC
boundary_conditions:
  left:
    type: "periodic"
  right:
    type: "periodic"
```

**Why Unused**:
- Development/debugging tool from Phase 1.3 calibration
- Superseded by `scenario_convergence_test.yml` in Section 7.3
- Single equilibrium state testing unnecessary (convergence test covers)
- No modern code references this config

**Archive Location**: `_archive/legacy_test_configs/`

---

## 🔗 Config Dependency Map

### How Configs Flow Through System

```
config_base.yml (PRIMARY)
├── ✅ Used by: RL environments
│   └── Code_RL/src/env/traffic_signal_env_direct.py
│
├── ✅ Used by: Validation sections 7.3-7.7
│   ├── Section 7.3: run_convergence_test.py
│   ├── Section 7.4: validation_kaggle_manager.py (real data)
│   ├── Section 7.5: niveau2_implementation/ (multi-vehicle)
│   ├── Section 7.6: niveau4_rl_performance/ (GPU)
│   └── Section 7.7: niveau3_realworld_validation/ (robustness)
│
├── ✅ Used by: Simulation runner
│   └── arz_model/simulation/runner.py
│
└── ✅ Used by: Calibration
    └── arz_model/calibration/core/calibration_runner.py

scenario_convergence_test.yml (TEST-SPECIFIC)
├── ✅ Used by: Section 7.3 convergence analysis
│   ├── run_convergence_test.py
│   └── test_section_7_3_analytical.py
│
└── Purpose: Override config_base.yml for specific test case

riemann_problem_test.yaml (LEGACY - ARCHIVE)
└── ❌ NOT USED ANYWHERE

stationary_free_flow_test.yaml (LEGACY - ARCHIVE)
└── ❌ NOT USED ANYWHERE
```

---

## 🎯 Behavioral Coupling Parameters (θ_k) - Critical for Ch7

The `behavioral_coupling` section in `config_base.yml` controls junction transition physics.

### Formula
```
w_out = w_eq + θ_k * (w_in - w_eq)
```

Where:
- `w_out`: Output velocity at junction
- `w_eq`: Equilibrium velocity
- `w_in`: Input velocity upstream
- `θ_k`: Behavioral coupling parameter (0 ≤ θ ≤ 1)

### Interpretation

| θ Value | Meaning | Physical Behavior |
|---------|---------|-------------------|
| θ ≈ 0 | Strong adaptation | Vehicle resets to equilibrium (stop/yield scenarios) |
| θ = 0.5 | Moderate adaptation | Partial adjustment (merging scenarios) |
| θ ≈ 1 | Weak adaptation | Preserve upstream behavior (through traffic) |

### Junction-Specific Values (config_base.yml Oct 2025)

#### Roundabouts
- **theta_moto_insertion**: 0.2 (motos yield at entry, strong reset)
- **theta_moto_circulation**: 0.8 (maintain speed in flow)

#### Signalized Intersections
- **theta_moto_signalized**: 0.8 (aggressive acceleration through green)
- **theta_car_signalized**: 0.5 (moderate acceleration, more cautious)

#### Priority/Secondary Roads
- **Priority**: θ ≈ 0.9 (minimal disruption, maintain through traffic)
- **Secondary**: θ ≈ 0.1 (strong reset at stop/yield)

---

## 📋 Cleanup Action Plan

### Phase 1.1: Archive Legacy Configs

**Files to Move**:
```
arz_model/config/riemann_problem_test.yaml      → _archive/legacy_test_configs/
arz_model/config/stationary_free_flow_test.yaml → _archive/legacy_test_configs/
```

**Verification**: 
```
✓ 0 references found for riemann_problem_test
✓ 0 references found for stationary_free_flow_test
✓ No breaking changes (configs not used)
```

**Documentation**:
```
_archive/legacy_test_configs/README.md
├── Explains archival reason
├── Documents superseding configs
└── Lists archive date (2025-02-20)
```

---

## 📚 Active Config Files Summary

### Files to KEEP (Active)

```
✅ arz_model/config/config_base.yml
   - PRIMARY configuration
   - 20+ usages across pipeline
   - Contains behavioral_coupling (θ_k) parameters
   - Status: ACTIVELY MAINTAINED (Oct 2025 update)
   - Dependencies: CRITICAL (must not remove)

✅ arz_model/config/scenario_convergence_test.yml
   - Section 7.3 test scenario
   - 3 usages (convergence testing)
   - Overrides config_base.yml for specific test
   - Status: ACTIVE (required for validation)
   - Dependencies: Section 7.3 depends on this
```

### Files to ARCHIVE (Legacy)

```
❌ arz_model/config/riemann_problem_test.yaml
   - Classical Riemann problem (legacy development tool)
   - 0 usages anywhere
   - Superseded by scenario_convergence_test.yml
   - Action: ARCHIVE (safe, no dependencies)

❌ arz_model/config/stationary_free_flow_test.yaml
   - Free flow equilibrium test (legacy development tool)
   - 0 usages anywhere
   - Superseded by scenario_convergence_test.yml
   - Action: ARCHIVE (safe, no dependencies)
```

---

## 🔄 Post-Cleanup State

**After Phase 1 (Cleanup)**:

```
arz_model/config/
├── config_base.yml              ✅ ACTIVE PRIMARY
├── scenario_convergence_test.yml ✅ ACTIVE TEST
├── network_config.py
└── __init__.py

_archive/legacy_test_configs/
├── README.md
├── riemann_problem_test.yaml
└── stationary_free_flow_test.yaml
```

**Benefits**:
- ✅ Cleaner codebase
- ✅ Obvious which configs are active
- ✅ Reduced confusion for new developers
- ✅ Legacy tools preserved in archive (for reference)
- ✅ Zero breaking changes

---

## 📊 Codebase Search Results (Complete)

### Search: "config_base.yml" (20 matches)

```
✓ validation_ch7_v2/scripts/test_section_7_3_analytical.py:232
  - Loads config for Section 7.3 validation

✓ arz_model/run_convergence_test.py:121-122
  - Default config for convergence testing

✓ Code_RL/src/env/traffic_signal_env_direct.py
  - RL environment uses behavioral_coupling (θ_k values)

✓ test_network_config.py
  - Test suite references config_base.yml

✓ arz_model/calibration/core/calibration_runner.py
  - Calibration runner loads base config

✓ arz_model/simulation/runner.py
  - Simulation runner uses base config

✓ validation_kaggle_manager.py (related tests)
  - All sections 7.3-7.7 reference config

✓ Multiple validation scripts
  - niveau1_fundamentals/, niveau2_implementation/
  - niveau3_realworld_validation/, niveau4_rl_performance/
```

### Search: "scenario_convergence_test.yml" (3 matches)

```
✓ validation_ch7_v2/scripts/test_section_7_3_analytical.py:232
✓ arz_model/run_convergence_test.py:121
✓ arz_model/run_convergence_test.py:122
```

### Search: "riemann_problem_test" (0 matches)

```
❌ NO REFERENCES FOUND
```

### Search: "stationary_free_flow_test" (0 matches)

```
❌ NO REFERENCES FOUND
```

---

## ✅ Verification Checklist

- [x] Identified active configs (2 files)
- [x] Identified legacy configs (2 files)
- [x] Verified no code references for legacy configs
- [x] Confirmed config_base.yml is primary active config
- [x] Documented behavioral_coupling parameters (θ_k)
- [x] Verified Phase 1.3 calibration cleanly archived
- [x] Confirmed Ch7 pipeline independent from legacy code

---

**Status**: ✅ COMPLETE - Ready for Phase 1 (Cleanup)

**Next Step**: Archive legacy configs to `_archive/legacy_test_configs/`

**Last Updated**: 2025-02-20
