# Legacy Test Configs Archive

**Archived Date**: 2025-02-20  
**Reason**: Configs not used by Ch7 validation pipeline (zero references in codebase)  
**Status**: Safe to remove - no breaking changes

---

## Archived Files

### 1. riemann_problem_test.yaml
- **Purpose**: Classical Riemann problem benchmark for hyperbolic solver testing
- **Usage**: 0 matches in entire codebase
- **Superseded By**: `scenario_convergence_test.yml` (Section 7.3 analytical validation)
- **Why Archived**: Development/debugging tool from Phase 1.3 calibration, no longer needed for Ch7 validation
- **Location**: `../_archive/legacy_test_configs/riemann_problem_test.yaml`

### 2. stationary_free_flow_test.yaml
- **Purpose**: Free flow equilibrium state validation test
- **Usage**: 0 matches in entire codebase
- **Superseded By**: `scenario_convergence_test.yml` (Section 7.3 analytical validation)
- **Why Archived**: Development/debugging tool from Phase 1.3 calibration, equilibrium testing now covered by convergence analysis
- **Location**: `../_archive/legacy_test_configs/stationary_free_flow_test.yaml`

---

## Active Configs (Still in Use)

These configs remain in `arz_model/config/` because they are actively used:

```
✅ config_base.yml (PRIMARY)
   - 20+ usages across validation, RL, simulation pipelines
   - Contains behavioral_coupling (θ_k) parameters
   - Status: ACTIVELY MAINTAINED (Oct 2025)

✅ scenario_convergence_test.yml (ACTIVE TEST)
   - 3 usages in Section 7.3 validation
   - Required for numerical scheme convergence testing
   - Status: ACTIVELY USED
```

---

## Verification

**Codebase Search Results**:
```
grep -r "riemann_problem_test"      → 0 matches
grep -r "stationary_free_flow_test" → 0 matches
```

**Conclusion**: Safe to archive. No code references these configs.

---

## Recovery

If needed to restore these configs:

1. Copy from `_archive/legacy_test_configs/` back to `arz_model/config/`
2. Git history is preserved - can use `git checkout` if necessary

---

**Archive Created**: 2025-02-20  
**Cleanup Status**: ✅ COMPLETE  
**Breaking Changes**: ❌ NONE  
