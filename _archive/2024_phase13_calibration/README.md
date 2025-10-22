# Archive: 2024 Phase 1.3 Calibration

**Created**: October 22, 2025  
**Reason**: Cleanup for Ch7 Validation Architecture Transition  
**Status**: LEGACY - Not for Production Use  

---

## Overview

This archive contains all components from the **2024 Phase 1.3 Calibration** phase that are no longer compatible with the **new Ch7 Validation Architecture (2025+)**.

These modules were part of the ARZ digital twin calibration framework that:
- Calibrated ARZ parameters using TomTom real traffic data
- Validated performance on Victoria Island corridor
- Implemented 2-phase optimization (parameters → road quality R(x))

All objectives for this phase have been **completed successfully** (October 2024 - September 2025), and the optimized model is now frozen for Ch7 validation work.

---

## Contents

### Code Modules (9 archived)

| File | Purpose | Why Archived |
|------|---------|--------------|
| `digital_twin_calibrator.py` | 2-phase ARZ calibration with TomTom data | Phase complete - Model frozen |
| `spatiotemporal_validator.py` | Spatio-temporal validation (GEH/MAPE/RMSE) | Metrics specific to road corridors, not for Ch7 |
| `tomtom_collector.py` | TomTom data collection API | Data already collected |
| `speed_processor.py` | Real traffic speed processing | Victoria Island specific |
| `group_manager.py` | Road segment grouping | Corridor-specific implementation |
| `victoria_island_config.py` | Configuration for Victoria Island | Not generalizable |
| `calibration_results_manager.py` | Result persistence and analysis | Legacy data management |
| `test_real_data_loader.py` | Unit tests for data loading | Legacy test suite |
| `corridor_loader.py` | Corridor loading (never completed) | TODO item - code never used |

### Assets

| Item | Purpose |
|------|---------|
| `groups_reference/` | Victoria Island network configuration (75 segments) |
| `results_archived/` | Historical calibration results from Phase 1.3 |

---

## What Happened to These Components

### ✅ Complete - No Longer Needed
- ARZ model parameters α, Vmax, τ, etc. successfully calibrated
- Road quality function R(x) optimized for 75 segments
- Victoria Island corridor validation achieved target metrics:
  - GEH < 5 (traffic flow)
  - MAPE < 15% (speed prediction)
  - RMSE < 10 km/h (speed accuracy)
- Model frozen for Ch7 validation (no more calibration)

### ❌ Not Compatible with Ch7
- **Different Data**: Ch7 uses abstract scenarios, not real TomTom data
- **Different Metrics**: Ch7 validates physics equations, not traffic flow
- **Different Scope**: Ch7 is generic, these modules are Victoria Island specific
- **Different Architecture**: Ch7 uses multi-kernel GPU, these are sequential

### 🔄 Transitioned To
New Ch7 architecture handles:
- Generic scenario-based validation
- Vehicle dynamics equations (Level 1, 2, 3)
- Multi-kernel GPU parallelization
- Distributed caching with versioning
- Framework-independent metrics

---

## Important Notes

### ⚠️ What Was NOT Archived

The following component **remains in active use**:

- **`real_data_loader.py`** (parent: `arz_model/calibration/data/`)
  - Used by: `test_section_7_4_calibration.py`
  - Purpose: Real-time calibration validation with Victoria Island data
  - Status: ✅ Active in Ch7 validation pipeline (Section 7.4)

### 📝 Historical Data

Original calibration results preserved in:
- `results_archived/test_group_calibration_20250908_124713.json`
- `results_archived/victoria_island_corridor_calibration_20250908_162605.json`

These document the final calibrated state of the ARZ model for Victoria Island.

---

## How to Reference

If you need to understand:

1. **How Phase 1.3 Calibration Worked**
   - See: `.audit/CALIBRATION_DATA_OBSOLESCENCE_AUDIT.md`
   - See: Original source files in this archive

2. **Current Architecture**
   - See: `validation_ch7_v2/scripts/niveau3_realworld_validation/README.md`
   - See: Ch7 validation documentation

3. **Migration Path**
   - Ch7 validation maintains Victoria Island validation capability
   - Uses abstracted vehicle dynamics from calibrated parameters
   - No direct data dependency on Phase 1.3 outputs

---

## Restore Instructions

**If Phase 1.3 components are needed in the future:**

```bash
# Option 1: Reference a specific archived module
git show HEAD^:_archive/2024_phase13_calibration/digital_twin_calibrator.py

# Option 2: Restore entire archive to staging
mkdir temp_restore
cp -r _archive/2024_phase13_calibration/* temp_restore/

# Option 3: Check git history
git log --follow -- arz_model/calibration/data/digital_twin_calibrator.py
```

---

## Git History

The archived modules remain in git history:

```bash
# See the last commit before archiving
git log --oneline | grep -i "archive\|cleanup" | head -1

# Restore a specific file from history
git checkout HEAD~1 -- arz_model/calibration/data/digital_twin_calibrator.py

# View full history of a module
git log -p -- _archive/2024_phase13_calibration/digital_twin_calibrator.py
```

---

## Timeline

- **March 2024**: Phase 1.3 Calibration started
- **September 2024**: Phase 1.3 Calibration completed
- **October 2024 - September 2025**: Calibration refinement and validation
- **October 22, 2025**: Archival for Ch7 Validation Transition

---

## Contact / Questions

For questions about:
- **Phase 1.3 Calibration**: See archived documentation and git history
- **Ch7 Validation**: See `validation_ch7_v2/` and updated README files
- **Architecture Transition**: See `.audit/` directory

---

**Status**: ARCHIVED FOR HISTORICAL REFERENCE  
**Recommendation**: Do not use in new development - reference Ch7 architecture instead
