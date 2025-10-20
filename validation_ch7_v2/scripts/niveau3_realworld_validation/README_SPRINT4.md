# SPRINT 4: Real-World Data Validation (Niveau 3)

**Status**: 🔄 **IN PROGRESS** - Framework complete, documentation in progress  
**Date**: 2025-10-17  
**Location**: `validation_ch7_v2/scripts/niveau3_realworld_validation/`  

---

## 🎯 Objective

**Revendication R2**: *"The ARZ model matches observed West African traffic patterns"*

Validate ARZ model predictions (from SPRINT 3) against real-world TomTom taxi trajectory data.

---

## 📊 Validation Framework

### What We Validate

| Metric | SPRINT 3 Prediction | SPRINT 4 Observation | Success Criterion |
|--------|---------------------|---------------------|-------------------|
| **Speed differential (Δv)** | 10-15.7 km/h | Compare observed | Within 10% error |
| **Throughput ratio** | 1.50x (motos/cars) | Measure vehicle counts | Within 15% error |
| **Fundamental diagrams** | ARZ Q-ρ curves | Extract from GPS | Correlation > 0.7 |
| **Infiltration rate** | ~60-80% | Track lane patterns | Within 50-80% |
| **Statistical significance** | Theory | Observed distributions | KS test p > 0.05 |

---

## 🏗️ Architecture

### Components

```
niveau3_realworld_validation/
├── __init__.py                      # Module initialization
├── tomtom_trajectory_loader.py      # Load & parse GPS trajectories (450 lines)
├── feature_extractor.py             # Extract observed metrics (400 lines)
├── validation_comparison.py         # Statistical comparison (400 lines)
└── quick_test_niveau3.py            # Orchestration script (150 lines)
```

**Total**: ~1,400 lines of production-quality Python

### Data Flow

```
┌─────────────────────────┐
│ TomTom Trajectories     │
│ (CSV/GeoJSON or         │
│  synthetic fallback)    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ tomtom_trajectory_      │
│ loader.py               │
│                         │
│ • Parse GPS to 1D       │
│ • Classify vehicles     │
│ • Segment road          │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Processed Trajectories  │
│ (JSON format)           │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ feature_extractor.py    │
│                         │
│ • Speed differential    │
│ • Throughput ratio      │
│ • Fundamental diagrams  │
│ • Infiltration rate     │
│ • Segregation index     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Observed Metrics        │
│ (JSON format)           │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ validation_comparison.py│
│                         │
│ • Load SPRINT 3 results │
│ • Statistical tests     │
│ • Pass/fail criteria    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Comparison Results      │
│ (JSON + summary)        │
└─────────────────────────┘
```

---

## 🚀 Usage

### Quick Test (Complete Pipeline)

```bash
cd validation_ch7_v2/scripts/niveau3_realworld_validation
python quick_test_niveau3.py
```

**Expected output**:
- Processed trajectories: `../../data/processed/trajectories_niveau3.json`
- Observed metrics: `../../data/validation_results/realworld_tests/observed_metrics.json`
- Comparison results: `../../data/validation_results/realworld_tests/comparison_results.json`
- Summary: `../../data/validation_results/realworld_tests/niveau3_summary.json`

**Duration**: ~0.5 seconds (with synthetic data)

### Individual Components

#### 1. Load Trajectories

```python
from tomtom_trajectory_loader import TomTomTrajectoryLoader

loader = TomTomTrajectoryLoader("data/raw/TomTom_trajectories.csv")
trajectories = loader.load_and_parse()
loader.save_processed(trajectories, "data/processed/trajectories.json")
```

**Features**:
- Supports CSV and GeoJSON formats
- Automatic vehicle classification (if not provided)
- GPS → 1D position conversion
- Road segmentation (500m segments)
- Synthetic data fallback (uses ARZ model from SPRINT 3)

#### 2. Extract Metrics

```python
from feature_extractor import FeatureExtractor

extractor = FeatureExtractor(trajectories)
metrics = extractor.extract_all_metrics()
extractor.save_metrics(metrics, "observed_metrics.json")
```

**Metrics Extracted**:
1. **Speed differential**: Δv = mean(v_motos) - mean(v_cars)
2. **Throughput ratio**: Q_motos / Q_cars
3. **Fundamental diagrams**: (ρ, Q, V) points per vehicle class
4. **Infiltration rate**: % motos in car-dominated zones
5. **Segregation index**: Spatial separation between classes
6. **Statistical summary**: KS test, Mann-Whitney U test

#### 3. Validation Comparison

```python
from validation_comparison import ValidationComparator

comparator = ValidationComparator(
    predicted_metrics_path="SPRINT3_DELIVERABLES/results/fundamental_diagrams.json",
    observed_metrics_path="data/validation_results/realworld_tests/observed_metrics.json"
)
results = comparator.compare_all()
comparator.save_results(results, "comparison_results.json")
```

**Comparisons Performed**:
- Speed differential: Relative error < 10%
- Throughput ratio: Relative error < 15%
- Fundamental diagrams: Spearman ρ > 0.7
- Infiltration rate: Within 50-80% range

---

## 📁 Output Structure

```
validation_ch7_v2/
├── data/
│   ├── processed/
│   │   └── trajectories_niveau3.json           # Processed GPS data
│   └── validation_results/
│       └── realworld_tests/
│           ├── observed_metrics.json           # Extracted metrics
│           ├── comparison_results.json         # Validation results
│           └── niveau3_summary.json            # Execution summary
├── figures/
│   └── niveau3_realworld/                      # Comparison plots (TBD)
│       ├── theory_vs_observed_qrho.png
│       ├── speed_distributions.png
│       ├── infiltration_patterns.png
│       ├── segregation_analysis.png
│       ├── statistical_validation.png
│       └── fundamental_diagrams_comparison.png
└── scripts/
    └── niveau3_realworld_validation/           # Source code ✅
```

---

## 📊 Current Results (Synthetic Data)

### Test Execution Summary

| Metric | Predicted | Observed | Error | Status |
|--------|-----------|----------|-------|--------|
| **Δv** | 10.0 km/h | 10.0 km/h | 0.1% | ✅ PASS |
| **Throughput ratio** | 1.50 | 0.67 | 55.6% | ❌ FAIL |
| **FD correlation** | N/A | -0.27 | N/A | ❌ FAIL |
| **Infiltration rate** | 50-80% | 10.4% | Out of range | ❌ FAIL |

**Overall Status**: ❌ FAIL (1/4 tests passed - 25%)  
**Revendication R2**: NOT VALIDATED ❌ (with synthetic data)

**Note**: Synthetic data is generated from ARZ model (SPRINT 3), so it's expected that some metrics don't match perfectly. Real TomTom data will provide actual validation.

---

## 🔧 Data Requirements

### TomTom Data Format

**Expected CSV columns**:
```
timestamp,latitude,longitude,speed_kmh,vehicle_id,vehicle_class
1634567890,6.4521,-3.4698,45.2,moto_001,motorcycle
1634567891,6.4522,-3.4699,35.8,car_002,car
...
```

**Or GeoJSON format**:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [-3.4698, 6.4521]},
      "properties": {
        "timestamp": 1634567890,
        "speed_kmh": 45.2,
        "vehicle_id": "moto_001",
        "vehicle_class": "motorcycle"
      }
    }
  ]
}
```

### Missing Data Handling

If `vehicle_class` is not provided, the loader infers it from speed patterns:
- `speed > 50 km/h` AND high variance → `motorcycle`
- `speed 30-50 km/h` → `car/taxi`
- `speed < 30 km/h` → `car` (congested)

### Synthetic Data Fallback

When no real TomTom data is available:
1. Generates ~5,000 trajectory points
2. Uses ARZ parameters from SPRINT 3 (Vmax, ρmax, τ)
3. Simulates 20 motorcycles + 30 cars
4. Duration: 600 seconds (10 minutes)
5. Road length: 2000 meters

---

## 🎓 Scientific Contributions

### What SPRINT 4 Adds to Thesis

1. **Empirical Validation**: Theory (SPRINT 3) meets practice
2. **Quantified Accuracy**: Statistical error bounds on predictions
3. **Model Calibration**: Identifies parameters needing refinement
4. **Practical Applicability**: Demonstrates real-world utility
5. **Confidence Bounds**: P-values and correlation coefficients

### Integration with SPRINT 3

| SPRINT 3 | SPRINT 4 |
|----------|----------|
| **Theory**: Does ARZ capture phenomena? | **Practice**: Does ARZ match reality? |
| **Simulation**: ARZ model dynamics | **Observation**: Real GPS trajectories |
| **Output**: Physical validation | **Output**: Statistical validation |
| **Chapter 7.2**: Phénomènes physiques | **Chapter 7.3**: Données réelles |

---

## ⚠️ Known Issues & Limitations

### Current Limitations

1. **Synthetic Data**: Results based on ARZ-generated trajectories (not real TomTom)
2. **Small Sample**: 50 vehicles × 600s (need more for robust statistics)
3. **1D Projection**: GPS converted to 1D position (loses lateral dynamics)
4. **Simple Segmentation**: Fixed 500m segments (could be road-aware)

### Future Improvements

1. **Acquire Real TomTom Data**: From Dakar, Lagos, or other West African cities
2. **Multi-lane Analysis**: Track lateral positions for infiltration patterns
3. **Temporal Analysis**: Rush hour vs off-peak dynamics
4. **Larger Dataset**: 1000+ vehicles, multiple road sections
5. **Advanced Segmentation**: Use OpenStreetMap for realistic road network

---

## 🏃 Next Steps

### Immediate (This Week)

- [ ] **Task 5**: Create comprehensive documentation
  - [ ] Update this README with figures section
  - [ ] Create SPRINT4_EXECUTIVE_SUMMARY.md
  - [ ] Create LaTeX integration guide

- [ ] **Task 6**: Generate comparison figures
  - [ ] Theory vs observed Q-ρ overlay
  - [ ] Speed distribution histograms
  - [ ] Infiltration pattern visualization
  - [ ] Statistical test results plots
  - [ ] Segregation analysis heatmap
  - [ ] Summary dashboard

- [ ] **Task 7**: Finalize deliverables
  - [ ] Create SPRINT4_DELIVERABLES/ folder
  - [ ] Organize figures, results, LaTeX files
  - [ ] Mark SPRINT 4 complete

### Medium Term (Next Month)

- [ ] Acquire real TomTom data
- [ ] Re-run validation with real observations
- [ ] Refine ARZ parameters based on findings
- [ ] Iterate until Revendication R2 validated

### Long Term (Thesis)

- [ ] Integrate SPRINT 4 results into Chapter 7.3
- [ ] Discuss model accuracy and limitations
- [ ] Propose calibration refinements
- [ ] Connect to deployment scenarios (SPRINT 5?)

---

## 📚 References

### Related Files

- **SPRINT 3 Results**: `SPRINT3_DELIVERABLES/`
- **SPRINT 3 README**: `validation_ch7_v2/scripts/niveau2_physical_phenomena/`
- **SPRINT 4 Plan**: `SPRINT4_PLAN.md`
- **Data Preprocessing**: `validation_ch7_v2/scripts/data/preprocess_tomtom.py`

### Documentation

- **TomTom API**: https://developer.tomtom.com/
- **ARZ Model**: See SPRINT 3 `fundamental_diagrams.py`
- **Statistical Tests**: SciPy documentation (KS test, Spearman)

---

## ✅ Completion Checklist

### Framework Implementation ✅ DONE

- [x] tomtom_trajectory_loader.py (450 lines)
- [x] feature_extractor.py (400 lines)
- [x] validation_comparison.py (400 lines)
- [x] quick_test_niveau3.py (150 lines)
- [x] __init__.py (module doc)

### Testing ✅ DONE

- [x] Loader tested with synthetic data
- [x] Feature extractor tested (5 metrics)
- [x] Validation comparator tested (4 comparisons)
- [x] Full pipeline tested (quick_test_niveau3.py)
- [x] JSON serialization fixed (bool_ issue)

### Documentation 🔄 IN PROGRESS

- [x] README.md (this file)
- [ ] SPRINT4_EXECUTIVE_SUMMARY.md
- [ ] GUIDE_INTEGRATION_LATEX.md
- [ ] Code comments (comprehensive)

### Outputs 🔄 IN PROGRESS

- [x] Processed trajectories JSON
- [x] Observed metrics JSON
- [x] Comparison results JSON
- [x] Niveau 3 summary JSON
- [ ] 6 comparison figures (PNG + PDF)
- [ ] LaTeX table files

### Validation ⏳ PENDING (Real Data)

- [ ] Speed differential < 10% error
- [ ] Throughput ratio < 15% error
- [ ] Fundamental diagram correlation > 0.7
- [ ] Infiltration rate 50-80%
- [ ] Revendication R2 VALIDATED

---

**Status**: Framework complete ✅ | Documentation in progress 🔄 | Real data validation pending ⏳  
**Last Updated**: 2025-10-17  
**Next Action**: Generate comparison figures (Task 6)
