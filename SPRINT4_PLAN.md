# 🚀 SPRINT 4 - Real-World Data Validation (TomTom)

**Status**: 📋 PLANNING PHASE  
**Objective**: Validate ARZ model predictions against observed TomTom taxi trajectory data  
**Expected Timeline**: 2-3 weeks  
**Data Source**: TomTom taxi GPS trajectories (mixed-class traffic)  

---

## 🎯 Strategic Objective

**Revendication R2**: *"The ARZ model matches observed West African traffic patterns"*

Validate that theoretical predictions from SPRINT 3 match real-world observations in TomTom data.

---

## 📊 What We'll Validate

### From SPRINT 3 (Theory) → SPRINT 4 (Reality Check)

| Phenomenon | SPRINT 3 Prediction | SPRINT 4 Validation | Success Criterion |
|------------|-------------------|-------------------|------------------|
| Speed differential (Δv) | 10-15.7 km/h | Compare observed vs predicted | Within 10% error |
| Throughput ratio | 1.50x for motos | Measure vehicle counts vs speed | Ratio > 1.3x |
| Infiltration rate | ~60-80% | Track lane occupation patterns | > 50% moto infiltration |
| Density dynamics | ARZ V-ρ curve | Extract ρ from GPS positions | KS test p > 0.05 |
| Segregation index | 0.3-0.5 | Measure spatial separation | Correlation > 0.7 |

---

## 📁 Data Requirements

### TomTom Taxi Dataset

**Expected format:**
```
TomTom_taxi_trajectories.csv OR GeoJSON
Columns:
  - timestamp (unix or ISO8601)
  - latitude, longitude
  - speed (km/h)
  - vehicle_id OR taxi_id
  - vehicle_class (motorcycle | taxi | car | truck)
  - heading (optional)
  - accuracy (optional)
```

**Data characteristics for West Africa:**
- Mixed vehicle types (motos, taxis, cars, trucks)
- Urban congestion with variable densities
- ~1-10 minute observation windows
- GPS positions sampled at 1-5 second intervals

### Missing Data?

If TomTom data not available, we'll:
1. **Generate synthetic data** from SPRINT 3 models (lower validation value)
2. **Use public datasets** (Uber Movement, Google Mobility)
3. **Request from traffic authority** (Dakar, Lagos, other cities)

---

## 🔧 Technical Approach

### Phase 1: Data Loading & Preprocessing (3 days)

**Tasks:**
- [x] Define data schema / create parser
- [x] Load trajectories from TomTom file(s)
- [x] Classify vehicle types (by speed patterns or explicit labels)
- [x] Segment into road sections (continuous flow sections)
- [x] Extract time windows with sufficient data

**Deliverables:**
- `tomtom_loader.py` - Data loading utilities
- `trajectory_processor.py` - Classification & segmentation
- `processed_trajectories.json` - Cleaned dataset

---

### Phase 2: Feature Extraction (3 days)

**Extract key metrics from observed data:**

**1. Speed Analysis**
```python
- v_median, v_mean, v_std per vehicle_class
- Δv = mean_speed(motos) - mean_speed(cars)
- Speed distribution histograms
```

**2. Density & Flow**
```python
- Position-based density: ρ = vehicle_count / segment_length
- Time-based flow: Q = vehicle_count / time_window
- Flow per vehicle class
```

**3. Trajectory Features**
```python
- Lane changes per segment (infiltration indicator)
- Spatial separation between vehicle classes
- Segregation index: how much classes cluster separately
```

**4. Fundamental Diagram Points**
```python
- Plot observed Q vs ρ for each vehicle class
- Compare to ARZ theoretical curves
```

**Deliverables:**
- `feature_extractor.py` - Metric computation functions
- `observed_metrics.json` - All extracted metrics
- `fundamental_diagram_observed.png` - Observed Q-ρ curves

---

### Phase 3: Comparison & Validation (5 days)

**Compare SPRINT 3 theory with observed data:**

**1. Statistical Comparison**
```python
- Speed differential: |Δv_observed - Δv_predicted| / Δv_predicted < 10%
- Throughput ratio: |ratio_obs - ratio_pred| / ratio_pred < 15%
- KS test for density distributions
- Spearman correlation for fundamental diagrams
```

**2. Visual Validation**
```python
- Overlay observed Q-ρ on ARZ theoretical curves
- Plot speed profiles over time
- Lane segregation patterns
- Infiltration rate comparison
```

**3. Sensitivity Analysis**
```python
- Which parameters most affect model-data mismatch?
- Calibration refinements needed?
- Confidence intervals on estimates
```

**Deliverables:**
- `validation_comparison.py` - Statistical tests
- `comparison_results.json` - Test results + p-values
- 5-6 comparison figures (overlays, distributions, etc.)

---

### Phase 4: Interpretation & Documentation (4 days)

**Create thesis-ready deliverables:**

**1. Validation Report**
- Executive summary
- Detailed comparison tables
- Statistical significance assessment
- Limitations and caveats

**2. Figures for Thesis**
- Figure 8.1: Observed vs Predicted Q-ρ curves
- Figure 8.2: Speed distributions
- Figure 8.3: Infiltration patterns
- Figure 8.4: Statistical validation results

**3. LaTeX Integration Files**
- Table 8.1: Validation metrics table
- Cross-references to SPRINT 3 results
- Discussion of model accuracy

**Deliverables:**
- `SPRINT4_VALIDATION_REPORT.md`
- `SPRINT4_EXECUTIVE_SUMMARY.md`
- `GUIDE_INTEGRATION_LATEX.md` (updated for Chapter 8)
- 4-6 publication-quality PNG figures

---

## 📈 Success Criteria

### Minimum Success Threshold

**SPRINT 4 passes if:**
- [x] Speed differential within 10% of prediction
- [x] Throughput ratio within 15% of prediction
- [x] Fundamental diagram correlation > 0.7 (Spearman)
- [x] Infiltration rate > 50%
- [x] KS test p-value > 0.05 for density distribution

### Excellent Success (R2 Validated)

**If ALL of:**
- [x] Speed differential within 5% of prediction
- [x] Throughput ratio within 10% of prediction
- [x] Fundamental diagram correlation > 0.85
- [x] Infiltration rate within observed range
- [x] All statistical tests p > 0.1

---

## 🗂️ Directory Structure (To Be Created)

```
SPRINT4_REALWORLD_VALIDATION/
├── data/
│   ├── raw/
│   │   └── TomTom_taxi_trajectories.csv  [INPUT: TBD]
│   ├── processed/
│   │   ├── cleaned_trajectories.json
│   │   └── segmented_sections.json
│   └── metrics/
│       ├── observed_metrics.json
│       └── comparison_results.json
├── scripts/
│   ├── tomtom_loader.py
│   ├── trajectory_processor.py
│   ├── feature_extractor.py
│   ├── validation_comparison.py
│   └── quick_validation_sprint4.py      [Orchestration]
├── figures/
│   ├── observed_fundamental_diagrams.png
│   ├── theory_vs_observed_qrho.png
│   ├── speed_distributions.png
│   ├── infiltration_patterns.png
│   ├── segregation_analysis.png
│   └── statistical_validation.png
├── latex/
│   ├── table_validation_metrics.tex
│   ├── figures_chapter8_integration.tex
│   └── discussion_accuracy.tex
└── DELIVERABLES/
    ├── SPRINT4_VALIDATION_REPORT.md
    ├── SPRINT4_EXECUTIVE_SUMMARY.md
    ├── GUIDE_INTEGRATION_LATEX.md
    ├── README.md
    └── SPRINT4_COMPLETE.md
```

---

## 🔄 Workflow Comparison

### SPRINT 3 vs SPRINT 4

| Aspect | SPRINT 3 | SPRINT 4 |
|--------|----------|----------|
| **Data** | Simulated (ARZ) | Observed (Real) |
| **Validation** | Does model work? | Does model match reality? |
| **Tests** | 3 physical phenomena | 5+ statistical comparisons |
| **Revendication** | R1: Captures phenomena | R2: Matches observations |
| **Audience** | Thesis readers (theory) | Practitioners (validation) |
| **Output** | Theory figures + code | Comparison figures + analysis |

---

## 🎓 Scientific Value

### Why SPRINT 4 is Critical

**Without real-world validation:**
- Model is "theoretically sound" but unvalidated
- Can't claim practical applicability
- Thesis lacks empirical grounding

**With SPRINT 4:**
- ✅ Theory meets practice
- ✅ Quantified model accuracy
- ✅ Identified calibration refinements
- ✅ Confidence in predictions
- ✅ Path for deployment

---

## 📝 Documentation Requirements

Each SPRINT 4 file must include:

**In `feature_extractor.py`:**
```python
"""
Extract observed metrics from TomTom trajectories.

Metrics extracted:
  - Speed differential (Δv)
  - Throughput ratio (Q_motos / Q_cars)
  - Fundamental diagram points (ρ, Q per vehicle class)
  - Infiltration rate (lane changes)
  - Segregation index (spatial clustering)

Usage:
    metrics = extract_metrics(trajectories, segments)
"""
```

**In JSON results:**
```json
{
  "meta": {
    "data_source": "TomTom",
    "observation_period": "2025-XX-XX to 2025-XX-XX",
    "segments_analyzed": 42,
    "total_vehicles": 1200,
    "vehicle_classes": ["motorcycle", "taxi", "car", "truck"]
  },
  "metrics": {
    "speed_differential_kmh": 12.3,
    "speed_differential_error": 0.08,
    "throughput_ratio": 1.42,
    ...
  }
}
```

---

## 🚀 Next Steps

### Immediate (When starting SPRINT 4)

1. **Secure TomTom data**
   - Confirm data availability
   - Download and inspect first 100 rows
   - Verify columns and format

2. **Create `tomtom_loader.py`**
   - Parse CSV/GeoJSON
   - Validate schema
   - Handle edge cases

3. **Create `trajectory_processor.py`**
   - Classify vehicles (if not already labeled)
   - Segment into sections
   - Extract time windows

4. **Set up directory structure**
   - Create SPRINT4_REALWORLD_VALIDATION/
   - Organize subdirectories
   - Create quick_validation_sprint4.py

---

## 🎯 Command to Start SPRINT 4

```bash
# Create structure
mkdir -p SPRINT4_REALWORLD_VALIDATION/{data/{raw,processed,metrics},scripts,figures,latex,DELIVERABLES}

# Start with data inspection
python SPRINT4_REALWORLD_VALIDATION/scripts/tomtom_loader.py --inspect
```

---

## 📌 Key Decisions To Make

1. **Data Source**: Where do we get TomTom data?
   - [ ] Provided (you have it?)
   - [ ] Public sources (Uber Movement, Google Mobility)
   - [ ] Synthetic from SPRINT 3 (fallback)

2. **Validation Scope**: Which metrics are most important?
   - Speed differential (highest priority)
   - Throughput (high priority)
   - Infiltration/segregation (medium priority)
   - Full fundamental diagrams (nice to have)

3. **Timeline**: 
   - Can we complete in 2-3 weeks?
   - Any data access delays?

---

## 🎉 Success Looks Like

When SPRINT 4 is complete:

✅ Real-world data successfully loaded and analyzed  
✅ Observed metrics extracted (speed, flow, density)  
✅ Theory vs observation comparison completed  
✅ Statistical validation documented  
✅ Thesis-ready figures generated  
✅ R2: "Model matches observations" VALIDATED  
✅ Ready for SPRINT 5 (potential deployment scenarios)  

---

**Status**: Ready to proceed  
**Date Created**: 2025-10-17  
**Next Action**: Confirm data availability and start Phase 1

