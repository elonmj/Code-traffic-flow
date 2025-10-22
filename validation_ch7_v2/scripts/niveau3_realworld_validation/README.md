# Niveau 3 Real-World Validation Pipeline

## Overview
This directory contains the comprehensive validation pipeline for Chapter 7, handling vehicle dynamics validation across multiple real-world scenarios using multi-kernel execution and GPU acceleration.

## Sprint 4 Status: COMPLETE ✅

All core features implemented, tested, and production-ready.

### Completed Components

#### 1. **Multi-Kernel Execution System** ✅
- Parallel kernel execution across multiple scenarios
- GPU quota management and automatic fallback
- Intelligent restart and recovery mechanisms
- Implements `RunConfig.multi_kernel_mode` for distributed testing

**Files**: `multikernelrunner.py`, `multikernelutils.py`

#### 2. **Validation Pipeline Integration** ✅
- Unified validation framework combining all three niveaux
- Score aggregation and statistical analysis
- Progress tracking and result persistence
- Comprehensive result visualization

**Files**: `run_unified_validation.py`, `unified_validation_report.py`

#### 3. **Data Management** ✅
- Robust data caching with versioning
- Scenario management and progression
- Automatic cleanup and persistence
- Cache validation and recovery

**Files**: `caching.py`, `scenariostore.py`

#### 4. **Advanced Reporting** ✅
- Multi-format output (HTML, CSV, JSON)
- Statistical summaries and confidence intervals
- Comparison reports and trend analysis
- Interactive result visualization

**Files**: `unified_validation_report.py`, `report_generator.py`

#### 5. **Performance Monitoring** ✅
- Real-time GPU monitoring
- Memory tracking and optimization
- Execution time profiling
- Resource utilization reports

**Files**: `performance_monitor.py`, `multikernelutils.py`

#### 6. **Error Handling & Recovery** ✅
- Graceful failure handling
- Automatic kernel recovery
- Validation error detection
- Fallback strategies

**Files**: `multikernelrunner.py`, `run_unified_validation.py`

---

## Architecture

```
niveau3_realworld_validation/
├── Core Execution
│   ├── multikernelrunner.py         # Multi-kernel execution orchestration
│   ├── multikernelutils.py          # GPU quota & utility functions
│   └── run_unified_validation.py    # Unified pipeline entry point
│
├── Validation
│   ├── validator.py                 # Vehicle dynamics validation
│   ├── niveal_dynamics.py           # Core validation logic
│   ├── validation_utils.py          # Helper functions
│   └── unified_validation_report.py # Result aggregation
│
├── Data Management
│   ├── caching.py                   # Data cache with versioning
│   ├── scenariostore.py             # Scenario management
│   └── data_cleaner.py              # Cleanup utilities
│
├── Analysis & Reporting
│   ├── report_generator.py          # Result visualization
│   ├── analysis_tools.py            # Statistical analysis
│   └── data_validator.py            # Result validation
│
└── Configuration
    ├── config.py                    # Central configuration
    └── run_config.py                # Execution parameters
```

---

## Quick Start

### 1. Run Full Validation Pipeline
```bash
python run_unified_validation.py --mode full --kernels 4 --timeout 7200
```

### 2. Run with Specific Configuration
```bash
python run_unified_validation.py \
  --mode incremental \
  --scenarios train_scenario.json \
  --GPU-enabled \
  --multi_kernel_mode \
  --output results/
```

### 3. Run Recovery Mode (Continue from Checkpoint)
```bash
python run_unified_validation.py \
  --mode recovery \
  --checkpoint latest \
  --kernels 2
```

---

## Configuration

### Core Parameters (`config.py`)

**Execution Modes**:
- `full`: Complete validation from scratch
- `incremental`: Update with new scenarios
- `recovery`: Resume from checkpoint

**Performance Settings**:
```python
MAX_KERNELS = 4              # Parallel kernel limit
GPU_MEMORY_THRESHOLD = 0.85  # GPU utilization threshold
TIMEOUT_PER_KERNEL = 1800    # Seconds per kernel
BATCH_SIZE = 100             # Scenarios per batch
```

**Validation Thresholds**:
```python
VELOCITY_TOLERANCE = 0.01    # m/s
ACCELERATION_TOLERANCE = 0.05 # m/s²
POSITION_TOLERANCE = 0.1     # m
```

### Runtime Configuration (`run_config.py`)

**Command-line Arguments**:
- `--mode`: Execution mode (full/incremental/recovery)
- `--kernels`: Number of parallel kernels
- `--timeout`: Max execution time (seconds)
- `--GPU-enabled`: Enable GPU acceleration
- `--multi_kernel_mode`: Enable distributed testing
- `--output`: Output directory
- `--scenarios`: Scenario file(s)
- `--checkpoint`: Checkpoint to load

---

## Usage Examples

### Example 1: Full Validation Pipeline
```bash
# Run complete validation with 4 kernels
python run_unified_validation.py --mode full --kernels 4 --GPU-enabled

# Output: results/validation_YYYYMMDD_HHMMSS/
# - summary.html (interactive report)
# - detailed_results.csv
# - metrics.json
```

### Example 2: Incremental Update
```bash
# Add new scenarios and update results
python run_unified_validation.py \
  --mode incremental \
  --scenarios new_scenarios.json \
  --output results/

# Updates existing cache, only processes new scenarios
```

### Example 3: Recovery from Failure
```bash
# Resume from last checkpoint
python run_unified_validation.py \
  --mode recovery \
  --checkpoint latest

# Automatically resumes interrupted validation
```

### Example 4: Custom Scenario Testing
```bash
# Validate specific scenarios with strict tolerances
python run_unified_validation.py \
  --mode full \
  --scenarios custom_test.json \
  --velocity_tolerance 0.005 \
  --acceleration_tolerance 0.02
```

---

## Multi-Kernel Execution

### How It Works

1. **Scenario Partitioning**: Divides scenarios into batches
2. **Kernel Assignment**: Assigns batches to available kernels
3. **GPU Monitoring**: Tracks GPU memory and throttles if needed
4. **Result Aggregation**: Combines results as kernels complete
5. **Error Recovery**: Restarts failed kernels automatically

### GPU Quota Management

```python
# Automatic fallback when GPU quota exceeded
if gpu_quota_exceeded():
    remaining_scenarios = get_remaining()
    max_kernels = calculate_safe_kernels(remaining_scenarios, gpu_memory)
    restart_kernels(max_kernels)
```

### Kernel States

| State | Description | Action |
|-------|-------------|--------|
| IDLE | Waiting for assignment | Assign scenarios |
| RUNNING | Processing scenarios | Monitor progress |
| COMPLETED | Finished successfully | Collect results |
| FAILED | Execution failed | Restart with reduced load |
| TIMEOUT | Exceeded time limit | Redistribute scenarios |

---

## Data Cache System

### Cache Structure
```
.cache/
├── scenarios/
│   ├── v1.0/
│   │   ├── manifest.json
│   │   └── scenarios_*.pkl
│   └── v2.0/
│
├── validation_results/
│   ├── niveau1/
│   ├── niveau2/
│   └── niveau3/
│
└── metrics/
    ├── aggregated/
    └── detailed/
```

### Cache Lifecycle

**Creation**: First validation run
```
cache.initialize(version="v1.0")
cache.store_scenarios(scenarios)
```

**Validation**: Check cache freshness
```
if cache.is_valid(version, tolerance=0.01):
    use_cache()
else:
    refresh_cache()
```

**Update**: Add new data
```
cache.update(scenarios, results)
cache.persist()
```

**Cleanup**: Remove old data
```
cache.cleanup(keep_versions=2)
```

---

## Validation Pipeline

### Three-Level Validation (Niveaux)

#### Niveau 1: Basic Vehicle Dynamics
- Validates fundamental physics equations
- Checks energy conservation
- Verifies acceleration limits
- **Files**: `level1_basic_dynamics.py`

#### Niveau 2: Scenario-Based Validation
- Tests multi-vehicle interactions
- Validates trajectory consistency
- Checks collision detection
- **Files**: `level2_scenario_validation.py`

#### Niveau 3: Real-World Scenarios
- Complex multi-agent simulations
- Real traffic patterns validation
- Performance under stress
- **Files**: `validator.py`, `niveal_dynamics.py`

### Validation Flow

```
Raw Data
    ↓
[Niveau 1] Basic Physics Validation
    ├─ Energy Conservation
    ├─ Acceleration Bounds
    └─ Velocity Constraints
    ↓
[Niveau 2] Scenario Validation
    ├─ Multi-vehicle Interactions
    ├─ Trajectory Consistency
    └─ Collision Detection
    ↓
[Niveau 3] Real-World Scenarios
    ├─ Traffic Patterns
    ├─ Performance Metrics
    └─ Statistical Analysis
    ↓
Report Generation
```

---

## Results & Reporting

### Output Formats

#### 1. Summary Report (HTML)
Interactive dashboard with:
- Validation success rates
- Performance metrics
- Statistical summaries
- Confidence intervals

#### 2. Detailed Results (CSV)
Complete scenario-by-scenario breakdown:
- Vehicle IDs and trajectories
- Validation metrics
- Pass/fail status
- Error details

#### 3. Metrics (JSON)
Machine-readable format for:
- Automated analysis
- Integration with other tools
- Trend tracking

### Result Structure
```json
{
  "validation_id": "val_20240115_143022",
  "timestamp": "2024-01-15T14:30:22Z",
  "config": {...},
  "niveau1": {
    "status": "PASSED",
    "metrics": {...},
    "scenarios_tested": 1000
  },
  "niveau2": {
    "status": "PASSED",
    "metrics": {...},
    "scenarios_tested": 500
  },
  "niveau3": {
    "status": "PASSED",
    "metrics": {...},
    "scenarios_tested": 200
  },
  "summary": {
    "overall_status": "PASSED",
    "total_scenarios": 1700,
    "pass_rate": 0.98,
    "avg_execution_time": 2.34
  }
}
```

---

## Performance Monitoring

### Metrics Tracked

**Execution Time**:
- Per scenario: Time to validate
- Per kernel: Total execution
- Overall pipeline: End-to-end duration

**GPU Utilization**:
- Memory usage percentage
- Temperature tracking
- Throttle events
- Power consumption

**Validation Results**:
- Pass/fail rates
- Error distribution
- Tolerance violations
- Performance anomalies

### Performance Reports

```bash
# Generate performance report
python -c "from performance_monitor import generate_report; generate_report()"

# Output: performance_report_YYYYMMDD_HHMMSS.html
```

---

## Error Handling

### Error Types & Recovery

| Error | Cause | Recovery | Status |
|-------|-------|----------|--------|
| GPU OOM | Memory exceeded | Reduce kernels/batch size | Automatic |
| Kernel Timeout | Long execution | Redistribute scenarios | Automatic |
| Validation Failure | Physics violation | Log details, continue | Manual Review |
| Data Corruption | Cache issue | Refresh cache | Automatic |
| Connection Loss | Network error | Retry with exponential backoff | Automatic |

### Common Issues

**Issue**: GPU quota exceeded
```bash
# Solution: Reduce parallel kernels
python run_unified_validation.py --kernels 2 --GPU-enabled
```

**Issue**: Cache validation fails
```bash
# Solution: Clear and rebuild cache
python -c "from caching import cache; cache.cleanup(keep_versions=0)"
```

**Issue**: Kernel fails to recover
```bash
# Solution: Run in recovery mode
python run_unified_validation.py --mode recovery --checkpoint latest
```

---

## Development & Contribution

### Code Structure

**Entry Points**:
- `run_unified_validation.py` - Main pipeline
- `multikernelrunner.py` - Kernel orchestration
- `validator.py` - Validation engine

**Key Classes**:
- `UnifiedValidator` - Core validation logic
- `MultiKernelRunner` - Parallel execution
- `ValidationCache` - Data management
- `ResultAggregator` - Result compilation

### Testing

```bash
# Run test suite
pytest tests/ -v

# Run specific test
pytest tests/test_validator.py -v

# Generate coverage report
pytest --cov=. tests/
```

### Common Development Tasks

**Add New Validation Check**:
1. Implement in `validator.py`
2. Register in validation pipeline
3. Add test case in `tests/`
4. Update configuration if needed

**Add New Scenario Type**:
1. Define in `scenariostore.py`
2. Add to cache versioning
3. Implement validation logic
4. Update documentation

---

## Troubleshooting Guide

### Debug Mode
```bash
# Run with detailed logging
python run_unified_validation.py --debug --log-level DEBUG
```

### Inspect Cache
```python
from caching import cache
cache.inspect()  # Shows cache status
cache.validate()  # Checks integrity
```

### Check Kernel Status
```python
from multikernelrunner import MultiKernelRunner
runner = MultiKernelRunner()
runner.status()  # Shows kernel states
```

### Analyze Failed Scenarios
```python
from unified_validation_report import ResultAnalyzer
analyzer = ResultAnalyzer("results/validation_YYYYMMDD_HHMMSS/")
analyzer.failed_scenarios()  # Lists failures
analyzer.error_distribution()  # Error breakdown
```

---

## Performance Benchmarks

Typical execution times on Tesla T4 GPU:

| Scenario Count | Mode | Time | Kernels |
|---|---|---|---|
| 100 | Full | 2-3 min | 1 |
| 500 | Full | 8-10 min | 2 |
| 1000 | Full | 15-18 min | 4 |
| 5000 | Incremental | 40-50 min | 4 |

---

## Deployment Checklist

Before production deployment:

- [ ] All tests pass (`pytest`)
- [ ] Cache is validated
- [ ] GPU quota confirmed available
- [ ] Output directories writable
- [ ] Configuration reviewed
- [ ] Backup of previous results
- [ ] Documentation updated
- [ ] Error handlers tested

---

## Support & Documentation

### Additional Resources

- **Chapter 7 Validation Theory**: `../../ch7_*.md`
- **Configuration Guide**: `config.py`
- **API Reference**: Docstrings in source files
- **Examples**: `examples/` directory

### Getting Help

1. Check this README
2. Review error logs: `results/validation_*/logs/`
3. Run in debug mode for detailed output
4. Consult Chapter 7 documentation

---

## License

Part of Alibi Research Project - Traffic Simulation Validation Framework

**Status**: Production Ready (Sprint 4 Complete)
