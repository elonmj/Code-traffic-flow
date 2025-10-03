# ARZ-RL Chapter 7 Validation System

## Overview

Complete validation system for Chapter 7 (ARZ-RL model validation) running on Kaggle GPU with intelligent output management.

## Features

###  Phase 1: NPZ Integration (COMPLETED)
- Automatic NPZ file saving for all validation tests
- Simulation states preserved for deep analysis
- Integration with code/io/data_manager.py

###  Phase 2: CLI Custom Commit Messages (COMPLETED)
- Optional --commit-message parameter
- Auto-generated fallback with timestamp
- Clean Git workflow automation

###  Phase 3: Kernel Script with Cleanup (COMPLETED)
- Full cleanup pattern from kaggle_manager_github.py
- Copy validation results  Remove repo  Only outputs preserved
- Output size: ~250 MB-2.5 GB (instead of ~5 GB+ with full repo)

## Usage

### Basic Usage

`ash
# Run Section 7.3 (Analytical Validation)
python validation_cli.py --section section_7_3_analytical

# With custom commit message
python validation_cli.py --section section_7_3_analytical --commit-message "Fixed WENO5 convergence"

# With custom timeout (default: 4000s)
python validation_cli.py --section section_7_4_calibration --timeout 5400
`

### Available Sections

| Section | Revendications | Description | Est. Time |
|---------|---------------|-------------|-----------|
| section_7_3_analytical | R1, R3 | Analytical validation, Riemann, convergence | 45 min |
| section_7_4_calibration | R2 | Victoria Island calibration | 60 min |
| section_7_5_digital_twin | R3, R4, R6 | Digital twin behavioral validation | 75 min |
| section_7_6_rl_performance | R5 | RL performance vs baselines | 90 min |
| section_7_7_robustness | R4, R6 | Robustness tests (GPU/CPU, extreme conditions) | 60 min |

## Output Structure

After kernel execution, download contains:

`
/kaggle/working/validation_results/
  section_7_3_analytical/
    npz/
      riemann_test_1_20251002_143055.npz
      riemann_test_2_20251002_143120.npz
      convergence_N100_20251002_143150.npz
      convergence_N200_20251002_143230.npz
      ...
    figures/
      riemann_shock_moto.png
      convergence_order_plot.pdf
    metrics/
      analytical_metrics.json
      analytical_summary.csv
    section_7_3_content.tex
  session_summary.json
  validation_log.txt
`

**NO Code-traffic-flow/ directory!** Only validation results preserved.

## Implementation Details

### NPZ File Saving

NPZ files contain complete simulation state:

`python
# Structure of each .npz file
times: np.array          # Simulation timesteps
states: np.array         # Shape (num_times, 4, N_physical)
grid_info: dict          # Grid parameters
params_dict: dict        # Model parameters
grid_object: pickled     # Full Grid1D object
params_object: pickled   # Full ModelParameters object
`

### Git Automation

Custom commit messages supported via CLI:

`python
# In ensure_git_up_to_date()
if commit_message:
    final_commit_message = commit_message  # Use custom
else:
    final_commit_message = f"Auto-commit - {timestamp}"  # Fallback
`

### Cleanup Pattern

Kernel script cleanup logic:

1. Clone repo from GitHub
2. Run validation tests
3. **Copy** alidation_ch7/results/  /kaggle/working/validation_results/
4. **Remove** entire cloned repository
5. Create session_summary.json
6. Only /kaggle/working/ preserved in output

## Architecture

`
validation_kaggle_manager.py
 ValidationKaggleManager (extends KaggleManagerGitHub)
    _build_validation_kernel_script()  # Phase 3: Kernel with cleanup
    run_validation_section()           # Phase 2: Custom commit
    _monitor_kernel_with_session_detection()

validation_cli.py
 argparse CLI with --commit-message support

validation_ch7/scripts/
 test_section_7_3_analytical.py         # Phase 1: NPZ saving integrated
 test_section_7_4_calibration.py
 test_section_7_5_digital_twin.py
 test_section_7_6_rl_performance.py
 test_section_7_7_robustness.py
`

## Testing

### Local Test (without Kaggle)

`ash
# Test NPZ saving locally
cd validation_ch7/scripts
python test_section_7_3_analytical.py
# Check validation_ch7/results/npz/ for .npz files
`

### Full Kaggle Test

`ash
# Run complete workflow
python validation_cli.py --section section_7_3_analytical --commit-message "Test validation system"

# Monitor at: https://www.kaggle.com/code/{kernel_slug}
# Download output after completion
# Verify: No Code-traffic-flow/, only validation_results/
`

## Troubleshooting

### NPZ files not generated

Check that code/io/data_manager.py is imported correctly:
`python
from code.io.data_manager import save_simulation_data
`

### Git commit fails

Ensure you're in a Git repository and have push access:
`ash
git status
git remote -v
`

### Kaggle kernel timeout

Increase timeout for long-running sections:
`ash
python validation_cli.py --section section_7_6_rl_performance --timeout 7200
`

## Next Steps

- [x] Phase 1: NPZ Integration
- [x] Phase 2: Custom Commit Messages  
- [x] Phase 3: Kernel Cleanup Logic
- [ ] Phase 4: Run and validate all 5 sections
- [ ] Phase 5: Download and verify NPZ files
- [ ] Phase 6: Integrate results into thesis

## References

- Research Document: .copilot-tracking/research/20251002-chapter7-validation-kaggle-architecture-research.md
- Base Implementation: kaggle_manager_github.py (proven pattern)
- Data Manager: code/io/data_manager.py (NPZ infrastructure)
