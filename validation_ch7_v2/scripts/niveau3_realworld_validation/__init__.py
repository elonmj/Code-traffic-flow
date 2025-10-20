"""
Niveau 3: Real-World Data Validation (SPRINT 4)

This module validates ARZ model predictions against observed TomTom taxi trajectory data.

Revendication R2: "The ARZ model matches observed West African traffic patterns"

Components:
-----------
- tomtom_trajectory_loader.py: Load and parse TomTom GPS trajectories
- feature_extractor.py: Extract speed, density, flow metrics from observed data
- validation_comparison.py: Statistical comparison (theory vs observed)
- quick_test_niveau3.py: Orchestration script for full validation

Validation Tests:
----------------
1. Speed differential comparison (Δv observed vs predicted)
2. Throughput ratio validation (Q_motos/Q_cars)
3. Fundamental diagram correlation (Q-ρ curves)
4. Infiltration pattern analysis
5. Statistical significance testing (KS test, Spearman correlation)

Expected Outputs:
----------------
- figures/niveau3_realworld/: Comparison plots (6 PNG + PDF files)
- data/validation_results/realworld_tests/: Metrics JSON files
- SPRINT4_DELIVERABLES/: Thesis-ready documentation
"""

__version__ = "1.0.0"
__author__ = "ARZ-RL Validation Team"
__date__ = "2025-10-17"
