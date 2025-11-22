# Thesis Execution Plan

This document outlines the workflow to generate all results for the thesis (Sections 7 & 8).

## Overview

The process is divided into 3 stages to manage execution time and resources on Kaggle.

| Stage | Script | Purpose | Output |
|-------|--------|---------|--------|
| **1** | `launch_thesis_stage1.py` | Model Validation (Riemann, Behavioral, MDP) | `results/thesis_stage1/*.npz`, `*.json` |
| **2** | `launch_thesis_stage2.py` | RL Training (PPO) & Baseline Evaluation | `results/thesis_stage2/*.json`, `logs/` |
| **3** | `launch_thesis_stage3.py` | Visualization & Figure Generation | `results/thesis_figures/*.png` |

## Execution Instructions

### Step 1: Model Validation (Section 7)
Run the validation suite to verify the ARZ model and MDP environment.

```bash
python launch_thesis_stage1.py --timeout 3600
```
*Estimated time: ~20-30 mins*

### Step 2: RL Training & Evaluation (Section 8)
Train the PPO agent and compare it against the Fixed-Time baseline.

```bash
python launch_thesis_stage2.py --timesteps 100000 --timeout 7200
```
*Estimated time: ~1.5 - 2 hours*

### Step 3: Visualization
Generate the final figures for the thesis using the data from Steps 1 & 2.

**Note:** Ensure the results from Step 1 and Step 2 are available in the `results/` directory before running this step.

```bash
python launch_thesis_stage3.py
```
*Estimated time: < 5 mins*

## Output Files

### Stage 1
- `table_7_1_riemann_validation.json`
- `table_7_2_behavioral_validation.json`
- `mdp_sanity_checks.json`
- `riemann_*.npz` (Profile data)
- `behavioral_*.npz` (Profile data)

### Stage 2
- `baseline_results.json`
- `rl_results.json`
- `comparison_summary.json`
- `logs/` (Training curves)
- `ppo_victoria_island_final.zip` (Trained model)

### Stage 3
- `fig_7_*.png` (Riemann profiles)
- `fig_8_training_curve.png`
- `fig_8_comparison.png`
