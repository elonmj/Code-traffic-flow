#!/usr/bin/env python3
"""
CRITICAL TEST: Do baseline metrics ACTUALLY CHANGE with different control strategies?
This is the definitive test to answer: "Are results hardcoded?"

If flow metric is IDENTICAL for RED_ONLY and GREEN_ONLY → Results ARE HARDCODED
If flow metric DIFFERS for RED_ONLY and GREEN_ONLY → Results ARE REAL
"""

import sys
import numpy as np
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Code_RL" / "src"))
sys.path.insert(0, str(project_root / "validation_ch7" / "scripts"))

from validation_ch7.scripts.test_section_7_6_rl_performance import RLPerformanceValidationTest

print("="*80)
print("CRITICAL TEST: Baseline Metrics Responsiveness")
print("="*80)
print("\nObjective: Verify metrics CHANGE when using different control strategies")
print("Expected:  RED_ONLY flow << GREEN_ONLY flow")
print("\nIf IDENTICAL → Results HARDCODED ❌")
print("If DIFFERENT → Results REAL ✅")
print("="*80)

# Quick scenario setup
test = RLPerformanceValidationTest(quick_test=True)
scenario_type = 'traffic_light_control'

print(f"\nSTEP 1: Generate scenario")
scenario_path = test._create_scenario_config(scenario_type)
print(f"✅ Created: {scenario_path}")

print(f"\nSTEP 2: Test RED_ONLY strategy")
print("  Running 300s simulation (20 control steps)...")
baseline_red = test.BaselineController(scenario_type, strategy='red_only')
red_states, _ = test.run_control_simulation(
    baseline_red, 
    scenario_path,
    duration=300.0,  # Very short: 300s
    control_interval=15.0,
    device='cpu',
    controller_type='BASELINE_RED_ONLY'
)

if red_states is None:
    print("❌ RED_ONLY simulation failed!")
    sys.exit(1)

print(f"  ✅ Collected {len(red_states)} state snapshots")

red_performance = test.evaluate_traffic_performance(red_states, scenario_type, scenario_path)
print(f"  RED_ONLY flow: {red_performance['total_flow']:.6f}")
print(f"  RED_ONLY efficiency: {red_performance['efficiency']:.6f}")

print(f"\nSTEP 3: Test GREEN_ONLY strategy")
print("  Running 300s simulation (20 control steps)...")
baseline_green = test.BaselineController(scenario_type, strategy='green_only')
green_states, _ = test.run_control_simulation(
    baseline_green, 
    scenario_path,
    duration=300.0,  # Same duration
    control_interval=15.0,
    device='cpu',
    controller_type='BASELINE_GREEN_ONLY'
)

if green_states is None:
    print("❌ GREEN_ONLY simulation failed!")
    sys.exit(1)

print(f"  ✅ Collected {len(green_states)} state snapshots")

green_performance = test.evaluate_traffic_performance(green_states, scenario_type, scenario_path)
print(f"  GREEN_ONLY flow: {green_performance['total_flow']:.6f}")
print(f"  GREEN_ONLY efficiency: {green_performance['efficiency']:.6f}")

print("\n" + "="*80)
print("CRITICAL ANALYSIS")
print("="*80)

flow_diff = green_performance['total_flow'] - red_performance['total_flow']
efficiency_diff = green_performance['efficiency'] - red_performance['efficiency']

print(f"\nFlow difference (GREEN - RED): {flow_diff:.6f}")
print(f"Efficiency difference (GREEN - RED): {efficiency_diff:.6f}")

# Determine if results are hardcoded
flow_identical = abs(flow_diff) < 1e-6  # Essentially zero
efficiency_identical = abs(efficiency_diff) < 1e-6

print("\n" + "="*80)
if flow_identical and efficiency_identical:
    print("❌ VERDICT: Results ARE HARDCODED")
    print("   Metrics are IDENTICAL despite different control strategies!")
    print("   The simulator state is not evolving properly, or metrics aren't extracted correctly.")
    sys.exit(1)
elif flow_diff > 0:  # GREEN should have more flow than RED
    print("✅ VERDICT: Results ARE REAL")
    print(f"   GREEN_ONLY flow is {flow_diff:.6f} units HIGHER than RED_ONLY")
    print(f"   This is physically correct: GREEN allows traffic to flow")
    sys.exit(0)
else:
    print("⚠️  VERDICT: UNEXPECTED - GREEN flow is LOWER than RED")
    print("   This suggests a bug in the control logic or metrics calculation")
    sys.exit(1)
