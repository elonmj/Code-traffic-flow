#!/usr/bin/env python
"""Quick local test for Bug #11 fix - uniform IC format"""

import sys
import os
sys.path.insert(0, 'd:/Projets/Alibi/Code project')
sys.path.insert(0, 'd:/Projets/Alibi/Code project/arz_model')

print("=" * 80)
print("QUICK LOCAL TEST - BUG #11 FIX")
print("=" * 80)

# Test 1: Create scenario config with uniform IC
print("\n[TEST 1] Creating scenario config...")
from validation_ch7.scripts.test_section_7_6_rl_performance import RLPerformanceValidationTest

validator = RLPerformanceValidationTest(quick_test=True)

# Generate traffic_light_control scenario
print("[TEST 1] Generating traffic_light_control.yml...")
scenario_path = validator._create_scenario_config('traffic_light_control')
print(f"[TEST 1] ✓ Scenario created: {scenario_path}")

# Read the YAML
import yaml
with open(scenario_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"\n[TEST 1] IC config: {config['initial_conditions']}")

# Test 2: Initialize SimulationRunner with uniform IC
print("\n[TEST 2] Initializing SimulationRunner...")
from arz_model.simulation.runner import SimulationRunner

try:
    runner = SimulationRunner(scenario_path, device='cpu', quiet=True)
    print("[TEST 2] ✓ SimulationRunner initialized successfully!")
    print(f"[TEST 2] Initial state shape: {runner.U.shape}")
    print(f"[TEST 2] Initial rho_m: mean={runner.U[0].mean():.4f}, std={runner.U[0].std():.6f}")
    print(f"[TEST 2] Initial w_m: mean={runner.U[1].mean():.4f}, std={runner.U[1].std():.6f}")
    
    # Check if uniform
    is_uniform = runner.U[0].std() < 1e-10
    print(f"[TEST 2] Is uniform IC? {is_uniform} (std < 1e-10)")
    
    if is_uniform:
        print("\n" + "=" * 80)
        print("✅ BUG #11 FIX VALIDATED - UNIFORM IC WORKS!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ PROBLEM: IC not uniform (has variation)")
        print("=" * 80)
        
except Exception as e:
    print(f"[TEST 2] ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
