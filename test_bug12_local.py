#!/usr/bin/env python
"""Local test for Bug #12 fix - verify initial_equilibrium_state is set for uniform IC"""
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'arz_model'))
sys.path.insert(0, str(project_root / 'validation_ch7' / 'scripts'))

import yaml
import os
from arz_model.simulation.runner import SimulationRunner

# Change to arz_model directory for base config access
os.chdir(project_root / 'arz_model')

print("="*80)
print("BUG #12 FIX VALIDATION TEST")
print("Testing: uniform IC → initial_equilibrium_state should be set")
print("="*80)

# Test 1: Create minimal uniform IC scenario
print("\n[TEST 1] Creating uniform IC scenario config...")
scenario_config = {
    'name': 'test_uniform_ic',
    'parameters': {
        'V0_m': 25.0,
        'V0_c': 22.2,
        'tau_m': 1.0,
        'tau_c': 1.2,
        'K_m_kmh': 10.0,
        'K_c_kmh': 15.0
    },
    'grid': {
        'xmin': 0.0,
        'xmax': 1000.0,
        'N': 100
    },
    'simulation': {
        't_final_sec': 10.0,
        'output_dt_sec': 1.0
    },
    'initial_conditions': {
        'type': 'uniform',
        'state': [0.08, 12.0, 0.1, 10.0]  # [rho_m, w_m, rho_c, w_c]
    },
    'boundary_conditions': {
        'left': {'type': 'inflow', 'state': [0.08, 12.0, 0.1, 10.0]},
        'right': {'type': 'outflow'}
    },
    'time_integration': {
        'dt': 0.1,
        'T_max': 10.0,
        'cfl': 0.45
    },
    'road_quality': {
        'type': 'uniform',
        'value': 2
    }
}

test_scenario_path = project_root / 'test_bug12_scenario.yml'
with open(test_scenario_path, 'w') as f:
    yaml.dump(scenario_config, f)

print(f"✓ Scenario created: {test_scenario_path}")
print(f"  IC type: uniform")
print(f"  IC state: {scenario_config['initial_conditions']['state']}")

# Test 2: Initialize SimulationRunner and check initial_equilibrium_state
print("\n[TEST 2] Initializing SimulationRunner...")
try:
    runner = SimulationRunner(str(test_scenario_path), device='cpu', quiet=True)
    print(f"✓ SimulationRunner initialized successfully")
    
    # Test 3: Check if initial_equilibrium_state was set
    print("\n[TEST 3] Checking initial_equilibrium_state...")
    if hasattr(runner, 'initial_equilibrium_state'):
        if runner.initial_equilibrium_state is not None:
            print(f"✓ initial_equilibrium_state IS SET: {runner.initial_equilibrium_state}")
            print(f"  Type: {type(runner.initial_equilibrium_state)}")
            print(f"  Length: {len(runner.initial_equilibrium_state) if hasattr(runner.initial_equilibrium_state, '__len__') else 'N/A'}")
            
            # Test 4: Verify it has correct format for set_traffic_signal_state
            print("\n[TEST 4] Verifying state format for traffic signal...")
            try:
                # Simulate what set_traffic_signal_state does
                base_state = runner.initial_equilibrium_state
                red_state = [
                    base_state[0],           # rho_m
                    base_state[1] * 0.5,     # w_m reduced
                    base_state[2],           # rho_c
                    base_state[3] * 0.5      # w_c reduced
                ]
                print(f"✓ Red phase state construction: SUCCESS")
                print(f"  Original: {base_state}")
                print(f"  Red (50% velocity): {red_state}")
            except Exception as e:
                print(f"✗ Red phase state construction: FAILED")
                print(f"  Error: {e}")
                sys.exit(1)
        else:
            print(f"✗ initial_equilibrium_state is None - BUG #12 NOT FIXED!")
            sys.exit(1)
    else:
        print(f"✗ initial_equilibrium_state attribute missing - BUG #12 NOT FIXED!")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✓✓✓ ALL TESTS PASSED - BUG #12 FIX VALIDATED ✓✓✓")
    print("="*80)
    print("\nSummary:")
    print("  ✓ Uniform IC scenario created")
    print("  ✓ SimulationRunner initialized")
    print("  ✓ initial_equilibrium_state is set (not None)")
    print("  ✓ State format compatible with set_traffic_signal_state")
    print("\n→ Safe to launch Kaggle kernel with Bug #12 fix")
    
except Exception as e:
    print(f"\n✗ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Cleanup
    if test_scenario_path.exists():
        test_scenario_path.unlink()
        print(f"\n[CLEANUP] Removed test scenario: {test_scenario_path}")
