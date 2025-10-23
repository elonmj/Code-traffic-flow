#!/usr/bin/env python3
"""Debug script to trace RL agent performance calculations"""
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "Code_RL"))

# Quick test to see what's happening
from validation_ch7.scripts.test_section_7_6_rl_performance import RLPerformanceValidationTest

# Create test instance
tester = RLPerformanceValidationTest(quick_test=True)

print("\n" + "="*80)
print("DEBUG: Check scenario configuration loading")
print("="*80)

scenario_path = tester._create_scenario_config('traffic_light_control')
print(f"Scenario path: {scenario_path}")
print(f"Scenario exists: {scenario_path.exists()}")

# Load the scenario to see what parameters are used
import yaml
with open(scenario_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"\nScenario parameters:")
print(f"  N: {config.get('N')}")
print(f"  xmin: {config.get('xmin')}")
print(f"  xmax: {config.get('xmax')}")
print(f"  t_final: {config.get('t_final')}")
print(f"  output_dt: {config.get('output_dt')}")

# Check for network and traffic control
print(f"\nNetwork segments: {list(config.get('network', {}).get('segments', {}).keys())}")
print(f"Nodes: {list(config.get('network', {}).get('nodes', {}).keys())}")

# Now let's trace what happens during evaluation
print("\n" + "="*80)
print("DEBUG: Trace baseline cache check")
print("="*80)

baseline_duration = 600.0 if tester.quick_test else 3600.0
control_interval = 15.0

cached_states = tester._load_baseline_cache(
    'traffic_light_control', 
    scenario_path,
    baseline_duration, 
    control_interval
)

if cached_states is not None:
    print(f"✅ Baseline cache found: {len(cached_states)} states")
    print(f"   Required for {baseline_duration}s at {control_interval}s interval: {int(baseline_duration / control_interval) + 1} states")
else:
    print(f"❌ No baseline cache found")

print("\n" + "="*80)
print("DEBUG: Check what states are actually returned")
print("="*80)

# Check if states are being collected properly by running a quick baseline
baseline_controller = tester.BaselineController('traffic_light_control')
baseline_states, _ = tester.run_control_simulation(
    baseline_controller, 
    scenario_path,
    duration=600.0,
    control_interval=15.0,
    device='cpu',
    controller_type='BASELINE'
)

print(f"Baseline states collected: {len(baseline_states) if baseline_states else 0}")
if baseline_states:
    print(f"  First state shape: {baseline_states[0].shape}")
    print(f"  Last state shape: {baseline_states[-1].shape}")
    
    # Check if states are actually changing
    import numpy as np
    diffs = []
    for i in range(1, len(baseline_states)):
        diff = np.abs(baseline_states[i] - baseline_states[i-1]).mean()
        diffs.append(diff)
    
    print(f"  Mean state change between steps: {np.mean(diffs):.6f}")
    print(f"  State differences: min={np.min(diffs):.6e}, max={np.max(diffs):.6e}")
