#!/usr/bin/env python3
"""
Quick test to validate corrected scenario configuration.
Tests ONE scenario (free_flow) to verify YAML structure works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'code'))

from test_section_7_5_digital_twin import DigitalTwinValidationTest


def quick_test():
    """Quick test of corrected scenario configuration."""
    print("="*80)
    print("QUICK TEST: Corrected Scenario Configuration")
    print("="*80)
    
    # Create test instance
    tester = DigitalTwinValidationTest()
    
    # Create ONE scenario
    print("\n[STEP 1] Creating free_flow scenario with corrected configuration...")
    scenario_path = tester.create_scenario_config('free_flow', grid_size=200, final_time=100.0)
    
    # Display generated YAML
    print(f"\n[STEP 2] Generated scenario: {scenario_path}")
    print("\nYAML Content:")
    print("-" * 80)
    with open(scenario_path) as f:
        content = f.read()
        print(content)
    print("-" * 80)
    
    # Verify key fields
    import yaml
    with open(scenario_path) as f:
        config = yaml.safe_load(f)
    
    print("\n[STEP 3] Verification:")
    ic = config.get('initial_conditions', {})
    print(f"  - IC type: {ic.get('type')}")
    print(f"  - R_val: {ic.get('R_val')}")
    
    if 'background_state' in ic:
        bg = ic['background_state']
        print(f"  - Background rho_m: {bg.get('rho_m')} veh/m ({bg.get('rho_m')*1000} veh/km)")
        print(f"  - Background rho_c: {bg.get('rho_c')} veh/m ({bg.get('rho_c')*1000} veh/km)")
    
    if 'perturbation' in ic:
        pert = ic['perturbation']
        print(f"  - Perturbation amplitude: {pert.get('amplitude')} veh/m")
        print(f"  - Wave number: {pert.get('wave_number')}")
    
    if 'road' in config:
        road = config['road']
        print(f"  - Road type: {road.get('type')}")
        print(f"  - Road R_val: {road.get('R_val')}")
    
    print(f"\n[RESULT] ✅ Scenario configuration created successfully!")
    print(f"[NEXT] Ready to test on Kaggle with corrected configurations.")
    
    return scenario_path


if __name__ == '__main__':
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
