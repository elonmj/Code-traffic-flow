#!/usr/bin/env python3
"""
Debug test: Verify if baseline results change with REAL control differences
"""

import sys
import os
import numpy as np
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from Code_RL.src.utils.config import create_scenario_config_with_lagos_data

def main():
    print("\n" + "="*80)
    print("HARDCODED RESULTS DEBUG TEST")
    print("="*80)
    print("\nStep 1: CLEAN CACHE")
    print("-" * 80)
    
    cache_dir = Path(project_root) / "validation_ch7/cache/section_7_6"
    if cache_dir.exists():
        print(f"Removing old cache: {cache_dir}")
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Cache cleaned")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Cache directory created")
    
    print("\nStep 2: GENERATE TEST SCENARIO")
    print("-" * 80)
    
    scenarios_dir = Path(project_root) / "validation_ch7/scenarios/rl_scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    
    scenario_config = scenarios_dir / "debug_scenario_network.yml"
    
    print(f"Generating test scenario...")
    create_scenario_config_with_lagos_data(
        'traffic_light_control',
        str(scenario_config),
        duration=7200,
        domain_length=2000
    )
    print(f"✅ Generated {scenario_config}")
    
    print("\nStep 3: RUN QUICK TEST (30 timesteps = 450s)")
    print("-" * 80)
    
    from validation_ch7.scripts.test_section_7_6_rl_performance import RLPerformanceValidationTest
    
    test = RLPerformanceValidationTest(quick_test=True)
    result = test.test_scenario(scenario_type='traffic_light_control', quick_test=True)
    
    print(f"\n✅ Test completed!")
    print(f"Result: {result}")
    
    if result.get('success'):
        print(f"\n📊 METRICS:")
        print(f"  Flow improvement: {result['improvements']['flow_improvement']:.2f}%")
        print(f"  Efficiency improvement: {result['improvements']['efficiency_improvement']:.2f}%")
        print(f"  Delay reduction: {result['improvements']['delay_reduction']:.2f}%")
    else:
        print(f"\n❌ Test failed: {result.get('error')}")

if __name__ == '__main__':
    main()
