#!/usr/bin/env python3
"""
Quick test script to verify Pydantic configuration integration
with validation_ch7 test script components.

Tests:
1. RLConfigBuilder import and instantiation
2. Scenario configuration creation
3. YAML generation from Pydantic configs
"""

import sys
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("QUICK TEST: Pydantic Configuration for Validation Script")
print("=" * 80)

# Test 1: Import RLConfigBuilder
print("\n[TEST 1] Importing RLConfigBuilder...")
try:
    from Code_RL.src.utils.config import RLConfigBuilder
    print("✅ SUCCESS: RLConfigBuilder imported")
except ImportError as e:
    print(f"❌ FAILED: Could not import RLConfigBuilder: {e}")
    sys.exit(1)

# Test 2: Create RL configuration
print("\n[TEST 2] Creating RL configuration for Lagos scenario...")
try:
    rl_config = RLConfigBuilder.for_training(
        scenario="lagos",
        N=100,
        episode_length=600.0,
        device='cpu'
    )
    print("✅ SUCCESS: RL configuration created")
    print(f"   - Grid size: N={rl_config.arz_simulation_config.grid.N}")
    print(f"   - Duration: {rl_config.arz_simulation_config.t_final}s")
except Exception as e:
    print(f"❌ FAILED: Could not create RL configuration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Convert to dictionary format suitable for YAML
print("\n[TEST 3] Converting Pydantic config to YAML-compatible dictionary...")
try:
    arz_config = rl_config.arz_simulation_config
    # Use mode='json' to get primitive types suitable for YAML serialization
    config_dict = arz_config.model_dump(mode='json')
    print("✅ SUCCESS: Converted to YAML-compatible dictionary")
    print(f"   - Keys: {list(config_dict.keys())[:5]}...")
except Exception as e:
    print(f"❌ FAILED: Could not convert to legacy format: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Save as YAML (simulate what validation script does)
print("\n[TEST 4] Saving configuration as YAML file...")
try:
    test_yaml_path = project_root / "test_scenario_output.yml"
    with open(test_yaml_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    print(f"✅ SUCCESS: YAML saved to {test_yaml_path}")
    
    # Verify file exists and can be read back
    with open(test_yaml_path, 'r') as f:
        loaded_config = yaml.safe_load(f)
    print(f"   - Loaded back successfully")
    print(f"   - File size: {test_yaml_path.stat().st_size} bytes")
    
    # Cleanup
    test_yaml_path.unlink()
    print(f"   - Cleanup: Removed test file")
except Exception as e:
    print(f"❌ FAILED: Could not save/load YAML: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Simulate validation script method
print("\n[TEST 5] Simulating validation script _create_scenario_config_pydantic()...")
try:
    scenario_type = 'traffic_light_control'
    
    # Map scenario types to RLConfigBuilder scenarios
    scenario_map = {
        'traffic_light_control': 'lagos',
        'ramp_metering': 'lagos',
        'adaptive_speed_control': 'simple'
    }
    
    builder_scenario = scenario_map.get(scenario_type, 'lagos')
    
    # Create RL configuration
    rl_config = RLConfigBuilder.for_training(
        scenario=builder_scenario,
        N=100,
        episode_length=600.0,
        device='cpu'
    )
    
    # Convert to YAML-compatible format
    arz_config = rl_config.arz_simulation_config
    config_dict = arz_config.model_dump(mode='json')
    
    print("✅ SUCCESS: Validation script method simulation completed")
    print(f"   - Scenario: {scenario_type} → {builder_scenario}")
    print(f"   - Config created and converted successfully")
    
except Exception as e:
    print(f"❌ FAILED: Simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("🎉 ALL TESTS PASSED")
print("=" * 80)
print("\nValidation script Pydantic integration is working correctly!")
print("The script can:")
print("  ✅ Import RLConfigBuilder")
print("  ✅ Create Pydantic configurations")
print("  ✅ Convert to legacy YAML format")
print("  ✅ Save and load YAML files")
print("  ✅ Simulate validation script workflow")
print("\nReady for use in test_section_7_6_rl_performance.py")
