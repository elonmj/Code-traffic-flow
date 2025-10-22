"""
Unit tests for ParameterManager class

Tests heterogeneous parameter management: global defaults + local overrides.

Author: ARZ Research Team
Date: 2025-10-21 (Phase 6 - Jour 2)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from arz_model.core.parameters import ModelParameters
from arz_model.core.parameter_manager import ParameterManager


def test_basic_get_global():
    """Test getting global parameters when no local overrides exist."""
    print("\n" + "="*60)
    print("TEST 1: Basic Global Parameter Access")
    print("="*60)
    
    # Setup
    global_params = ModelParameters()
    global_params.tau_c = 1.5  # Set a test value
    pm = ParameterManager(global_params)
    
    # Test: Get parameter with no overrides should return global default
    tau_c_global = pm.get('seg_test', 'tau_c')
    
    print(f"Global tau_c: {tau_c_global:.2f} s")
    print(f"Expected: {global_params.tau_c:.2f} s")
    
    assert tau_c_global == global_params.tau_c, "Should return global default"
    print("✅ PASS: Returns global default when no local override")


def test_set_and_get_local():
    """Test setting and getting local parameter overrides."""
    print("\n" + "="*60)
    print("TEST 2: Local Parameter Override")
    print("="*60)
    
    # Setup
    global_params = ModelParameters()
    global_params.tau_c = 1.5  # Global default
    pm = ParameterManager(global_params)
    
    # Set local override for arterial segment
    arterial_tau = 1.0  # Faster relaxation
    pm.set_local('seg_arterial', 'tau_c', arterial_tau)
    
    # Test: Get local override
    tau_c_arterial = pm.get('seg_arterial', 'tau_c')
    tau_c_other = pm.get('seg_other', 'tau_c')
    
    print(f"Arterial tau_c (local): {tau_c_arterial:.2f} s")
    print(f"Other tau_c (global): {tau_c_other:.2f} s")
    
    assert tau_c_arterial == arterial_tau, "Should return local override"
    assert tau_c_other == global_params.tau_c, "Should return global for other segments"
    print("✅ PASS: Local override takes precedence")


def test_set_local_dict():
    """Test setting multiple local parameters at once."""
    print("\n" + "="*60)
    print("TEST 3: Multiple Local Overrides")
    print("="*60)
    
    # Setup
    global_params = ModelParameters()
    pm = ParameterManager(global_params)
    
    # Set multiple overrides for arterial - using custom parameters
    arterial_params = {
        'vmax_arterial': 13.89,  # 50 km/h (custom param)
        'vmax_moto_arterial': 15.28,  # 55 km/h (custom param)
        'tau_c': 1.0
    }
    pm.set_local_dict('seg_arterial', arterial_params)
    
    # Test: All overrides applied
    vmax = pm.get('seg_arterial', 'vmax_arterial')
    vmax_moto = pm.get('seg_arterial', 'vmax_moto_arterial')
    tau_c = pm.get('seg_arterial', 'tau_c')
    
    print(f"Arterial vmax: {vmax:.2f} m/s (override)")
    print(f"Arterial vmax_moto: {vmax_moto:.2f} m/s (override)")
    print(f"Arterial tau_c: {tau_c:.2f} s (override)")
    
    assert vmax == 13.89
    assert vmax_moto == 15.28
    assert tau_c == 1.0
    print("✅ PASS: Multiple overrides applied correctly")


def test_get_all():
    """Test getting complete ModelParameters with overrides."""
    print("\n" + "="*60)
    print("TEST 4: Get Complete Parameters (get_all)")
    print("="*60)
    
    # Setup
    global_params = ModelParameters()
    global_params.tau_c = 1.5  # Global default
    global_params.tau_m = 2.0  # Global default
    global_params.alpha = 0.5  # Global default
    pm = ParameterManager(global_params)
    
    # Set local overrides for residential (custom params)
    pm.set_local_dict('seg_residential', {
        'vmax_residential': 5.56,   # 20 km/h (custom)
        'tau_c': 2.0    # Slower relaxation
    })
    
    # Test: Get complete parameters
    residential_params = pm.get_all('seg_residential')
    
    print(f"Residential tau_c: {residential_params.tau_c:.2f} s (override)")
    print(f"Residential tau_m: {residential_params.tau_m:.2f} s (global default)")
    print(f"Residential alpha: {residential_params.alpha:.2f} (global default)")
    
    # Check overrides applied
    assert residential_params.tau_c == 2.0, "Local tau_c should be applied"
    
    # Check globals unchanged
    assert residential_params.tau_m == global_params.tau_m, "Global tau_m should remain"
    assert residential_params.alpha == global_params.alpha, "Global alpha should remain"
    
    # Custom params are stored but not as ModelParameters attributes
    # They're accessible via the local_overrides dict
    custom_params = pm.get_overrides('seg_residential')
    assert custom_params['vmax_residential'] == 5.56, "Custom param stored correctly"
    print(f"Custom param vmax_residential: {custom_params['vmax_residential']:.2f} m/s")
    
    print("✅ PASS: get_all() returns correct hybrid parameters")


def test_has_local():
    """Test checking for local overrides."""
    print("\n" + "="*60)
    print("TEST 5: Check for Local Overrides (has_local)")
    print("="*60)
    
    # Setup
    global_params = ModelParameters()
    pm = ParameterManager(global_params)
    
    pm.set_local('seg_arterial', 'vmax_arterial', 13.89)
    
    # Test: Check for overrides
    has_arterial = pm.has_local('seg_arterial')
    has_arterial_vmax = pm.has_local('seg_arterial', 'vmax_arterial')
    has_arterial_tau = pm.has_local('seg_arterial', 'tau_c')
    has_other = pm.has_local('seg_other')
    
    print(f"seg_arterial has overrides: {has_arterial}")
    print(f"seg_arterial has vmax_arterial override: {has_arterial_vmax}")
    print(f"seg_arterial has tau_c override: {has_arterial_tau}")
    print(f"seg_other has overrides: {has_other}")
    
    assert has_arterial == True, "Should detect overrides"
    assert has_arterial_vmax == True, "Should detect vmax_arterial override"
    assert has_arterial_tau == False, "Should not detect non-existent tau_c"
    assert has_other == False, "Should not detect overrides for other segments"
    
    print("✅ PASS: has_local() works correctly")


def test_heterogeneous_network():
    """Test realistic heterogeneous network scenario."""
    print("\n" + "="*60)
    print("TEST 6: Realistic Heterogeneous Network")
    print("="*60)
    
    # Setup global defaults
    global_params = ModelParameters()
    pm = ParameterManager(global_params)
    
    # Define arterial segments (50 km/h) - using custom params
    arterial_params = {
        'vmax_arterial': 13.89,  # 50 km/h (custom)
        'vmax_moto_arterial': 15.28,  # 55 km/h (custom)
        'tau_c': 1.0
    }
    pm.set_local_dict('seg_main_1', arterial_params)
    pm.set_local_dict('seg_main_2', arterial_params)
    
    # Define residential segment (20 km/h) - using custom params
    residential_params = {
        'vmax_residential': 5.56,   # 20 km/h (custom)
        'vmax_moto_residential': 6.94,   # 25 km/h (custom)
        'tau_c': 2.0,   # Slower reaction
        'tau_m': 3.0
    }
    pm.set_local_dict('seg_residential', residential_params)
    
    # Test: Get parameters for each segment
    main1_vmax = pm.get('seg_main_1', 'vmax_arterial')
    main2_vmax = pm.get('seg_main_2', 'vmax_arterial')
    residential_vmax = pm.get('seg_residential', 'vmax_residential')
    
    print(f"\n🚗 Arterial seg_main_1: vmax = {main1_vmax:.2f} m/s (50 km/h)")
    print(f"🚗 Arterial seg_main_2: vmax = {main2_vmax:.2f} m/s (50 km/h)")
    print(f"🏘️  Residential: vmax = {residential_vmax:.2f} m/s (20 km/h)")
    
    # Verify speed ratio
    speed_ratio = main1_vmax / residential_vmax
    print(f"\n📊 Speed Ratio (Arterial/Residential): {speed_ratio:.2f}x")
    
    assert main1_vmax == 13.89
    assert main2_vmax == 13.89
    assert residential_vmax == 5.56
    assert abs(speed_ratio - 2.5) < 0.01, "Speed ratio should be ~2.5"
    
    print("✅ PASS: Heterogeneous network configured correctly")


def test_clear_local():
    """Test clearing local overrides."""
    print("\n" + "="*60)
    print("TEST 7: Clear Local Overrides")
    print("="*60)
    
    # Setup
    global_params = ModelParameters()
    pm = ParameterManager(global_params)
    
    pm.set_local_dict('seg_test', {
        'vmax_test': 13.89,
        'tau_c': 1.0
    })
    
    # Test: Clear specific parameter
    pm.clear_local('seg_test', 'vmax_test')
    assert not pm.has_local('seg_test', 'vmax_test'), "vmax_test should be cleared"
    assert pm.has_local('seg_test', 'tau_c'), "tau_c should remain"
    print("✅ Specific parameter cleared")
    
    # Test: Clear all overrides
    pm.clear_local('seg_test')
    assert not pm.has_local('seg_test'), "All overrides should be cleared"
    print("✅ All overrides cleared")


def test_summary():
    """Test parameter manager summary."""
    print("\n" + "="*60)
    print("TEST 8: Parameter Manager Summary")
    print("="*60)
    
    # Setup heterogeneous network
    global_params = ModelParameters()
    pm = ParameterManager(global_params)
    
    pm.set_local_dict('seg_arterial_1', {'vmax_arterial': 13.89, 'vmax_moto': 15.28, 'tau_c': 1.0})
    pm.set_local_dict('seg_arterial_2', {'vmax_arterial': 13.89, 'vmax_moto': 15.28})
    pm.set_local_dict('seg_residential', {'vmax_residential': 5.56, 'tau_c': 2.0})
    
    # Get summary
    summary = pm.summary()
    print(f"\n{summary}")
    
    segments = pm.list_segments_with_overrides()
    print(f"\n📋 Segments with overrides: {segments}")
    
    assert len(segments) == 3, "Should have 3 segments with overrides"
    print("✅ PASS: Summary generated correctly")


def run_all_tests():
    """Run all ParameterManager tests."""
    print("\n" + "="*70)
    print("  PARAMETER MANAGER TEST SUITE")
    print("  Phase 6 - Jour 2: Heterogeneous Network Parameters")
    print("="*70)
    
    tests = [
        test_basic_get_global,
        test_set_and_get_local,
        test_set_local_dict,
        test_get_all,
        test_has_local,
        test_heterogeneous_network,
        test_clear_local,
        test_summary
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ FAIL: {test.__name__}")
            print(f"   Error: {e}")
            failed += 1
    
    # Final summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! ParameterManager is ready for integration.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review and fix issues.")
    
    print("="*70)


if __name__ == '__main__':
    run_all_tests()
