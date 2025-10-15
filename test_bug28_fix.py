"""
Test Bug #28 fix - Phase change detection in reward function
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Code_RL', 'src'))

import torch
from env.traffic_signal_env_direct import TrafficSignalEnvDirect

def test_phase_change_detection():
    """Test that phase changes are correctly detected after Bug #28 fix"""
    
    print("=" * 80)
    print("TEST: Phase Change Detection (Bug #28 Fix Validation)")
    print("=" * 80)
    
    # Initialize environment
    scenario_path = "validation_output/results/joselonm_arz-validation-76rlperformance-xrld/section_7_6_rl_performance/data/scenarios/traffic_light_control.yml"
    
    env = TrafficSignalEnvDirect(
        scenario_config_path=scenario_path,
        decision_interval=15.0,
        device='cpu'
    )
    
    print("\n✅ Environment initialized")
    print(f"   Initial phase: {env.current_phase}")
    
    # Reset environment
    obs = env.reset()
    print(f"\n🔄 Environment reset, phase: {env.current_phase}")
    
    # Test scenarios
    test_cases = [
        {"name": "RED → RED (no change)", "action": 0, "expected_penalty": False},
        {"name": "RED → GREEN (change)", "action": 1, "expected_penalty": True},
        {"name": "GREEN → GREEN (no change)", "action": 1, "expected_penalty": False},
        {"name": "GREEN → RED (change)", "action": 0, "expected_penalty": True},
        {"name": "RED → RED again", "action": 0, "expected_penalty": False},
    ]
    
    print("\n" + "=" * 80)
    print("Running test cases...")
    print("=" * 80)
    
    all_passed = True
    
    for i, test in enumerate(test_cases, 1):
        prev_phase = env.current_phase
        obs, reward, done, truncated, info = env.step(test["action"])
        current_phase = env.current_phase
        
        # Phase change detection
        actual_change = (current_phase != prev_phase)
        expected_penalty = test["expected_penalty"]
        
        # Reward should include -0.1 penalty if phase changed
        has_penalty = (reward <= -0.05)  # Allow some tolerance for queue component
        
        status = "✅" if (has_penalty == expected_penalty) else "❌"
        
        print(f"\n{status} Test {i}: {test['name']}")
        print(f"   Phase: {prev_phase} → {current_phase}")
        print(f"   Action: {test['action']}")
        print(f"   Actual change: {actual_change}")
        print(f"   Expected penalty: {expected_penalty}")
        print(f"   Has penalty: {has_penalty}")
        print(f"   Reward: {reward:.6f}")
        
        if has_penalty != expected_penalty:
            all_passed = False
            print(f"   ❌ FAILED: Expected penalty={expected_penalty}, got penalty={has_penalty}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Bug #28 fix validated!")
    else:
        print("❌ SOME TESTS FAILED - Bug #28 fix needs review")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = test_phase_change_detection()
    sys.exit(0 if success else 1)
