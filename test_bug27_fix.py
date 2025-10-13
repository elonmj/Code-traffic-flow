#!/usr/bin/env python3
"""
Quick test to verify Bug #27 fix: Control interval now 15s instead of 60s
This should show DIFFERENT results for baseline vs RL.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from validation_ch7.scripts.test_section_7_6_rl_performance import RLPerformanceValidationTest

def main():
    print("="*80)
    print("BUG #27 FIX VERIFICATION TEST")
    print("="*80)
    print()
    print("Testing configuration:")
    print("  - Control interval: 15s (was 60s)")
    print("  - Expected: Baseline and RL produce DIFFERENT results")
    print("  - Duration: 600s (10 minutes)")
    print("  - Scenario: traffic_light_control only")
    print()
    
    # Run quick test (minimal timesteps, short duration)
    test = RLPerformanceValidationTest(quick_test=True)
    
    print("[STEP 1/3] Training RL agent (minimal timesteps)...")
    test.train_rl_agent('traffic_light_control', total_timesteps=100, device='cpu')
    
    print("\n[STEP 2/3] Running baseline controller...")
    scenario_path = test._create_scenario_config('traffic_light_control')
    baseline_controller = test.BaselineController('traffic_light_control')
    baseline_states, _ = test.run_control_simulation(
        baseline_controller, 
        scenario_path,
        duration=600.0,  # 10 minutes
        device='cpu'
    )
    
    print("\n[STEP 3/3] Running RL controller...")
    model_path = test.models_dir / "rl_agent_traffic_light_control.zip"
    rl_controller = test.RLController('traffic_light_control', model_path)
    rl_states, _ = test.run_control_simulation(
        rl_controller, 
        scenario_path,
        duration=600.0,  # 10 minutes
        device='cpu'
    )
    
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    # Calculate metrics
    baseline_perf = test.evaluate_traffic_performance(baseline_states, 'traffic_light_control')
    rl_perf = test.evaluate_traffic_performance(rl_states, 'traffic_light_control')
    
    print(f"\nBaseline Performance:")
    print(f"  Flow: {baseline_perf['total_flow']:.6f}")
    print(f"  Efficiency: {baseline_perf['efficiency']:.6f}")
    print(f"  Delay: {baseline_perf['delay']:.2f}s")
    
    print(f"\nRL Performance:")
    print(f"  Flow: {rl_perf['total_flow']:.6f}")
    print(f"  Efficiency: {rl_perf['efficiency']:.6f}")
    print(f"  Delay: {rl_perf['delay']:.2f}s")
    
    # Check if results are different
    flow_diff = abs(baseline_perf['total_flow'] - rl_perf['total_flow'])
    eff_diff = abs(baseline_perf['efficiency'] - rl_perf['efficiency'])
    delay_diff = abs(baseline_perf['delay'] - rl_perf['delay'])
    
    print(f"\nDifferences:")
    print(f"  Flow: {flow_diff:.6f}")
    print(f"  Efficiency: {eff_diff:.6f}")
    print(f"  Delay: {delay_diff:.2f}s")
    
    # Verdict
    print("\n" + "="*80)
    if flow_diff > 0.001 or eff_diff > 0.001 or delay_diff > 0.1:
        print("✅ BUG #27 FIX VERIFIED: Baseline and RL produce DIFFERENT results!")
        print("   Fix is working correctly. Ready for full Kaggle validation.")
        return 0
    else:
        print("❌ BUG #27 FIX FAILED: Results are still identical!")
        print("   Further investigation needed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
