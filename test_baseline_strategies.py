#!/usr/bin/env python3
"""
🔍 CRITICAL TEST: Verify that baseline metrics CHANGE with different control strategies

This test:
1. Runs baseline with 'RED_ONLY' strategy (very poor)
2. Runs baseline with 'green_only' strategy (very good)
3. Compares flow metrics
4. If identical → Results are HARDCODED
5. If different → Results are REAL and responsive to control
"""

import sys
import os
import numpy as np
from pathlib import Path
import shutil
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from Code_RL.src.utils.config import create_scenario_config_with_lagos_data
from validation_ch7.scripts.test_section_7_6_rl_performance import RLPerformanceValidationTest

def test_baseline_strategies():
    """Test that baseline metrics change with different strategies"""
    
    print("\n" + "="*80)
    print("CRITICAL TEST: BASELINE METRICS RESPONSIVENESS")
    print("="*80)
    
    print("\nObjective: Verify that flow metrics CHANGE when using different control strategies")
    print("Expected: RED_ONLY flow << GREEN_ONLY flow")
    print("\nIf results are identical → Results are HARDCODED")
    print("If results differ → Results are REAL\n")
    
    # Step 1: Clean cache
    print("STEP 1: CLEANING CACHE")
    print("-" * 80)
    cache_dir = Path(project_root) / "validation_ch7/cache/section_7_6"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"✅ Cache removed")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 2: Generate test scenario
    print("\nSTEP 2: GENERATING TEST SCENARIO")
    print("-" * 80)
    scenarios_dir = Path(project_root) / "validation_ch7/scenarios/rl_scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    
    scenario_config = scenarios_dir / "test_baseline_strategies_network.yml"
    print("Generating scenario...")
    create_scenario_config_with_lagos_data(
        'traffic_light_control',
        str(scenario_config),
        duration=1800,  # Short 30min
        domain_length=2000
    )
    print(f"✅ Generated {scenario_config}")
    
    # Step 3: Initialize test infrastructure
    print("\nSTEP 3: INITIALIZING TEST INFRASTRUCTURE")
    print("-" * 80)
    test_instance = RLPerformanceValidationTest(quick_test=True)
    test_instance.debug_logger.info("Starting baseline strategy comparison")
    
    # Step 4: Test different baseline strategies
    print("\nSTEP 4: TESTING BASELINE STRATEGIES")
    print("-" * 80)
    
    results = {}
    strategies = ['red_only', 'alternating', 'fixed_time_optimal', 'green_only']
    
    for strategy in strategies:
        print(f"\n▶️  Testing strategy: {strategy}")
        print(f"   {'═'*70}")
        
        # Clean cache for this test (to ensure fresh simulation)
        cache_dir = Path(project_root) / "validation_ch7/cache/section_7_6"
        baseline_cache_file = cache_dir / "traffic_light_control_baseline_cache.pkl"
        if baseline_cache_file.exists():
            baseline_cache_file.unlink()
            print(f"   ✅ Old cache removed")
        
        try:
            # Create baseline controller with this strategy
            baseline_controller = test_instance.BaselineController(
                'traffic_light_control', 
                strategy=strategy
            )
            
            # Run simulation
            print(f"   Running simulation ({scenario_config.name})...")
            start_time = time.time()
            
            states, actions = test_instance.run_control_simulation(
                baseline_controller,
                scenario_config,
                duration=1800,  # 30min
                control_interval=15.0,
                device='cpu',  # Use CPU for speed
                controller_type=f'BASELINE_{strategy.upper()}'
            )
            
            elapsed = time.time() - start_time
            print(f"   ✅ Simulation complete ({elapsed:.1f}s, {len(states) if states else 0} states)")
            
            if states:
                # Evaluate performance
                states_copy = [state.copy() for state in states]
                perf = test_instance.evaluate_traffic_performance(states_copy, 'traffic_light_control', scenario_config)
                results[strategy] = perf
                
                print(f"   📊 METRICS:")
                print(f"      Flow: {perf['total_flow']:.6f}")
                print(f"      Efficiency: {perf['efficiency']:.6f}")
                print(f"      Delay: {perf['delay']:.2f}s")
                print(f"      Speed: {perf['avg_speed']:.2f} m/s")
            else:
                print(f"   ❌ Simulation failed (no states returned)")
                results[strategy] = None
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results[strategy] = None
    
    # Step 5: Comparative analysis
    print("\n\nSTEP 5: COMPARATIVE ANALYSIS")
    print("="*80)
    
    # Extract valid results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) < 2:
        print("❌ NOT ENOUGH VALID RESULTS - Cannot compare")
        return
    
    flows = {strat: valid_results[strat]['total_flow'] for strat in valid_results}
    efficiencies = {strat: valid_results[strat]['efficiency'] for strat in valid_results}
    speeds = {strat: valid_results[strat]['avg_speed'] for strat in valid_results}
    delays = {strat: valid_results[strat]['delay'] for strat in valid_results}
    
    print("\n📊 TOTAL FLOW:")
    for strat, flow in sorted(flows.items(), key=lambda x: x[1], reverse=True):
        print(f"   {strat:20s}: {flow:.6f}")
    
    flow_range = max(flows.values()) - min(flows.values())
    flow_var = np.var([v for v in flows.values()])
    print(f"\n   Range: {flow_range:.6f}")
    print(f"   Variance: {flow_var:.6e}")
    
    print("\n📊 EFFICIENCY:")
    for strat, eff in sorted(efficiencies.items(), key=lambda x: x[1], reverse=True):
        print(f"   {strat:20s}: {eff:.6f}")
    
    eff_range = max(efficiencies.values()) - min(efficiencies.values())
    eff_var = np.var([v for v in efficiencies.values()])
    print(f"\n   Range: {eff_range:.6f}")
    print(f"   Variance: {eff_var:.6e}")
    
    print("\n📊 AVERAGE SPEED:")
    for strat, speed in sorted(speeds.items(), key=lambda x: x[1], reverse=True):
        print(f"   {strat:20s}: {speed:.2f} m/s ({speed*3.6:.1f} km/h)")
    
    speed_range = max(speeds.values()) - min(speeds.values())
    speed_var = np.var([v for v in speeds.values()])
    print(f"\n   Range: {speed_range:.2f} m/s")
    print(f"   Variance: {speed_var:.6e}")
    
    # Final verdict
    print("\n" + "="*80)
    print("VERDICT:")
    print("="*80)
    
    THRESHOLD = 1e-10
    all_identical = (flow_range < THRESHOLD) and (eff_range < THRESHOLD) and (speed_range < 0.001)
    
    if all_identical:
        print("🚨 CRITICAL: All metrics are IDENTICAL across strategies!")
        print("   → Results are HARDCODED")
        print("   → Simulation is NOT actually running with different controls")
    else:
        print("✅ EXCELLENT: Metrics DIFFER across strategies!")
        print(f"   → Flow improvement (GREEN vs RED): {(flows.get('green_only', 0) - flows.get('red_only', 0))*100:.1f}%")
        print(f"   → Speed improvement (GREEN vs RED): {(speeds.get('green_only', 0) - speeds.get('red_only', 0))*3.6:.1f} km/h")
        print(f"   → Results are REAL and responsive to control strategies")
    
    print("="*80 + "\n")

if __name__ == '__main__':
    test_baseline_strategies()
