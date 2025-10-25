#!/usr/bin/env python3
"""
Debug script to check if RL results are truly dynamic or hardcoded.

This script:
1. Runs baseline simulation with FIXED control (RED always)
2. Runs baseline simulation with DIFFERENT control (GREEN always)
3. Compares flow metrics to see if they differ
4. If results are same → results are HARDCODED
5. If results differ → results are REAL
"""

import sys
import os
import numpy as np
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import ARZ simulator
from arz_model.simulation.runner import SimulationRunner

# Import Code_RL utils
code_rl_path = project_root / "Code_RL"
sys.path.append(str(code_rl_path))
from Code_RL.src.utils.config import create_scenario_config_with_lagos_data

# Import test utilities
from validation_ch7.scripts.test_section_7_6_rl_performance import RLPerformanceValidationTest

def run_test_scenario(scenario_name, scenario_config_path, control_strategy):
    """
    Run a scenario with specific control strategy.
    
    control_strategy: 'RED_ONLY', 'GREEN_ONLY', or 'ALTERNATING'
    """
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"CONTROL STRATEGY: {control_strategy}")
    print(f"{'='*80}\n")
    
    # Load scenario config
    with open(scenario_config_path, 'r') as f:
        scenario_config = yaml.safe_load(f)
    
    domain_length = scenario_config['xmax'] - scenario_config['xmin']
    duration = 3600  # 1 hour
    decision_interval = 15.0  # 15s (from Bug #27)
    
    print(f"Domain length: {domain_length}m")
    print(f"Duration: {duration}s")
    print(f"Decision interval: {decision_interval}s\n")
    
    # Initialize simulator
    base_config_path = Path(project_root) / "arz_model/config/config_base.yml"
    runner = SimulationRunner(
        scenario_config_path=str(scenario_config_path),
        base_config_path=str(base_config_path),
        quiet=True,
        device='cpu'
    )
    
    # Collect states over time
    states = []
    current_phase = 0
    timesteps = int(duration / decision_interval)
    
    for step in range(timesteps):
        # Apply control strategy
        if control_strategy == 'RED_ONLY':
            phase = 0  # RED (stop)
        elif control_strategy == 'GREEN_ONLY':
            phase = 1  # GREEN (go)
        elif control_strategy == 'ALTERNATING':
            phase = step % 2  # Alternate every decision_interval
        
        # Set traffic signal
        runner.set_traffic_signal_state('left', phase_id=phase)
        
        # Run simulation
        target_time = runner.t + decision_interval
        runner.run(t_final=target_time, output_dt=decision_interval)
        
        # Extract state
        # State format: [rho_m, w_m, rho_c, w_c, ...]
        state = runner.get_full_state()
        states.append(state)
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1:3d}/{timesteps}: t={runner.t:7.1f}s | Phase={phase}")
    
    # Evaluate performance
    test_instance = RLPerformanceValidationTest(quick_test=True)
    performance = test_instance.evaluate_traffic_performance(states, 'traffic_light_control')
    
    print(f"\nFINAL METRICS:")
    print(f"  Total Flow: {performance['total_flow']:.6f}")
    print(f"  Avg Speed: {performance['avg_speed']:.2f} m/s ({performance['avg_speed']*3.6:.1f} km/h)")
    print(f"  Efficiency: {performance['efficiency']:.6f}")
    print(f"  Delay: {performance['delay']:.2f}s")
    print(f"  Throughput: {performance['throughput']:.2f}")
    
    return performance

def main():
    print("\n" + "="*80)
    print("DEBUG: RL RESULTS VALIDATION")
    print("="*80)
    print("\nObjective: Verify if traffic flow metrics are REAL or HARDCODED")
    print("Method: Run 3 different control strategies and compare flows\n")
    
    # Generate test scenario
    scenarios_dir = Path(project_root) / "validation_ch7/scenarios/rl_scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    
    scenario_config = scenarios_dir / "debug_scenario_network.yml"
    control_config = scenarios_dir / "debug_scenario_network_traffic_control.yml"
    
    print(f"Generating test scenario...")
    create_scenario_config_with_lagos_data(
        'traffic_light_control',
        str(scenario_config),
        duration=3600,  # Short 1-hour test
        domain_length=2000
    )
    print(f"✅ Generated {scenario_config}")
    
    # Run tests with different control strategies
    results = {}
    
    print("\n" + "="*80)
    print("TEST 1: RED_ONLY (traffic light always RED = STOP)")
    print("Expected: Low flow, vehicles backed up")
    print("="*80)
    results['RED_ONLY'] = run_test_scenario(
        "Signalized Network (RED Always)",
        scenario_config,
        "RED_ONLY"
    )
    
    print("\n" + "="*80)
    print("TEST 2: GREEN_ONLY (traffic light always GREEN = GO)")
    print("Expected: High flow, minimal delays")
    print("="*80)
    results['GREEN_ONLY'] = run_test_scenario(
        "Signalized Network (GREEN Always)",
        scenario_config,
        "GREEN_ONLY"
    )
    
    print("\n" + "="*80)
    print("TEST 3: ALTERNATING (traffic light alternates RED/GREEN)")
    print("Expected: Intermediate flow")
    print("="*80)
    results['ALTERNATING'] = run_test_scenario(
        "Signalized Network (Alternating)",
        scenario_config,
        "ALTERNATING"
    )
    
    # Compare results
    print("\n\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80 + "\n")
    
    # Extract metrics
    flows = {strategy: results[strategy]['total_flow'] for strategy in results}
    efficiencies = {strategy: results[strategy]['efficiency'] for strategy in results}
    delays = {strategy: results[strategy]['delay'] for strategy in results}
    
    print("TOTAL FLOW:")
    for strategy in ['RED_ONLY', 'GREEN_ONLY', 'ALTERNATING']:
        print(f"  {strategy:12s}: {flows[strategy]:.6f}")
    
    flow_variance = np.var([flows[s] for s in flows])
    flow_range = max(flows.values()) - min(flows.values())
    
    print(f"\n  Variance: {flow_variance:.6e}")
    print(f"  Range: {flow_range:.6f}")
    
    if flow_range < 1e-10:
        print("  ❌ HARDCODED: Flows are IDENTICAL across strategies!")
        print("     → Results are HARDCODED or not using actual simulation data")
    else:
        print(f"  ✅ REAL: Flows differ by {flow_range*100:.2f}% - Results are genuine!")
    
    print("\nEFFICIENCY:")
    for strategy in ['RED_ONLY', 'GREEN_ONLY', 'ALTERNATING']:
        print(f"  {strategy:12s}: {efficiencies[strategy]:.6f}")
    
    eff_variance = np.var([efficiencies[s] for s in efficiencies])
    eff_range = max(efficiencies.values()) - min(efficiencies.values())
    
    print(f"\n  Variance: {eff_variance:.6e}")
    print(f"  Range: {eff_range:.6f}")
    
    if eff_range < 1e-10:
        print("  ❌ HARDCODED: Efficiencies are IDENTICAL!")
    else:
        print(f"  ✅ REAL: Efficiencies differ by {eff_range*100:.2f}%!")
    
    print("\nDELAY:")
    for strategy in ['RED_ONLY', 'GREEN_ONLY', 'ALTERNATING']:
        print(f"  {strategy:12s}: {delays[strategy]:7.2f}s")
    
    delay_variance = np.var([delays[s] for s in delays])
    delay_range = max(delays.values()) - min(delays.values())
    
    print(f"\n  Variance: {delay_variance:.6e}")
    print(f"  Range: {delay_range:.2f}s")
    
    if delay_range < 0.01:
        print("  ❌ HARDCODED: Delays are IDENTICAL!")
    else:
        print(f"  ✅ REAL: Delays differ by {delay_range:.2f}s!")
    
    # Final verdict
    print("\n" + "="*80)
    all_identical = (flow_range < 1e-10) and (eff_range < 1e-10) and (delay_range < 0.01)
    
    if all_identical:
        print("🚨 CRITICAL: All metrics are HARDCODED or NOT using simulation data!")
        print("   → Results are NOT reflecting actual control strategies")
        print("   → Issue is in evaluate_traffic_performance() or state extraction")
    else:
        print("✅ CONFIRMED: Metrics are REAL and responsive to control strategies!")
        print("   → GREEN_ONLY should have BEST flow and lowest delay")
        print("   → RED_ONLY should have WORST flow and highest delay")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
