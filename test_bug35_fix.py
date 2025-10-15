#!/usr/bin/env python3
"""
Test Bug #35 Fix: Verify that velocities now relax to equilibrium

This test:
1. Creates a simple scenario with high density inflow
2. Runs simulation with road quality properly loaded
3. Verifies velocities decrease as density increases
4. Confirms queue detection triggers (v < 5 m/s)
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from arz_model.simulation.runner import SimulationRunner
from arz_model.core.parameters import ModelParameters

def create_test_scenario():
    """Create a minimal test scenario YAML config"""
    test_config = {
        'scenario_name': 'bug35_fix_test',
        'N': 50,
        'xmin': 0.0,
        'xmax': 500.0,
        't_final': 30.0,
        'output_dt': 5.0,
        'CFL': 0.4,
        
        # ✅ CRITICAL: Road quality must be defined
        'road': {
            'quality_type': 'uniform',
            'quality_value': 2  # Good quality road (Vmax_m ≈ 19.4 m/s)
        },
        
        # Initial condition: Low density
        'initial_conditions': {
            'type': 'uniform',
            'state': [10.0, 15.0, 5.0, 15.0]  # [rho_m (veh/km), w_m (m/s), rho_c (veh/km), w_c (m/s)]
        },
        
        # Heavy inflow to build up density
        'boundary_conditions': {
            'left': {
                'type': 'inflow',
                'state': [200.0, 3.0, 100.0, 3.0]  # High density, equilibrium speed
            },
            'right': {
                'type': 'outflow'
            }
        }
    }
    
    # Write to temporary file
    import yaml
    config_path = Path(__file__).parent / 'test_bug35_scenario.yml'
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    return str(config_path)


def main():
    """Run test and verify Bug #35 is fixed"""
    
    print("="*80)
    print("BUG #35 FIX VERIFICATION TEST")
    print("="*80)
    
    # Create test scenario
    print("\n📝 Creating test scenario...")
    config_path = create_test_scenario()
    print(f"   Config: {config_path}")
    
    # Run simulation
    print("\n▶️  Running simulation...")
    try:
        runner = SimulationRunner(
            scenario_config_path=config_path,
            base_config_path='arz_model/config/config_base.yml',
            device='cpu',
            quiet=False
        )
        
        # Check road quality was loaded
        if runner.grid.road_quality is None:
            print("❌ FAIL: Road quality not loaded!")
            return 1
        else:
            print(f"✅ Road quality loaded: {np.unique(runner.grid.road_quality)}")
        
        # Run simulation
        times, states = runner.run()
        
        print(f"\n✅ Simulation completed: {len(times)} timesteps")
        
    except ValueError as e:
        if "Road quality array not loaded" in str(e):
            print(f"❌ FAIL: Road quality loading failed!")
            print(f"   Error: {e}")
            return 1
        else:
            raise
    
    # Analyze results
    print("\n📊 Analyzing velocity evolution...")
    
    # Extract velocities at first and last timesteps
    U_initial = states[0]  # (4, N_physical)
    U_final = states[-1]
    
    # Calculate physical velocities
    params = runner.params
    from arz_model.core.physics import calculate_pressure, calculate_physical_velocity
    
    # Initial state
    rho_m_0 = U_initial[0]
    w_m_0 = U_initial[1]
    rho_c_0 = U_initial[2]
    w_c_0 = U_initial[3]
    p_m_0, p_c_0 = calculate_pressure(
        rho_m_0, rho_c_0, params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c
    )
    v_m_0, v_c_0 = calculate_physical_velocity(w_m_0, w_c_0, p_m_0, p_c_0)
    
    # Final state
    rho_m_f = U_final[0]
    w_m_f = U_final[1]
    rho_c_f = U_final[2]
    w_c_f = U_final[3]
    p_m_f, p_c_f = calculate_pressure(
        rho_m_f, rho_c_f, params.alpha, params.rho_jam, params.epsilon,
        params.K_m, params.gamma_m, params.K_c, params.gamma_c
    )
    v_m_f, v_c_f = calculate_physical_velocity(w_m_f, w_c_f, p_m_f, p_c_f)
    
    # Check upstream region (cells 0-5) where inflow directly impacts
    upstream_cells = slice(0, 5)
    
    rho_upstream_initial = np.mean(rho_m_0[upstream_cells] + rho_c_0[upstream_cells])
    rho_upstream_final = np.mean(rho_m_f[upstream_cells] + rho_c_f[upstream_cells])
    v_upstream_initial = np.mean(v_m_0[upstream_cells])
    v_upstream_final = np.mean(v_m_f[upstream_cells])
    
    print(f"\n   Upstream region (cells 0-5):")
    print(f"   Initial: ρ={rho_upstream_initial*1000:.1f} veh/km, v={v_upstream_initial:.2f} m/s")
    print(f"   Final:   ρ={rho_upstream_final*1000:.1f} veh/km, v={v_upstream_final:.2f} m/s")
    
    # Also check cell-by-cell for first 5 cells
    print(f"\n   Cell-by-cell analysis (first 5 cells):")
    for i in range(5):
        print(f"   Cell {i}: ρ={rho_m_f[i]*1000:.1f} veh/km, v={v_m_f[i]:.2f} m/s")
    
    # Verification criteria
    print("\n🔍 Verification:")
    
    checks_passed = 0
    checks_total = 3
    
    # Check 1: Density should increase
    if rho_upstream_final > rho_upstream_initial * 1.5:
        print(f"   ✅ CHECK 1: Density increased (Δρ = +{(rho_upstream_final - rho_upstream_initial)*1000:.1f} veh/km)")
        checks_passed += 1
    else:
        print(f"   ❌ CHECK 1: Density didn't increase enough")
    
    # Check 2: Velocity should decrease
    velocity_decrease = v_upstream_initial - v_upstream_final
    if velocity_decrease > 3.0:  # At least 3 m/s decrease
        print(f"   ✅ CHECK 2: Velocity decreased (Δv = -{velocity_decrease:.2f} m/s)")
        checks_passed += 1
    else:
        print(f"   ❌ CHECK 2: Velocity didn't decrease enough (Δv = -{velocity_decrease:.2f} m/s, need > 3.0 m/s)")
    
    # Check 3: Queue should form (v < 5 m/s somewhere)
    min_velocity = np.min(v_m_f[upstream_cells])
    if min_velocity < 5.0:
        print(f"   ✅ CHECK 3: Queue detected (min v = {min_velocity:.2f} m/s < 5.0 m/s)")
        checks_passed += 1
    else:
        print(f"   ❌ CHECK 3: No queue detected (min v = {min_velocity:.2f} m/s >= 5.0 m/s)")
    
    # Final verdict
    print("\n" + "="*80)
    if checks_passed == checks_total:
        print("✅ ALL CHECKS PASSED - BUG #35 IS FIXED!")
        print("   Velocities now relax to equilibrium as expected")
        print("   Queue detection should work for RL training")
        print("="*80)
        return 0
    else:
        print(f"❌ SOME CHECKS FAILED ({checks_passed}/{checks_total} passed)")
        print("   Bug #35 may not be fully resolved")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
