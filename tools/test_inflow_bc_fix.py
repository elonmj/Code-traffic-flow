#!/usr/bin/env python
"""
Quick local test to verify inflow BC fix maintains traffic instead of draining.
Tests CPU mode for faster execution.
"""
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from arz_model.core.parameters import ModelParameters
from arz_model.grid.grid1d import Grid1D
from arz_model.simulation.runner import SimulationRunner

def test_inflow_bc_fix():
    """Test that inflow BC now maintains traffic with proper momentum."""
    print("=" * 80)
    print("TESTING INFLOW BC FIX - Quick Local Validation")
    print("=" * 80)
    
    # Minimal config for fast test
    config = {
        'simulation': {
            'L': 1000.0,  # 1km domain
            'N': 100,  # 100 cells
            'T': 120.0,  # 2 minutes simulation
            'CFL': 0.5,
            'save_interval': 20.0
        },
        'physics': {
            'alpha': 0.5,
            'rho_jam': 0.18,
            'v_max_m': 30.0,
            'v_max_c': 25.0,
            'epsilon': 1e-8,
            'K_m': 1.0,
            'gamma_m': 2.0,
            'K_c': 1.0,
            'gamma_c': 2.0
        },
        'initial_conditions': {
            'type': 'riemann',
            'U_L': [0.1, 15.0, 0.12, 12.0],  # High density with momentum
            'U_R': [0.03, 25.0, 0.04, 20.0],  # Low density free flow
            'split_pos': 500.0
        },
        'boundary_conditions': {
            'left': {'type': 'outflow'},  # Start with outflow
            'right': {'type': 'outflow'}
        },
        'device': 'cpu',
        'output_dir': 'test_bc_fix_output'
    }
    
    # Create params and runner
    params = ModelParameters(config)
    runner = SimulationRunner(params, quiet=False)  # Enable logging
    
    print("\n[TEST 1] Initial simulation with outflow BC (should drain)")
    print("-" * 80)
    runner.run_until(t=30.0)
    state = runner.get_current_state()
    rho_m_mean_initial = np.mean(state[0, :])
    print(f"  After 30s with outflow BC: mean rho_m = {rho_m_mean_initial:.6f}")
    
    print("\n[TEST 2] Switch to inflow BC with full state (should inject traffic)")
    print("-" * 80)
    # Switch left BC to inflow with the high-density state
    runner.current_bc_params['left'] = {
        'type': 'inflow',
        'state': [0.1, 15.0, 0.12, 12.0]  # High density WITH momentum
    }
    print("[BC SWITCH] Left → inflow [0.1, 15.0, 0.12, 12.0]")
    
    # Continue simulation
    runner.run_until(t=60.0)
    state = runner.get_current_state()
    rho_m_mean_after_inflow = np.mean(state[0, :])
    print(f"  After 30s with inflow BC: mean rho_m = {rho_m_mean_after_inflow:.6f}")
    
    # Check if traffic increased (fix working)
    improvement = (rho_m_mean_after_inflow - rho_m_mean_initial) / (rho_m_mean_initial + 1e-8)
    print(f"\n[RESULT] Density change: {improvement:+.1%}")
    
    if improvement > 0.1:  # At least 10% increase
        print("✅ PASS: Inflow BC successfully injected traffic!")
        print("   → Fix is working: w_m and w_c are now imposed, not extrapolated")
        return True
    else:
        print("❌ FAIL: Traffic did not increase significantly")
        print("   → Fix may not be working correctly")
        return False

if __name__ == '__main__':
    try:
        success = test_inflow_bc_fix()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
