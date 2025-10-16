#!/usr/bin/env python3
"""
Debug test for Bug #36: Trace current_bc_params through GPU call stack
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from arz_model.grid.grid1d import Grid1D
from arz_model.core.parameters import ModelParameters
from arz_model.simulation.runner import SimulationRunner
from arz_model.simulation.initial_conditions import uniform_state

def test_bug36_debug_logging():
    """Run a minimal simulation with debug logging enabled"""
    
    print("\n" + "="*70)
    print("BUG #36 DEBUG TRACE - Parameter Propagation Test")
    print("="*70 + "\n")
    
    # 1. Create grid
    num_cells = 20  # Small for quick testing
    x_min, x_max = 0.0, 100.0
    num_ghost_cells = 2
    grid = Grid1D(num_cells, x_min, x_max, num_ghost_cells)
    
    # 2. Create parameters - with DYNAMIC inflow BC capability
    params = ModelParameters()
    params.device = 'gpu'
    params.spatial_scheme = 'weno5'
    params.time_scheme = 'euler'
    
    # Set STATIC initial BC (will be overridden)
    params.boundary_conditions = {
        'left': {'type': 'inflow', 'state': [0.05, 0.1, 0.01, 0.1]},  # STATIC: 0.05 veh/m
        'right': {'type': 'outflow'}
    }
    
    # 3. Create initial state
    rho_m_init = 0.02
    w_m_init = 0.1
    rho_c_init = 0.01
    w_c_init = 0.1
    U = uniform_state(grid, rho_m_init, w_m_init, rho_c_init, w_c_init)
    
    # 4. Create runner
    runner = SimulationRunner(grid, params, quiet=False)
    runner.initialize(U, scenario_yaml=None)
    
    print("\n" + "-"*70)
    print("PHASE 1: Initial state with STATIC BC")
    print("-"*70)
    print(f"Static BC left inflow state: {params.boundary_conditions['left']['state']}")
    print(f"Running 1 timestep with static BC...\n")
    
    # Run one step with static BC (should use 0.05 veh/m)
    try:
        runner.advance(1)
    except Exception as e:
        print(f"Error in Phase 1: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "-"*70)
    print("PHASE 2: Update to DYNAMIC BC and run")
    print("-"*70)
    
    # Now update current_bc_params DYNAMICALLY (simulating traffic signal change)
    new_bc_params = {
        'left': {'type': 'inflow', 'state': [0.3, 0.3, 0.05, 0.3]},  # DYNAMIC: 0.3 veh/m (6x increase)
        'right': {'type': 'outflow'}
    }
    
    print(f"Setting current_bc_params to dynamic BC with inflow 0.3 veh/m")
    runner.current_bc_params = new_bc_params
    print(f"runner.current_bc_params: {runner.current_bc_params}\n")
    print(f"Running 2 more timesteps with DYNAMIC BC...\n")
    
    # Run steps with dynamic BC
    try:
        runner.advance(2)
    except Exception as e:
        print(f"Error in Phase 2: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "-"*70)
    print("PHASE 3: Debug Log Analysis")
    print("-"*70)
    print("\nDEBUG LOG SUMMARY:")
    print("1. If you see '[DEBUG_WENO_GPU_NATIVE] current_bc_params IS NONE!' - Parameter lost before GPU!")
    print("2. If you see '[DEBUG_BC_DISPATCHER] Using current_bc_params (dynamic)' - Parameter reached dispatcher")
    print("3. If you see '[DEBUG_BC_GPU] inflow_L: [0.3, ...' - Parameter reached GPU kernel")
    print("\nExpected upstream density after Phase 2:")
    print("  - WITHOUT fix: ~0.05 veh/m (static BC still used)")
    print("  - WITH fix: ~0.15-0.20 veh/m (dynamic BC applied, 0.3 at inflow)")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        test_bug36_debug_logging()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
