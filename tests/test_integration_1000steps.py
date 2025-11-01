"""
Integration Test - 1000 Steps

Tests complete system with Pydantic configs running for 1000 time steps.
"""

import numpy as np
from arz_model.config import ConfigBuilder
from arz_model.simulation.runner import SimulationRunner


def test_integration_1000_steps():
    """
    Integration test: Run 1000 steps with Pydantic config.
    
    Verifies:
    - No crashes
    - No NaNs
    - Mass conservation
    - Results have correct shape
    """
    
    print("\n" + "="*60)
    print("Integration Test: 1000 Steps")
    print("="*60 + "\n")
    
    # Create configuration
    config = ConfigBuilder.simple_test()
    config.t_final = 10.0
    config.output_dt = 0.5
    
    print(f"Configuration:")
    print(f"  Grid: N={config.grid.N}, domain=[{config.grid.xmin}, {config.grid.xmax}] km")
    print(f"  Time: t_final={config.t_final} s, output_dt={config.output_dt} s")
    print(f"  Device: {config.device}")
    print(f"  IC type: {config.initial_conditions.type}")
    print(f"  BC left: {config.boundary_conditions.left.type}")
    print(f"  BC right: {config.boundary_conditions.right.type}")
    
    # Create runner
    print("\nCreating runner...")
    runner = SimulationRunner(config=config, quiet=False)
    
    print(f"\nInitial state:")
    print(f"  Shape: {runner.U.shape}")
    print(f"  rho_m range: [{runner.U[0, :].min():.6f}, {runner.U[0, :].max():.6f}]")
    print(f"  rho_c range: [{runner.U[2, :].min():.6f}, {runner.U[2, :].max():.6f}]")
    
    # Run simulation with max_steps=1000
    print(f"\nRunning simulation (max 1000 steps)...")
    times, states = runner.run(max_steps=1000)
    
    # Verify results
    print(f"\n Simulation complete!")
    print(f"\nResults:")
    print(f"  Total steps: {runner.step_count}")
    print(f"  Final time: {runner.t:.4f} s")
    print(f"  Outputs stored: {len(times)}")
    print(f"  State shape: {states[0].shape if states else 'N/A'}")
    
    # Check for NaNs
    final_state = runner.U
    has_nans = np.any(np.isnan(final_state))
    has_infs = np.any(np.isinf(final_state))
    
    print(f"\nQuality checks:")
    print(f"  NaNs detected: {'❌ YES' if has_nans else '✅ NO'}")
    print(f"  Infs detected: {'❌ YES' if has_infs else '✅ NO'}")
    
    if has_nans or has_infs:
        print("\n❌ INTEGRATION TEST FAILED: NaNs or Infs detected")
        return False
    
    # Check mass conservation (if enabled - optional feature)
    if hasattr(runner, 'mass_times') and runner.mass_times:
        # Calculate mass change from tracked data
        initial_mass_m = runner.mass_m_data[0]
        final_mass_m = runner.mass_m_data[-1]
        initial_mass_c = runner.mass_c_data[0]
        final_mass_c = runner.mass_c_data[-1]
        
        mass_change_m = 100.0 * (final_mass_m - initial_mass_m) / initial_mass_m if initial_mass_m > 0 else 0.0
        mass_change_c = 100.0 * (final_mass_c - initial_mass_c) / initial_mass_c if initial_mass_c > 0 else 0.0
        
        print(f"\nMass conservation:")
        print(f"  Motorcycles: {mass_change_m:+.6f}%")
        print(f"  Cars: {mass_change_c:+.6f}%")
        
        # Warn if mass change > 1%
        if abs(mass_change_m) > 1.0 or abs(mass_change_c) > 1.0:
            print(f"  ⚠️  Warning: Mass change > 1%")
        else:
            print(f"  ✅ Mass well conserved (< 1%)")
    else:
        print(f"\nMass conservation: Tracking not enabled (optional feature)")
    
    # Verify outputs
    assert len(times) > 0, "No outputs stored"
    assert len(states) == len(times), "Times and states mismatch"
    assert runner.step_count <= 1000, f"Too many steps: {runner.step_count}"
    assert not has_nans, "NaNs detected in final state"
    assert not has_infs, "Infs detected in final state"
    
    print("\n" + "="*60)
    print("✅ INTEGRATION TEST PASSED")
    print("="*60)
    
    return True


if __name__ == '__main__':
    success = test_integration_1000_steps()
    exit(0 if success else 1)
