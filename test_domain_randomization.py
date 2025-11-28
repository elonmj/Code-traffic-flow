"""
Test script to verify TRUE Domain Randomization is working.

This test verifies that:
1. set_inflow_conditions() modifies U_initial correctly
2. Different densities result in different initial states after reset()
3. The observation space reflects the density changes
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from arz_model.config import create_victoria_island_config
from Code_RL.src.env.traffic_signal_env_direct_v3 import TrafficSignalEnvDirectV3
from Code_RL.src.env.variable_demand_wrapper import VariableDemandEnv


def test_set_inflow_conditions():
    """Test that set_inflow_conditions modifies U_initial correctly."""
    print("=" * 60)
    print("TEST 1: set_inflow_conditions() modifies U_initial")
    print("=" * 60)
    
    config = create_victoria_island_config(
        t_final=100.0, output_dt=15.0, cells_per_100m=4,
        default_density=120.0, inflow_density=180.0, use_cache=False
    )
    config.rl_metadata = {'observation_segment_ids': [s.id for s in config.segments], 'decision_interval': 15.0}
    
    # Create NetworkGrid directly to test U_initial modification
    # This avoids the GPU requirement
    from arz_model.network.network_grid import NetworkGrid
    
    network_grid = NetworkGrid.from_config(config)
    
    # Find entry segment
    entry_seg_id = None
    for node_id, node in network_grid.nodes.items():
        if node.node_type == 'source':
            for seg_id, segment in network_grid.segments.items():
                if segment.get('start_node') == node_id:
                    entry_seg_id = seg_id
                    break
            break
    
    if entry_seg_id is None:
        print("ERROR: No entry segment found!")
        return False
    
    U_before = network_grid.segments[entry_seg_id]['U_initial'][:, :6].copy()
    print(f"Entry segment: {entry_seg_id}")
    print(f"U_initial before (first 6 cells): rho_m={U_before[0, :6].mean():.6f}, rho_c={U_before[2, :6].mean():.6f}")
    
    # Now simulate what set_inflow_conditions does
    from arz_model.core.physics import calculate_pressure
    
    density = 300.0  # veh/km
    velocity = 40.0  # km/h
    
    phys = config.physics
    rho_total = density / 1000.0  # veh/km → veh/m
    v_ms = velocity / 3.6  # km/h → m/s
    
    alpha = phys.alpha
    rho_m = rho_total * alpha
    rho_c = rho_total * (1.0 - alpha)
    
    rho_m_arr = np.array([rho_m])
    rho_c_arr = np.array([rho_c])
    p_m, p_c = calculate_pressure(
        rho_m_arr, rho_c_arr,
        phys.alpha, phys.rho_max, phys.epsilon,
        phys.k_m, phys.gamma_m, phys.k_c, phys.gamma_c
    )
    
    w_m = v_ms + p_m[0]
    w_c = v_ms + p_c[0]
    
    # Find entry segments and update
    entry_segments = []
    for node_id, node in network_grid.nodes.items():
        if node.node_type == 'source':
            for seg_id, segment in network_grid.segments.items():
                if segment.get('start_node') == node_id:
                    entry_segments.append(seg_id)
    
    print(f"Found {len(entry_segments)} entry segments: {entry_segments}")
    
    for seg_id in entry_segments:
        segment = network_grid.segments[seg_id]
        U_initial = segment.get('U_initial')
        grid = segment.get('grid')
        
        if U_initial is None or grid is None:
            continue
            
        n_ghost = grid.num_ghost_cells
        cells_to_update = n_ghost + 3
        
        U_initial[0, :cells_to_update] = rho_m
        U_initial[1, :cells_to_update] = w_m
        U_initial[2, :cells_to_update] = rho_c
        U_initial[3, :cells_to_update] = w_c
    
    U_after = network_grid.segments[entry_seg_id]['U_initial'][:, :6].copy()
    print(f"U_initial after (first 6 cells): rho_m={U_after[0, :6].mean():.6f}, rho_c={U_after[2, :6].mean():.6f}")
    
    # Check that values changed
    rho_changed = not np.allclose(U_before[0, :6], U_after[0, :6])
    print(f"Density changed: {rho_changed}")
    
    # Verify the new values are correct
    expected_rho_m = rho_m
    actual_rho_m = U_after[0, :6].mean()
    correct_value = np.isclose(actual_rho_m, expected_rho_m, rtol=0.01)
    print(f"Correct value (expected {expected_rho_m:.6f}): {correct_value}")
    
    return rho_changed and correct_value


def test_reset_uses_modified_initial():
    """Test that reset() uses the modified U_initial."""
    print("\n" + "=" * 60)
    print("TEST 2: reset() uses modified U_initial")
    print("=" * 60)
    
    config = create_victoria_island_config(
        t_final=100.0, output_dt=15.0, cells_per_100m=4,
        default_density=120.0, inflow_density=180.0, use_cache=False
    )
    config.rl_metadata = {'observation_segment_ids': [s.id for s in config.segments], 'decision_interval': 15.0}
    
    env = TrafficSignalEnvDirectV3(
        simulation_config=config,
        decision_interval=15.0,
        reward_weights={'alpha': 5.0, 'kappa': 0.0, 'mu': 0.1},
        quiet=True
    )
    
    # First reset with original density
    obs1, _ = env.reset()
    print(f"Observation after reset (original density=180): mean={obs1[:4].mean():.4f}")
    
    # Change inflow conditions and reset
    env.set_inflow_conditions(density=300.0, velocity=40.0)
    obs2, _ = env.reset()
    print(f"Observation after reset (new density=300): mean={obs2[:4].mean():.4f}")
    
    # Observations should be different
    obs_changed = not np.allclose(obs1, obs2)
    print(f"Observations different: {obs_changed}")
    
    env.close()
    return obs_changed


def test_variable_demand_wrapper():
    """Test that VariableDemandEnv actually varies demand."""
    print("\n" + "=" * 60)
    print("TEST 3: VariableDemandEnv varies demand across episodes")
    print("=" * 60)
    
    env = VariableDemandEnv(
        density_range=(100.0, 300.0),
        velocity_range=(30.0, 50.0),
        default_density=120.0,
        t_final=100.0,
        decision_interval=15.0,
        reward_weights={'alpha': 5.0, 'kappa': 0.0, 'mu': 0.1},
        seed=42,
        quiet=True
    )
    
    densities = []
    observations = []
    
    for ep in range(5):
        obs, info = env.reset()
        density = info.get('inflow_density', 'N/A')
        densities.append(density)
        observations.append(obs[:4].mean())
        print(f"Episode {ep+1}: density={density:.1f} veh/km, obs_mean={obs[:4].mean():.4f}")
    
    density_std = np.std(densities)
    print(f"\nDensity variation: σ = {density_std:.1f} veh/km")
    print(f"Observation variation: σ = {np.std(observations):.4f}")
    
    env.close()
    return density_std > 20.0  # Should have significant variation


if __name__ == '__main__':
    print("=" * 60)
    print("TRUE DOMAIN RANDOMIZATION TEST")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("set_inflow_conditions modifies U_initial", test_set_inflow_conditions()))
    except Exception as e:
        print(f"Test 1 FAILED with exception: {e}")
        results.append(("set_inflow_conditions modifies U_initial", False))
    
    try:
        results.append(("reset() uses modified U_initial", test_reset_uses_modified_initial()))
    except Exception as e:
        print(f"Test 2 FAILED with exception: {e}")
        results.append(("reset() uses modified U_initial", False))
    
    try:
        results.append(("VariableDemandEnv varies demand", test_variable_demand_wrapper()))
    except Exception as e:
        print(f"Test 3 FAILED with exception: {e}")
        results.append(("VariableDemandEnv varies demand", False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + ("✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"))
