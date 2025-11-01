"""
Quick Junction Flux Blocking Verification Tests

Fast tests (< 2 minutes total) to verify Bug #8 fix works correctly:
1. Junction blocking is active in production code
2. RED signal increases density, GREEN allows flow
3. No numerical instabilities

These are SMOKE TESTS - comprehensive validation is in test_junction_phase2_simple.py
"""

import sys
import os
import pytest
import numpy as np

# Add Code_RL/src to path for env imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Code_RL', 'src')))


def test_junction_blocking_active():
    """
    CRITICAL: Verify junction blocking is active in production configuration.
    
    This is the SMOKE TEST that verifies Bug #8 is fixed.
    Should complete in ~30 seconds.
    """
    from arz_model.config.builders import RLNetworkConfigBuilder
    from env.traffic_signal_env_direct import TrafficSignalEnvDirect
    
    print("\n" + "="*70)
    print("SMOKE TEST: Junction Blocking Active")
    print("="*70)
    
    config = RLNetworkConfigBuilder.simple_corridor(segments=2)
    env = TrafficSignalEnvDirect(simulation_config=config, quiet=True)
    obs, _ = env.reset()
    
    initial_rho = obs[0]
    
    # Run 5 steps with RED signal (action=0) - SHORT test
    for _ in range(5):
        obs, _, _, _, _ = env.step(action=0)
    
    final_rho = obs[0]
    density_increase = final_rho - initial_rho
    
    print(f"Initial ρ: {initial_rho:.4f}")
    print(f"Final ρ:   {final_rho:.4f}")
    print(f"Increase:  {density_increase:.4f}")
    
    # Must show density increase (congestion forming)
    assert density_increase > 0.005, \
        f"Junction blocking should cause congestion. Increase: {density_increase:.4f}"
    
    print("✅ Junction blocking is ACTIVE!")


def test_red_vs_green_behavior():
    """
    Verify RED accumulates density more than GREEN.
    Should complete in ~60 seconds.
    """
    from arz_model.config.builders import RLNetworkConfigBuilder
    from env.traffic_signal_env_direct import TrafficSignalEnvDirect
    
    print("\n" + "="*70)
    print("SMOKE TEST: RED vs GREEN Behavior")
    print("="*70)
    
    # RED scenario
    config_red = RLNetworkConfigBuilder.simple_corridor(segments=2)
    env_red = TrafficSignalEnvDirect(simulation_config=config_red, quiet=True)
    obs_red, _ = env_red.reset()
    initial_rho_red = obs_red[0]
    
    for _ in range(5):
        obs_red, _, _, _, _ = env_red.step(action=0)  # RED
    
    final_rho_red = obs_red[0]
    red_increase = final_rho_red - initial_rho_red
    
    # GREEN scenario
    config_green = RLNetworkConfigBuilder.simple_corridor(segments=2)
    env_green = TrafficSignalEnvDirect(simulation_config=config_green, quiet=True)
    obs_green, _ = env_green.reset()
    initial_rho_green = obs_green[0]
    
    for _ in range(5):
        obs_green, _, _, _, _ = env_green.step(action=1)  # GREEN
    
    final_rho_green = obs_green[0]
    green_increase = final_rho_green - initial_rho_green
    
    print(f"RED increase:   {red_increase:+.4f}")
    print(f"GREEN increase: {green_increase:+.4f}")
    
    # RED must accumulate more than GREEN
    assert red_increase > green_increase, \
        f"RED should accumulate more. RED: {red_increase:.4f}, GREEN: {green_increase:.4f}"
    
    print("✅ RED blocks, GREEN flows!")


def test_no_numerical_errors():
    """
    Verify no NaN/Inf errors during simulation.
    Should complete in ~30 seconds.
    """
    from arz_model.config.builders import RLNetworkConfigBuilder
    from env.traffic_signal_env_direct import TrafficSignalEnvDirect
    
    print("\n" + "="*70)
    print("SMOKE TEST: Numerical Stability")
    print("="*70)
    
    config = RLNetworkConfigBuilder.simple_corridor(segments=2)
    env = TrafficSignalEnvDirect(simulation_config=config, quiet=True)
    obs, _ = env.reset()
    
    # Run 10 steps with RED signal
    for step in range(10):
        obs, _, _, _, _ = env.step(action=0)
        
        # Check for numerical issues
        assert np.all(np.isfinite(obs)), \
            f"NaN or Inf detected at step {step}"
        assert np.all(obs >= 0), \
            f"Negative values detected at step {step}"
    
    print(f"✅ 10 steps completed without numerical errors!")
    print(f"   Final observation: ρ={obs[0]:.4f}, v={obs[1]:.4f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
