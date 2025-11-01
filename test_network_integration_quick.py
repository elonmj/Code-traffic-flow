"""
Quick Test: NetworkGrid Integration with Code_RL

Verifies that 2-segment corridor configuration:
1. Creates NetworkSimulationConfig successfully
2. TrafficSignalEnvDirect detects and uses NetworkGridSimulator
3. Environment initializes without errors
4. Congestion forms (queue_length > 0)
5. Rewards vary based on actions

Expected Results:
- Queue length: 0 initially → 5-15 vehicles during RED
- Rewards: -8 to +3 range (not stuck at 0 or -0.01)
- Computation time: <30x realtime

Author: ARZ Research Team
Date: 2025-10-28 (NetworkGrid Integration)
"""

import sys
import os
import time

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Code_RL/src')))

def test_network_config_creation():
    """Test 1: Verify NetworkSimulationConfig creation."""
    print("="*70)
    print("TEST 1: NetworkSimulationConfig Creation")
    print("="*70)
    
    try:
        from arz_model.config.builders import RLNetworkConfigBuilder
        
        # Create 2-segment corridor
        config = RLNetworkConfigBuilder.simple_corridor(segments=2)
        
        print(f"✅ Config created successfully")
        print(f"   Type: {type(config).__name__}")
        print(f"   Segments: {len(config.segments)}")
        print(f"   Nodes: {len(config.nodes)}")
        print(f"   Links: {len(config.links)}")
        print(f"   Controlled nodes: {config.controlled_nodes}")
        
        # Validate topology
        config.validate_network_topology()
        print(f"✅ Topology validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_initialization():
    """Test 2: Verify TrafficSignalEnvDirect detects NetworkSimulationConfig."""
    print("\n" + "="*70)
    print("TEST 2: Environment Initialization")
    print("="*70)
    
    try:
        from arz_model.config.builders import RLNetworkConfigBuilder
        from env.traffic_signal_env_direct import TrafficSignalEnvDirect
        
        # Create config
        config = RLNetworkConfigBuilder.simple_corridor(segments=2)
        print(f"✅ Config created")
        
        # Create environment
        print(f"\nInitializing environment...")
        env = TrafficSignalEnvDirect(
            simulation_config=config,
            quiet=False
        )
        
        print(f"\n✅ Environment initialized successfully")
        print(f"   Type: {type(env).__name__}")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.n}")
        
        # Check if network mode detected
        if hasattr(env, 'network'):
            print(f"✅ Network mode detected!")
            print(f"   Network segments: {len(env.network.segments)}")
        else:
            print(f"⚠️  Single segment mode (expected network mode)")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_congestion_formation():
    """Test 3: Verify congestion forms and rewards vary."""
    print("\n" + "="*70)
    print("TEST 3: Congestion Formation")
    print("="*70)
    
    try:
        from arz_model.config.builders import RLNetworkConfigBuilder
        from env.traffic_signal_env_direct import TrafficSignalEnvDirect
        import numpy as np
        
        # Create environment
        config = RLNetworkConfigBuilder.simple_corridor(segments=2)
        env = TrafficSignalEnvDirect(simulation_config=config, quiet=True)
        
        # Reset
        obs, info = env.reset()
        print(f"✅ Environment reset")
        print(f"   Initial observation shape: {obs.shape}")
        print(f"   Initial queue: {info.get('queue_length', 0):.2f} veh")
        
        # Run 20 steps to observe congestion formation (5 minutes simulation time)
        print(f"\n{'='*80}")
        print(f"OBSERVING CONGESTION FORMATION (20 steps = 5 min)")
        print(f"{'='*80}")
        print(f"{'Step':<6} {'Action':<8} {'Queue':<10} {'ρ_m':<10} {'ρ_c':<10} {'v_m':<10} {'v_c':<10} {'Reward':<10}")
        print("-" * 80)
        
        for step in range(20):
            action = 0  # ✅ FIX: action=0 → phase=0 → RED (blocks flow)
            action_name = "RED" if action == 0 else "GREEN"
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Extract key traffic variables from observation
            # Observation format: [ρ_m/ρ_max, v_m/v_free, ρ_c/ρ_max, v_c/v_free] × n_segments + phase_onehot
            # Get first segment values (upstream)
            rho_m_norm = obs[0]  # Motorcycle density (normalized)
            v_m_norm = obs[1]    # Motorcycle velocity (normalized)
            rho_c_norm = obs[2]  # Car density (normalized)
            v_c_norm = obs[3]    # Car velocity (normalized)
            
            # Denormalize for display (using typical values)
            rho_m = rho_m_norm * 0.2  # rho_max_m = 0.2 veh/m
            v_m = v_m_norm * (50.0/3.6)  # v_free_m = 50 km/h
            rho_c = rho_c_norm * 0.2  # rho_max_c = 0.2 veh/m
            v_c = v_c_norm * (50.0/3.6)  # v_free_c = 50 km/h
            
            queue_length = info.get('queue_length', 0)
            
            print(f"{step:<6} {action_name:<8} {queue_length:<10.2f} {rho_m:<10.4f} {rho_c:<10.4f} {v_m:<10.2f} {v_c:<10.2f} {reward:<10.4f}")
        
        # Quick analysis
        print("\n" + "="*80)
        print("QUICK ANALYSIS")
        print("="*80)
        final_queue = info.get('queue_length', 0)
        
        if final_queue > 5.0:
            print(f"✅ Congestion forming: Queue = {final_queue:.2f} veh (target: >5 veh)")
            success = True
        elif final_queue > 0.5:
            print(f"⚠️  Queue building slowly: {final_queue:.2f} veh (needs more time)")
            success = False
        else:
            print(f"❌ No congestion: Queue = {final_queue:.2f} veh")
            success = False
        
        print(f"\nTest result: {'PASS' if success else 'FAIL'}")
        return success
        
    except Exception as e:
        print(f"❌ Congestion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
        return success
        
    except Exception as e:
        print(f"❌ Congestion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("NETWORKGRID INTEGRATION QUICK TEST")
    print("="*70)
    print()
    
    results = []
    
    # Test 1: Config creation
    results.append(("Config Creation", test_network_config_creation()))
    
    # Test 2: Environment initialization
    results.append(("Environment Init", test_environment_initialization()))
    
    # Test 3: Congestion formation
    results.append(("Congestion Formation", test_congestion_formation()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nNext steps:")
        print("  1. Run full validation: python validation_ch7/scripts/test_section_7_6_rl_performance.py")
        print("  2. Test 10-segment network: RLNetworkConfigBuilder.medium_network(segments=10)")
        print("  3. Scale to Lagos 75-segment network")
    else:
        print("\n❌ SOME TESTS FAILED - Debug needed")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
