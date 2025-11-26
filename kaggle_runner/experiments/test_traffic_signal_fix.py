"""
Test Traffic Signal Fix: Validate that RL actions affect traffic flow

This script tests the critical fix for the traffic signal flux blocking bug:
- Task 5.1: Unit tests for GPUMemoryPool.update_light_factors()
- Task 5.2: Integration test - verify rewards change with phases
- Task 5.3: Performance benchmark - step latency < 1ms

The bug: light_factor = 1.0 was hardcoded in time_integration.py,
preventing RL actions from affecting traffic flow.

Usage:
    # Via Kaggle executor (recommended - requires GPU)
    python kaggle_runner/executor.py --target kaggle_runner/experiments/test_traffic_signal_fix.py --timeout 600
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from numba import cuda

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 80)
print("TEST: TRAFFIC SIGNAL FIX VALIDATION")
print("=" * 80)
print(f"Python: {sys.version}")
print(f"CUDA available: {cuda.is_available()}")
if cuda.is_available():
    print(f"GPU: {cuda.get_current_device().name}")
print("=" * 80)

# Results storage
results = {
    'task_5_1_unit_tests': {},
    'task_5_2_integration_test': {},
    'task_5_3_performance_benchmark': {},
    'overall_success': False
}

output_dir = Path("results/traffic_signal_fix_test")
output_dir.mkdir(parents=True, exist_ok=True)


def test_5_1_unit_tests():
    """
    Task 5.1: Unit tests for update_light_factors()
    
    Tests:
    - Light factors initialization to 1.0
    - Single segment update
    - Multiple segment updates
    - Unknown segment handling
    - Empty update handling
    - Toggle between GREEN and RED
    """
    print("\n" + "=" * 60)
    print("TASK 5.1: UNIT TESTS FOR update_light_factors()")
    print("=" * 60)
    
    from arz_model.numerics.gpu.memory_pool import GPUMemoryPool
    
    # Test configuration - matches the signature of GPUMemoryPool.__init__
    config = {
        'segment_ids': ['seg1', 'seg2', 'seg3'],
        'N_per_segment': {'seg1': 50, 'seg2': 100, 'seg3': 75},
        'ghost_cells': 3  # WENO5 needs 3 ghost cells
    }
    
    test_results = {}
    
    # Test 1: Initialization
    print("\n[Test 1] Light factors initialization...")
    try:
        pool = GPUMemoryPool(**config)
        assert pool.d_light_factors is not None, "d_light_factors not allocated"
        assert pool.d_light_factors.shape == (3,), f"Wrong shape: {pool.d_light_factors.shape}"
        
        light_factors = pool.d_light_factors.copy_to_host()
        assert np.allclose(light_factors, 1.0), f"Not initialized to 1.0: {light_factors}"
        
        test_results['initialization'] = 'PASSED'
        print("  ✓ PASSED: Light factors initialized to 1.0 (GREEN)")
        pool.clear()
    except Exception as e:
        test_results['initialization'] = f'FAILED: {e}'
        print(f"  ✗ FAILED: {e}")
    
    # Test 2: Single update
    print("\n[Test 2] Single segment update...")
    try:
        pool = GPUMemoryPool(**config)
        pool.update_light_factors({'seg1': 0.01})
        
        light_factors = pool.d_light_factors.copy_to_host()
        idx_seg1 = pool.segment_id_to_index['seg1']
        idx_seg2 = pool.segment_id_to_index['seg2']
        
        assert np.isclose(light_factors[idx_seg1], 0.01), f"seg1 not 0.01: {light_factors[idx_seg1]}"
        assert np.isclose(light_factors[idx_seg2], 1.0), f"seg2 changed: {light_factors[idx_seg2]}"
        
        test_results['single_update'] = 'PASSED'
        print("  ✓ PASSED: Single segment updated correctly")
        pool.clear()
    except Exception as e:
        test_results['single_update'] = f'FAILED: {e}'
        print(f"  ✗ FAILED: {e}")
    
    # Test 3: Multiple updates
    print("\n[Test 3] Multiple segment updates...")
    try:
        pool = GPUMemoryPool(**config)
        pool.update_light_factors({'seg1': 0.01, 'seg2': 0.5, 'seg3': 0.25})
        
        light_factors = pool.d_light_factors.copy_to_host()
        
        assert np.isclose(light_factors[pool.segment_id_to_index['seg1']], 0.01)
        assert np.isclose(light_factors[pool.segment_id_to_index['seg2']], 0.5)
        assert np.isclose(light_factors[pool.segment_id_to_index['seg3']], 0.25)
        
        test_results['multiple_updates'] = 'PASSED'
        print("  ✓ PASSED: Multiple segments updated correctly")
        pool.clear()
    except Exception as e:
        test_results['multiple_updates'] = f'FAILED: {e}'
        print(f"  ✗ FAILED: {e}")
    
    # Test 4: Unknown segment (should be skipped)
    print("\n[Test 4] Unknown segment handling...")
    try:
        pool = GPUMemoryPool(**config)
        pool.update_light_factors({'unknown_seg': 0.01, 'seg1': 0.5})  # Should not raise
        
        light_factors = pool.d_light_factors.copy_to_host()
        assert np.isclose(light_factors[pool.segment_id_to_index['seg1']], 0.5)
        
        test_results['unknown_segment'] = 'PASSED'
        print("  ✓ PASSED: Unknown segment silently skipped")
        pool.clear()
    except Exception as e:
        test_results['unknown_segment'] = f'FAILED: {e}'
        print(f"  ✗ FAILED: {e}")
    
    # Test 5: Empty update
    print("\n[Test 5] Empty update handling...")
    try:
        pool = GPUMemoryPool(**config)
        pool.update_light_factors({})  # Should not raise
        
        light_factors = pool.d_light_factors.copy_to_host()
        assert np.allclose(light_factors, 1.0), "Empty update changed values"
        
        test_results['empty_update'] = 'PASSED'
        print("  ✓ PASSED: Empty update does nothing")
        pool.clear()
    except Exception as e:
        test_results['empty_update'] = f'FAILED: {e}'
        print(f"  ✗ FAILED: {e}")
    
    # Test 6: Toggle GREEN/RED
    print("\n[Test 6] Toggle between GREEN and RED...")
    try:
        pool = GPUMemoryPool(**config)
        
        # Start GREEN (1.0)
        light_factors = pool.d_light_factors.copy_to_host()
        assert np.allclose(light_factors, 1.0), "Initial state not GREEN"
        
        # Switch to RED (0.01)
        pool.update_light_factors({'seg1': 0.01, 'seg2': 0.01, 'seg3': 0.01})
        light_factors = pool.d_light_factors.copy_to_host()
        assert np.allclose(light_factors, 0.01), "Not switched to RED"
        
        # Switch back to GREEN
        pool.update_light_factors({'seg1': 1.0, 'seg2': 1.0, 'seg3': 1.0})
        light_factors = pool.d_light_factors.copy_to_host()
        assert np.allclose(light_factors, 1.0), "Not switched back to GREEN"
        
        test_results['toggle_green_red'] = 'PASSED'
        print("  ✓ PASSED: Toggle GREEN ↔ RED works correctly")
        pool.clear()
    except Exception as e:
        test_results['toggle_green_red'] = f'FAILED: {e}'
        print(f"  ✗ FAILED: {e}")
    
    # Test 7: get_batched_arrays returns light factors
    print("\n[Test 7] get_batched_arrays returns light factors...")
    try:
        pool = GPUMemoryPool(**config)
        
        # Initialize segments first
        for seg_id in config['segment_ids']:
            N_phys = config['N_per_segment'][seg_id]
            U_init = np.random.rand(4, N_phys).astype(np.float64)
            pool.initialize_segment_state(seg_id, U_init)
        
        result = pool.get_batched_arrays()
        assert len(result) == 5, f"Expected 5 elements, got {len(result)}"
        
        d_U, d_R, d_offsets, d_lengths, d_light_factors = result
        assert d_light_factors is not None, "d_light_factors is None"
        assert d_light_factors.shape == (3,), f"Wrong shape: {d_light_factors.shape}"
        
        test_results['get_batched_arrays'] = 'PASSED'
        print("  ✓ PASSED: get_batched_arrays returns 5 elements including light_factors")
        pool.clear()
    except Exception as e:
        test_results['get_batched_arrays'] = f'FAILED: {e}'
        print(f"  ✗ FAILED: {e}")
    
    # Summary
    passed = sum(1 for v in test_results.values() if v == 'PASSED')
    total = len(test_results)
    
    print("\n" + "-" * 40)
    print(f"TASK 5.1 SUMMARY: {passed}/{total} tests passed")
    print("-" * 40)
    
    results['task_5_1_unit_tests'] = {
        'tests': test_results,
        'passed': passed,
        'total': total,
        'success': passed == total
    }
    
    return passed == total


def test_5_2_integration():
    """
    Task 5.2: Integration test - verify rewards change with phases
    
    This is the CRITICAL test that verifies the bug is fixed:
    - Run simulation with all GREEN phases
    - Run simulation with all RED phases
    - Compare rewards - they MUST be different!
    
    Before the fix: rewards were constant ~286-288 regardless of phase
    After the fix: RED phases should cause congestion → lower rewards
    """
    print("\n" + "=" * 60)
    print("TASK 5.2: INTEGRATION TEST - REWARDS MUST CHANGE WITH PHASES")
    print("=" * 60)
    
    from Code_RL.src.env.traffic_signal_env_direct_v3 import TrafficSignalEnvDirectV3
    from arz_model.config import create_victoria_island_config
    
    # Create congested scenario to make signal effects visible
    print("\n[Setup] Creating congested traffic scenario...")
    config = create_victoria_island_config(
        t_final=120.0,  # Short episode for testing
        output_dt=15.0,
        cells_per_100m=4,
        default_density=80.0,   # HIGH congestion
        inflow_density=100.0,
        use_cache=False
    )
    # Add RL metadata directly to the config
    config.rl_metadata = {
        'observation_segment_ids': [s.id for s in config.segments],
        'decision_interval': 15.0,
    }
    
    # Test 1: Run episode with ALL GREEN (action=0 = keep phase, initial phase=0=GREEN)
    print("\n[Test 1] Running episode with ALL GREEN phases...")
    try:
        env_green = TrafficSignalEnvDirectV3(config, quiet=True)
        obs, info = env_green.reset()
        
        rewards_green = []
        done = False
        step = 0
        while not done:
            action = 0  # Keep current phase (GREEN)
            obs, reward, terminated, truncated, info = env_green.step(action)
            rewards_green.append(reward)
            done = terminated or truncated
            step += 1
        
        total_reward_green = sum(rewards_green)
        avg_reward_green = np.mean(rewards_green)
        print(f"  GREEN: {len(rewards_green)} steps, total={total_reward_green:.2f}, avg={avg_reward_green:.2f}")
        env_green.close()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['task_5_2_integration_test'] = {'success': False, 'error': str(e)}
        return False
    
    # Test 2: Run episode with ALL RED (action=1 = switch to phase 1 = RED, keep switching)
    print("\n[Test 2] Running episode with FREQUENT RED phases (switching every step)...")
    try:
        env_red = TrafficSignalEnvDirectV3(config, quiet=True)
        obs, info = env_red.reset()
        
        rewards_red = []
        done = False
        step = 0
        while not done:
            action = 1  # Switch phase every step → more RED time
            obs, reward, terminated, truncated, info = env_red.step(action)
            rewards_red.append(reward)
            done = terminated or truncated
            step += 1
        
        total_reward_red = sum(rewards_red)
        avg_reward_red = np.mean(rewards_red)
        print(f"  RED/SWITCH: {len(rewards_red)} steps, total={total_reward_red:.2f}, avg={avg_reward_red:.2f}")
        env_red.close()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['task_5_2_integration_test'] = {'success': False, 'error': str(e)}
        return False
    
    # CRITICAL COMPARISON
    print("\n" + "-" * 40)
    print("CRITICAL COMPARISON (Bug Fix Verification)")
    print("-" * 40)
    
    reward_diff = abs(total_reward_green - total_reward_red)
    reward_pct_diff = (reward_diff / max(abs(total_reward_green), 1.0)) * 100
    
    print(f"  Total Reward (GREEN): {total_reward_green:.2f}")
    print(f"  Total Reward (RED/SWITCH): {total_reward_red:.2f}")
    print(f"  Difference: {reward_diff:.2f} ({reward_pct_diff:.1f}%)")
    
    # Success criteria: rewards MUST differ by at least 5%
    # Before fix: rewards were constant ~286-288 (< 1% variation)
    success = reward_pct_diff >= 5.0
    
    if success:
        print(f"\n  ✓ SUCCESS: Rewards differ by {reward_pct_diff:.1f}% (threshold: 5%)")
        print("  → RL actions now affect traffic flow!")
        print("  → Agent CAN learn from the environment!")
    else:
        print(f"\n  ✗ FAILURE: Rewards differ by only {reward_pct_diff:.1f}% (threshold: 5%)")
        print("  → Bug may still be present - signals not affecting flow")
        print("  → Check that light_factor is being applied in kernel")
    
    results['task_5_2_integration_test'] = {
        'rewards_green': {
            'steps': len(rewards_green),
            'total': total_reward_green,
            'avg': avg_reward_green,
            'values': rewards_green
        },
        'rewards_red': {
            'steps': len(rewards_red),
            'total': total_reward_red,
            'avg': avg_reward_red,
            'values': rewards_red
        },
        'difference': reward_diff,
        'difference_pct': reward_pct_diff,
        'success': success
    }
    
    return success


def test_5_3_performance():
    """
    Task 5.3: Performance benchmark - step latency < 1ms
    
    Verify that the fix doesn't regress performance.
    Target: average step latency < 1ms for GPU-accelerated simulation.
    """
    print("\n" + "=" * 60)
    print("TASK 5.3: PERFORMANCE BENCHMARK - STEP LATENCY < 1ms")
    print("=" * 60)
    
    from Code_RL.src.env.traffic_signal_env_direct_v3 import TrafficSignalEnvDirectV3
    from arz_model.config import create_victoria_island_config
    
    # Standard config for performance testing
    print("\n[Setup] Creating standard Victoria Island config...")
    config = create_victoria_island_config(
        t_final=300.0,  # 5 minutes
        output_dt=15.0,
        cells_per_100m=4,
        default_density=50.0,
        inflow_density=70.0,
        use_cache=False
    )
    # Add RL metadata directly to the config
    config.rl_metadata = {
        'observation_segment_ids': [s.id for s in config.segments],
        'decision_interval': 15.0,
    }
    
    print("\n[Benchmark] Running 50 steps with timing...")
    try:
        env = TrafficSignalEnvDirectV3(config, quiet=True)
        obs, info = env.reset()
        
        # Warm-up (JIT compilation)
        for _ in range(5):
            obs, _, done, _, _ = env.step(0)
            if done:
                obs, _ = env.reset()
        
        # Timed steps
        step_times = []
        for i in range(50):
            start = time.perf_counter()
            action = np.random.randint(0, 2)  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            step_times.append(elapsed)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        env.close()
        
        avg_latency = np.mean(step_times)
        std_latency = np.std(step_times)
        max_latency = np.max(step_times)
        min_latency = np.min(step_times)
        p95_latency = np.percentile(step_times, 95)
        
        print(f"\n  Step Latency Statistics (50 steps):")
        print(f"    Average: {avg_latency:.3f} ms")
        print(f"    Std Dev: {std_latency:.3f} ms")
        print(f"    Min: {min_latency:.3f} ms")
        print(f"    Max: {max_latency:.3f} ms")
        print(f"    P95: {p95_latency:.3f} ms")
        
        # Success criteria: average < 1ms, P95 < 2ms
        success = avg_latency < 1.0 and p95_latency < 2.0
        
        if success:
            print(f"\n  ✓ SUCCESS: avg={avg_latency:.3f}ms < 1ms, p95={p95_latency:.3f}ms < 2ms")
        else:
            print(f"\n  ✗ FAILURE: Performance regression detected")
            print(f"    Target: avg < 1ms, p95 < 2ms")
        
        results['task_5_3_performance_benchmark'] = {
            'avg_latency_ms': avg_latency,
            'std_latency_ms': std_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'p95_latency_ms': p95_latency,
            'step_times_ms': step_times,
            'success': success
        }
        
        return success
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['task_5_3_performance_benchmark'] = {'success': False, 'error': str(e)}
        return False


def main():
    """Run all tests for Phase 5 validation."""
    print("\n" + "=" * 80)
    print("STARTING TRAFFIC SIGNAL FIX VALIDATION")
    print("=" * 80)
    
    start_time = time.time()
    
    # Check CUDA availability
    if not cuda.is_available():
        print("\n⚠ ERROR: CUDA not available! These tests require GPU.")
        print("Run on Kaggle with GPU accelerator enabled.")
        results['overall_success'] = False
        results['error'] = 'CUDA not available'
        
        # Save results
        with open(output_dir / "test_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return 1
    
    # Run tests
    success_5_1 = test_5_1_unit_tests()
    success_5_2 = test_5_2_integration()
    success_5_3 = test_5_3_performance()
    
    # Overall summary
    elapsed = time.time() - start_time
    overall_success = success_5_1 and success_5_2 and success_5_3
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"  Task 5.1 (Unit Tests):      {'✓ PASSED' if success_5_1 else '✗ FAILED'}")
    print(f"  Task 5.2 (Integration):     {'✓ PASSED' if success_5_2 else '✗ FAILED'}")
    print(f"  Task 5.3 (Performance):     {'✓ PASSED' if success_5_3 else '✗ FAILED'}")
    print("-" * 40)
    print(f"  OVERALL:                    {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 80)
    
    if overall_success:
        print("\n🎉 TRAFFIC SIGNAL FIX VALIDATED!")
        print("   RL agent actions NOW affect traffic flow.")
        print("   The agent CAN learn to optimize signal timing.")
    else:
        print("\n⚠ FIX VALIDATION INCOMPLETE")
        print("   Review failed tests and check implementation.")
    
    results['overall_success'] = overall_success
    results['elapsed_seconds'] = elapsed
    
    # Save results
    with open(output_dir / "test_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir / 'test_results.json'}")
    
    return 0 if overall_success else 1


if __name__ == '__main__':
    sys.exit(main())
