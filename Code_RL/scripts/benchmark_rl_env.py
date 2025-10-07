"""
Performance Benchmark for TrafficSignalEnvDirect

Measures step time to verify <1ms target for direct coupling.
Compares against theoretical server-based overhead.
"""

import time
import numpy as np
import os
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect


def benchmark_environment(n_steps: int = 1000, warmup_steps: int = 10):
    """
    Benchmark the direct environment performance.
    
    Args:
        n_steps: Number of steps to measure (default: 1000)
        warmup_steps: Number of warmup steps before measurement
        
    Returns:
        dict: Performance statistics
    """
    print("=" * 70)
    print("TrafficSignalEnvDirect Performance Benchmark")
    print("=" * 70)
    
    # Initialize environment
    scenario_path = project_root / 'scenarios' / 'scenario_calibration_victoria_island.yml'
    
    print(f"\nInitializing environment...")
    print(f"  Scenario: {scenario_path.name}")
    print(f"  Decision interval: 10.0s")
    print(f"  Device: CPU")
    
    env = TrafficSignalEnvDirect(
        scenario_config_path=str(scenario_path),
        decision_interval=10.0,
        episode_max_time=3600.0,
        quiet=True,
        device='cpu'
    )
    
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")
    
    # Warmup
    print(f"\nWarming up ({warmup_steps} steps)...")
    env.reset(seed=42)
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        env.step(action)
    
    # Benchmark
    print(f"\nBenchmarking ({n_steps} steps)...")
    env.reset(seed=42)
    
    step_times = []
    
    for i in range(n_steps):
        action = env.action_space.sample()
        
        start = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        end = time.perf_counter()
        
        step_time_ms = (end - start) * 1000  # Convert to milliseconds
        step_times.append(step_time_ms)
        
        if terminated or truncated:
            env.reset(seed=42 + i)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{n_steps} steps")
    
    env.close()
    
    # Calculate statistics
    step_times = np.array(step_times)
    
    stats = {
        'mean_ms': np.mean(step_times),
        'median_ms': np.median(step_times),
        'std_ms': np.std(step_times),
        'min_ms': np.min(step_times),
        'max_ms': np.max(step_times),
        'p95_ms': np.percentile(step_times, 95),
        'p99_ms': np.percentile(step_times, 99),
        'n_steps': n_steps
    }
    
    return stats, step_times


def print_results(stats: dict, step_times: np.ndarray):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    print(f"\nStep Time Statistics (ms):")
    print(f"  Mean:      {stats['mean_ms']:8.3f} ms")
    print(f"  Median:    {stats['median_ms']:8.3f} ms")
    print(f"  Std Dev:   {stats['std_ms']:8.3f} ms")
    print(f"  Min:       {stats['min_ms']:8.3f} ms")
    print(f"  Max:       {stats['max_ms']:8.3f} ms")
    print(f"  95th %ile: {stats['p95_ms']:8.3f} ms")
    print(f"  99th %ile: {stats['p99_ms']:8.3f} ms")
    
    # Success criteria
    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA")
    print(f"{'='*70}")
    
    target_ms = 1.0
    success = stats['p95_ms'] < target_ms
    
    print(f"\nTarget: 95th percentile < {target_ms:.1f} ms")
    print(f"Actual: 95th percentile = {stats['p95_ms']:.3f} ms")
    print(f"Status: {'✓ PASS' if success else '✗ FAIL'}")
    
    # Performance multiplier vs server-based
    estimated_server_step_ms = 20.0  # Conservative estimate for HTTP roundtrip
    speedup = estimated_server_step_ms / stats['median_ms']
    
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    print(f"\nDirect coupling (this):  {stats['median_ms']:8.3f} ms/step")
    print(f"Server-based (estimate): {estimated_server_step_ms:8.1f} ms/step")
    print(f"Speedup:                 {speedup:8.1f}x faster")
    
    # Training time estimate
    million_steps = 1_000_000
    direct_time_s = (stats['median_ms'] / 1000) * million_steps
    server_time_s = (estimated_server_step_ms / 1000) * million_steps
    
    print(f"\n{'='*70}")
    print(f"TRAINING TIME ESTIMATE (1M steps)")
    print(f"{'='*70}")
    print(f"\nDirect coupling:  {direct_time_s/60:8.1f} minutes ({direct_time_s:,.0f}s)")
    print(f"Server-based:     {server_time_s/60:8.1f} minutes ({server_time_s:,.0f}s)")
    print(f"Time saved:       {(server_time_s - direct_time_s)/60:8.1f} minutes")
    
    # Histogram
    print(f"\n{'='*70}")
    print("DISTRIBUTION HISTOGRAM")
    print(f"{'='*70}\n")
    
    bins = [0, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf]
    bin_labels = ['<0.5ms', '0.5-1ms', '1-2ms', '2-5ms', '5-10ms', '>10ms']
    
    hist, _ = np.histogram(step_times, bins=bins)
    
    for label, count in zip(bin_labels, hist):
        pct = (count / len(step_times)) * 100
        bar = '█' * int(pct / 2)  # Scale for display
        print(f"{label:>8s}: {bar:50s} {count:5d} ({pct:5.1f}%)")
    
    print(f"\n{'='*70}\n")
    
    return success


def save_results(stats: dict, step_times: np.ndarray, output_file: str = None):
    """Save benchmark results to file."""
    if output_file is None:
        output_file = project_root / 'validation_output' / 'benchmark_rl_env_results.txt'
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("TrafficSignalEnvDirect Performance Benchmark Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Steps measured: {stats['n_steps']}\n\n")
        f.write("Step Time Statistics (ms):\n")
        f.write(f"  Mean:      {stats['mean_ms']:.3f}\n")
        f.write(f"  Median:    {stats['median_ms']:.3f}\n")
        f.write(f"  Std Dev:   {stats['std_ms']:.3f}\n")
        f.write(f"  Min:       {stats['min_ms']:.3f}\n")
        f.write(f"  Max:       {stats['max_ms']:.3f}\n")
        f.write(f"  95th:      {stats['p95_ms']:.3f}\n")
        f.write(f"  99th:      {stats['p99_ms']:.3f}\n\n")
        
        target_ms = 1.0
        success = stats['p95_ms'] < target_ms
        f.write(f"Target: <{target_ms} ms (95th percentile)\n")
        f.write(f"Result: {'PASS' if success else 'FAIL'}\n")
    
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark TrafficSignalEnvDirect performance')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps to benchmark')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup steps')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    
    args = parser.parse_args()
    
    # Run benchmark
    stats, step_times = benchmark_environment(n_steps=args.steps, warmup_steps=args.warmup)
    
    # Print results
    success = print_results(stats, step_times)
    
    # Save results
    save_results(stats, step_times, output_file=args.output)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
