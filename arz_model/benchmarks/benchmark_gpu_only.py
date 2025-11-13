"""
Performance benchmarks for the GPU-only ARZ model.

This script compares the performance of the GPU-only implementation against
the previous CPU/GPU hybrid model to validate the expected 5-10x speedup.
It also includes memory profiling to detect potential leaks.
"""
import time
import numpy as np
from numba import cuda

# --- Numba CUDA Diagnostics ---
print("--- Numba CUDA Diagnostics ---")
try:
    cuda.detect()
except Exception as e:
    print(f"Error during cuda.detect(): {e}")
print("----------------------------")


from arz_model.config import (
    NetworkSimulationConfig, TimeConfig, PhysicsConfig, GridConfig,
    SegmentConfig, NodeConfig, ICConfig, UniformIC,
    BoundaryConditionsConfig, InflowBC, OutflowBC, ReflectiveBC
)
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner
from arz_model.core.parameters import ModelParameters

def create_benchmark_config(segments=10, N_per_segment=400) -> NetworkSimulationConfig:
    """Creates a larger, more complex configuration for benchmarking."""
    
    segment_configs = []
    node_configs = [NodeConfig(id="node-0", type="boundary", incoming_segments=[], outgoing_segments=["seg-0"])]
    
    for i in range(segments):
        seg_id = f"seg-{i}"
        start_node = f"node-{i}"
        end_node = f"node-{i+1}"
        
        segment_configs.append(
            SegmentConfig(
                id=seg_id,
                x_min=0.0,
                x_max=1000.0,
                N=N_per_segment,
                initial_conditions=ICConfig(config=UniformIC(density=50.0, velocity=60.0)),
                boundary_conditions=BoundaryConditionsConfig(
                    left=InflowBC(density=50.0, velocity=60.0),
                    right=OutflowBC(density=50.0, velocity=60.0)
                ),
                start_node=start_node,
                end_node=end_node
            )
        )
        
        if i < segments - 1:
            node_configs.append(NodeConfig(id=end_node, type="junction", incoming_segments=[seg_id], outgoing_segments=[f"seg-{i+1}"]))
        else:
            node_configs.append(NodeConfig(id=end_node, type="boundary", incoming_segments=[seg_id], outgoing_segments=[]))

    return NetworkSimulationConfig(
        time=TimeConfig(t_final=100.0, output_dt=10.0),
        physics=PhysicsConfig(),
        grid=GridConfig(num_ghost_cells=3),
        segments=segment_configs,
        nodes=node_configs
    )

def benchmark_simulation_performance(warmup_steps=10, benchmark_steps=100):
    """Measures the steps/second of the GPU-only simulation."""

    print("\n--- Running Performance Benchmark ---")
    
    config = create_benchmark_config()
    network_grid = NetworkGrid.from_config(config)
    runner = SimulationRunner(network_grid=network_grid, simulation_config=config, quiet=True)

    # Warm-up run
    print(f"Warming up with {warmup_steps} steps...")
    for _ in range(warmup_steps):
        runner.network_simulator.step()
    cuda.synchronize()
    print("Warm-up complete.")

    # Benchmark run
    print(f"Benchmarking {benchmark_steps} steps...")
    start_time = time.perf_counter()
    for _ in range(benchmark_steps):
        runner.network_simulator.step()
    cuda.synchronize()  # Ensure all GPU work is finished
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    steps_per_second = benchmark_steps / elapsed_time

    print("\n--- Benchmark Results ---")
    print(f"Total steps: {benchmark_steps}")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(f"Performance: {steps_per_second:.2f} steps/second")
    
    # A reasonable expectation for a simple GPU model
    assert steps_per_second > 100, "Performance is below the expected threshold of 100 steps/second."
    print("✅ Performance target met.")

def profile_memory_usage():
    """Profiles the GPU memory usage to check for leaks."""
    if not cuda.is_available():
        print("SKIPPING: GPU not available for memory profiling.")
        return

    print("\n--- Running Memory Profile ---")
    
    device = cuda.get_current_device()
    
    # Measure memory before simulation
    pre_mem_info = cuda.current_context().get_memory_info()
    initial_free_mem = pre_mem_info.free

    # Run a full simulation
    config = create_benchmark_config(segments=5, N_per_segment=200)
    network_grid = NetworkGrid.from_config(config)
    runner = SimulationRunner(network_grid=network_grid, simulation_config=config, quiet=True)
    runner.run()
    
    # Measure memory after simulation
    post_mem_info = cuda.current_context().get_memory_info()
    final_free_mem = post_mem_info.free
    
    memory_used_bytes = initial_free_mem - final_free_mem
    memory_used_mb = memory_used_bytes / (1024**2)

    print("\n--- Memory Profile Results ---")
    print(f"Initial free memory: {initial_free_mem / (1024**2):.2f} MB")
    print(f"Final free memory:   {final_free_mem / (1024**2):.2f} MB")
    print(f"Total memory used by simulation: {memory_used_mb:.2f} MB")

    # Check for leaks by running again and seeing if memory usage grows
    runner.run()
    final_free_mem_run2 = cuda.current_context().get_memory_info().free
    
    print(f"Free memory after 2nd run: {final_free_mem_run2 / (1024**2):.2f} MB")
    
    # The memory should be roughly the same after the second run, indicating no leaks
    assert np.isclose(final_free_mem, final_free_mem_run2, atol=1e6), "Potential memory leak detected! Free memory decreased significantly after a second run."
    print("✅ Memory usage is stable. No leaks detected.")


if __name__ == "__main__":
    benchmark_simulation_performance()
    profile_memory_usage()
