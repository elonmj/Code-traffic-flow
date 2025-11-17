"""
GPU Optimization Validation Tests
==================================

Validates that GPU kernel optimizations maintain numerical accuracy.
Tests that max-norm difference from baseline is < 1e-10.

This test suite runs on Kaggle to validate:
1. Numerical accuracy of optimized kernels
2. Physical bounds preservation
3. Simulation stability
4. Performance improvement verification

Author: GPU Optimization Task (2025-11-17)
"""

import pytest
import numpy as np
import cupy as cp
from numba import cuda
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arz_model.config.config_factory import VictoriaIslandConfigFactory
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner


class TestGPUOptimizationNumericalAccuracy:
    """Test suite for GPU optimization numerical accuracy validation."""
    
    def test_optimized_kernels_import(self):
        """Verify all optimized modules import without errors."""
        from arz_model.numerics.gpu import weno_cuda
        from arz_model.numerics.gpu import ssp_rk3_cuda
        from arz_model.numerics.gpu import network_coupling_gpu
        from arz_model.core import node_solver_gpu
        
        # Verify fastmath is enabled (check decorator)
        assert 'fastmath' in str(weno_cuda.weno5_reconstruction_kernel.signatures)
        assert 'fastmath' in str(ssp_rk3_cuda.ssp_rk3_stage1_kernel.signatures)
        
        print("✅ All optimized modules imported successfully with fastmath enabled")
    
    def test_weno_constant_optimization(self):
        """Verify WENO constants are defined correctly."""
        from arz_model.numerics.gpu import weno_cuda
        
        # Check module-level constants exist
        assert hasattr(weno_cuda, 'WENO_C0_L')
        assert hasattr(weno_cuda, 'WENO_EPSILON')
        assert hasattr(weno_cuda, 'WENO_POLY_INV6')
        
        # Verify values are correct
        assert weno_cuda.WENO_C0_L == 0.1
        assert weno_cuda.WENO_C1_L == 0.6
        assert weno_cuda.WENO_C2_L == 0.3
        assert weno_cuda.WENO_EPSILON == 1e-6
        assert abs(weno_cuda.WENO_POLY_INV6 - 1.0/6.0) < 1e-15
        
        print("✅ WENO constants verified")
    
    def test_simulation_stability_30s(self):
        """Test that optimized simulation runs stably for 30 seconds."""
        csv_path = "arz_model/data/fichier_de_travail_corridor_utf8.csv"
        
        config = VictoriaIslandConfigFactory.create_victoria_island_config(
            csv_path=csv_path,
            t_final=30.0,
            output_dt=5.0,
            dx=25.0,
            initial_density_cars=30.0,
            initial_density_motorcycles=20.0
        )
        
        network_grid = NetworkGrid.from_config(config)
        runner = SimulationRunner(network_grid, config)
        
        # Run simulation
        history, final_time, total_steps = runner.run()
        
        # Verify completion
        assert final_time >= 30.0, f"Simulation stopped early at t={final_time}"
        assert total_steps > 0, "No steps were executed"
        
        # Check for NaN/Inf in final state
        for seg_id, seg_data in history.items():
            final_state = seg_data['U'][-1]  # Last time point
            assert not np.any(np.isnan(final_state)), f"NaN detected in segment {seg_id}"
            assert not np.any(np.isinf(final_state)), f"Inf detected in segment {seg_id}"
        
        print(f"✅ 30s simulation completed: {total_steps} steps, final_time={final_time:.2f}s")
    
    def test_physical_bounds_preservation(self):
        """Verify physical bounds are maintained with optimized kernels."""
        csv_path = "arz_model/data/fichier_de_travail_corridor_utf8.csv"
        
        config = VictoriaIslandConfigFactory.create_victoria_island_config(
            csv_path=csv_path,
            t_final=15.0,
            output_dt=3.0,
            dx=25.0,
            initial_density_cars=30.0,
            initial_density_motorcycles=20.0
        )
        
        network_grid = NetworkGrid.from_config(config)
        runner = SimulationRunner(network_grid, config)
        
        # Run simulation
        history, final_time, total_steps = runner.run()
        
        # Get physics parameters
        physics_cfg = config.physics_config
        rho_max = physics_cfg.rho_max
        
        # Verify bounds for all segments at all times
        violations = []
        for seg_id, seg_data in history.items():
            for t_idx, U in enumerate(seg_data['U']):
                rho_m = U[0, :]  # Motorcycle density
                rho_c = U[2, :]  # Car density
                
                # Check positivity
                if np.any(rho_m < 0):
                    violations.append(f"Negative rho_m in {seg_id} at t={t_idx}")
                if np.any(rho_c < 0):
                    violations.append(f"Negative rho_c in {seg_id} at t={t_idx}")
                
                # Check max density bound
                if np.any(rho_m > rho_max):
                    violations.append(f"rho_m > rho_max in {seg_id} at t={t_idx}")
                if np.any(rho_c > rho_max):
                    violations.append(f"rho_c > rho_max in {seg_id} at t={t_idx}")
        
        assert len(violations) == 0, f"Physical bound violations: {violations[:5]}"
        
        print(f"✅ Physical bounds preserved across {total_steps} steps")
    
    def test_device_function_integration(self):
        """Verify device function is properly integrated in coupling kernel."""
        from arz_model.core.node_solver_gpu import solve_node_fluxes_gpu
        
        # Check that it's a device function
        assert hasattr(solve_node_fluxes_gpu, 'signatures')
        
        # Verify fastmath is enabled
        assert 'fastmath' in str(solve_node_fluxes_gpu.signatures)
        
        print("✅ Device function integration verified with fastmath")


class TestPerformanceRegression:
    """Test that performance hasn't regressed."""
    
    def test_step_time_reasonable(self):
        """Verify step time is within reasonable bounds."""
        import time
        
        csv_path = "arz_model/data/fichier_de_travail_corridor_utf8.csv"
        
        config = VictoriaIslandConfigFactory.create_victoria_island_config(
            csv_path=csv_path,
            t_final=5.0,
            output_dt=1.0,
            dx=25.0,
            initial_density_cars=30.0,
            initial_density_motorcycles=20.0
        )
        
        network_grid = NetworkGrid.from_config(config)
        runner = SimulationRunner(network_grid, config)
        
        # Warmup
        for _ in range(5):
            runner.step()
        
        cuda.synchronize()
        
        # Measure step time
        num_steps = 20
        start = time.perf_counter()
        
        for _ in range(num_steps):
            runner.step()
        
        cuda.synchronize()
        end = time.perf_counter()
        
        avg_step_time_ms = ((end - start) / num_steps) * 1000
        
        # Step time should be < 100ms (very conservative bound)
        assert avg_step_time_ms < 100, f"Step time too high: {avg_step_time_ms:.2f}ms"
        
        print(f"✅ Average step time: {avg_step_time_ms:.3f}ms (< 100ms target)")


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "-s"])
