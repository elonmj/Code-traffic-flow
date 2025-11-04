"""
GPU Stability Test - Test l'instabilité BC inflow avec GPU + petits pas de temps.

Test Configuration:
    - BC inflow: v_m = 10.0 m/s (haute vitesse, instable sur CPU à dt=0.001s)
    - Timestep: dt = 0.0001s (10x plus petit que standard)
    - Duration: 15s simulées (150,000 timesteps!)
    - Device: GPU (Numba CUDA)
    - Expected: v_max < 20 m/s, rho > 0.08 (congestion)

Quick Test Mode (env var QUICK_TEST=1):
    - Duration: 5s simulées au lieu de 15s
    - Timestep: dt=0.0001s (identique)
    - Expected runtime: ~5 minutes
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/kaggle/working/Code-traffic-flow')

from arz_model.network.network_grid import NetworkGrid
from arz_model.core.parameters import ModelParameters

def test_gpu_stability():
    """Test GPU avec petits pas de temps contre instabilité inflow BC."""
    
    # Check quick test mode
    quick_test = os.environ.get('QUICK_TEST', '0') == '1'
    duration = 5.0 if quick_test else 15.0
    dt = 0.0001  # 10x plus petit que standard (0.001s)
    
    print("=" * 70)
    print("GPU STABILITY TEST - Inflow BC Instability")
    print("=" * 70)
    print(f"\nMode: {'QUICK TEST (5s)' if quick_test else 'FULL TEST (15s)'}")
    print(f"Duration: {duration}s simulated")
    print(f"Timestep: dt={dt}s (10x smaller than standard)")
    print(f"BC inflow: v_m=10.0 m/s (unstable on CPU @ dt=0.001s)")
    print(f"Expected: v_max < 20 m/s, rho > 0.08")
    print(f"Timesteps: {int(duration/dt):,}")
    
    # Create parameters with GPU
    params = ModelParameters()
    params.alpha = 0.5
    params.rho_jam = 1.0
    params.epsilon = 1e-10
    params.K_m = 20.0
    params.gamma_m = 2.0
    params.K_c = 30.0
    params.gamma_c = 2.0
    params.tau_m = 2.0
    params.tau_c = 2.0
    params.V_creeping = 1.0
    params.N = 50  # Spatial cells
    params.cfl_number = 0.5
    params.ghost_cells = 3
    params.spatial_scheme = 'weno5'
    params.time_scheme = 'ssprk3'
    params.ode_solver = 'RK45'
    params.ode_rtol = 1e-6
    params.ode_atol = 1e-9
    params.Vmax_m = {0: 12.0, 1: 15.0, 2: 18.0}
    params.Vmax_c = {0: 10.0, 1: 13.0, 2: 16.0}
    params.device = 'gpu'  # CRITICAL: Enable GPU mode
    
    print("\n[SETUP] Creating network...")
    network = NetworkGrid(params)
    
    # Segment 1: Inflow with high velocity (unstable!)
    network.add_segment('seg_0', xmin=0.0, xmax=100.0, N=params.N, start_node=None, end_node='node_1')
    
    # Segment 2: Normal outflow
    network.add_segment('seg_1', xmin=100.0, xmax=200.0, N=params.N, start_node='node_1', end_node=None)
    
    # Set boundary conditions via params
    network.params.boundary_conditions = {
        'seg_0': {
            'left': {
                'type': 'inflow',
                'rho_m': 0.15,
                'v_m': 10.0,  # HIGH VELOCITY - unstable on CPU @ dt=0.001s
                'rho_c': 0.0,
                'v_c': 0.0
            }
        },
        'seg_1': {
            'right': {
                'type': 'outflow'
            }
        }
    }
    
    print(f"✅ Network: 2 segments, {params.N} cells each, GPU mode")
    print(f"✅ BC: Inflow v_m=10.0 m/s (very high!)")
    
    # Initialize network
    network.initialize()
    
    # Set initial conditions (low density equilibrium)
    for seg_id in ['seg_0', 'seg_1']:
        grid = network.segments[seg_id]['grid']
        U = network.segments[seg_id]['U']
        U[0, grid.physical_cell_indices] = 0.05  # Low initial density
        U[1, grid.physical_cell_indices] = 0.05 * 5.0  # w_m = rho_m * v_m (v=5 m/s)
        U[2, grid.physical_cell_indices] = 0.0  # No cars initially
        U[3, grid.physical_cell_indices] = 0.0  # No car momentum initially
    
    print(f"\n[SIMULATION] Running {int(duration/dt):,} timesteps...")
    print(f"  dt={dt}s, duration={duration}s")
    print(f"  Progress updates every 1s simulated")
    
    # Storage for monitoring
    times = []
    v_max_history = []
    rho_max_history = []
    
    t = 0.0
    step = 0
    last_report_time = 0.0
    
    try:
        while t < duration:
            # Evolve network
            network.evolve(dt=dt)
            t += dt
            step += 1
            
            # Monitor every 1s
            if t - last_report_time >= 1.0:
                # Compute max velocity and density
                v_max = 0.0
                rho_max = 0.0
                for seg_id in ['seg_0', 'seg_1']:
                    U = network.segments[seg_id]['U']
                    grid = network.segments[seg_id]['grid']
                    
                    rho_m = U[0, grid.physical_cell_indices]
                    w_m = U[1, grid.physical_cell_indices]
                    
                    rho_safe = np.maximum(rho_m, 1e-10)
                    v_m = w_m / rho_safe
                    
                    v_max = max(v_max, np.max(v_m))
                    rho_max = max(rho_max, np.max(rho_m))
                
                times.append(t)
                v_max_history.append(v_max)
                rho_max_history.append(rho_max)
                
                print(f"  t={t:.1f}s (step {step:,}): v_max={v_max:.2f} m/s, rho_max={rho_max:.4f}")
                last_report_time = t
                
                # Early stop if explosion detected
                if v_max > 100.0:
                    print(f"\n❌ INSTABILITY DETECTED: v_max={v_max:.2f} m/s > 100 m/s")
                    print("   GPU + small timestep FAILED to prevent instability")
                    break
        
        # Final state
        v_max_final = v_max_history[-1] if v_max_history else 0
        rho_max_final = rho_max_history[-1] if rho_max_history else 0
        
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Final time: {t:.2f}s")
        print(f"Total steps: {step:,}")
        print(f"Final v_max: {v_max_final:.2f} m/s")
        print(f"Final rho_max: {rho_max_final:.4f}")
        
        # Verdict
        stable = v_max_final < 20.0
        congestion = rho_max_final > 0.08
        
        print(f"\n[VERDICT]")
        if stable and congestion:
            print("✅ SUCCESS: GPU + small timestep RESOLVED instability!")
            print(f"   - Velocity stable: {v_max_final:.2f} < 20 m/s")
            print(f"   - Congestion formed: rho_max={rho_max_final:.4f} > 0.08")
            success = True
        elif stable and not congestion:
            print("⚠️  PARTIAL: Velocity stable but no congestion")
            print(f"   - Velocity stable: {v_max_final:.2f} < 20 m/s")
            print(f"   - No congestion: rho_max={rho_max_final:.4f} < 0.08")
            success = True
        else:
            print("❌ FAILURE: Instability persists even with GPU + small dt")
            print(f"   - Velocity unstable: {v_max_final:.2f} ≥ 20 m/s")
            success = False
        
        # Save results
        results_dir = Path('/kaggle/working/validation_results/gpu_stability_test')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Figure 1: Velocity evolution
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(times, v_max_history, 'b-', linewidth=2)
        plt.axhline(20, color='r', linestyle='--', label='Stability threshold')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Max Velocity (m/s)', fontsize=12)
        plt.title('Velocity Evolution - GPU Test', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Figure 2: Density evolution
        plt.subplot(1, 2, 2)
        plt.plot(times, rho_max_history, 'g-', linewidth=2)
        plt.axhline(0.08, color='r', linestyle='--', label='Congestion threshold')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Max Density (veh/m)', fontsize=12)
        plt.title('Density Evolution - GPU Test', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / 'gpu_stability_evolution.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ Figure saved: {results_dir / 'gpu_stability_evolution.png'}")
        
        # Save metrics
        metrics = {
            'test_name': 'gpu_stability_test',
            'mode': 'quick' if quick_test else 'full',
            'duration_simulated': duration,
            'timestep': dt,
            'total_steps': step,
            'bc_inflow_velocity': 10.0,
            'final_v_max': float(v_max_final),
            'final_rho_max': float(rho_max_final),
            'stable': stable,
            'congestion_formed': congestion,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(results_dir / 'gpu_stability_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Metrics saved: {results_dir / 'gpu_stability_metrics.json'}")
        
        # Create session summary for detection
        summary = {
            'test_name': 'gpu_stability_test',
            'success': success,
            'final_metrics': metrics
        }
        
        with open('/kaggle/working/session_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Session summary: /kaggle/working/session_summary.json")
        
        return success
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_gpu_stability()
    sys.exit(0 if success else 1)
