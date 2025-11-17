#!/usr/bin/env python3
"""
Kaggle Benchmark: GPU Batched Architecture
==========================================

Mesure le temps réel d'exécution de la simulation Victoria Island avec l'architecture batched.

OBJECTIF:
- Comparer vs baseline CPU (413s pour 240s simulation)
- Comparer vs Phase 2.5 GPU (520s pour 120s simulation)
- Vérifier l'élimination des NumbaPerformanceWarning
- Mesurer le speedup réel

VALEURS DE RÉFÉRENCE:
- Baseline CPU: 413s pour 240s sim = 1.72 s/sim-s
- Phase 2.5 GPU: 520s pour 120s sim = 4.34 s/sim-s (2.5× PLUS LENT que baseline!)

OBJECTIF BATCHED:
- Target: <200s pour 240s sim = 0.83 s/sim-s
- Speedup vs baseline: 2-3×
- Speedup vs Phase 2.5: 4-6×
- Warnings: 0 (100% → 0%)
"""

import os
import sys
import time
import pickle
import warnings
import io
from contextlib import redirect_stdout, redirect_stderr

print("=" * 80)
print("KAGGLE BENCHMARK: GPU BATCHED ARCHITECTURE")
print("=" * 80)
print()

# Capture warnings
warning_capture = []
def warning_handler(message, category, filename, lineno, file=None, line=None):
    warning_str = f"{category.__name__}: {message}"
    warning_capture.append(warning_str)
    # Still print to console
    print(f"⚠️  {warning_str}")

warnings.showwarning = warning_handler

# Add project root to path
project_root = '/kaggle/working/arz_model'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"[INFO] Python: {sys.version}")
print(f"[INFO] Working directory: {os.getcwd()}")
print(f"[INFO] Project root: {project_root}")
print()

# Import after path setup
from arz_model.config.config_factory import create_victoria_island_config
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner

def count_numba_warnings():
    """Count NumbaPerformanceWarning occurrences."""
    numba_warnings = [w for w in warning_capture if 'NumbaPerformanceWarning' in w]
    return len(numba_warnings)

def main():
    print("=" * 80)
    print("PHASE 1: CONFIGURATION")
    print("=" * 80)
    
    # Configuration identique au baseline
    config = create_victoria_island_config(
        default_density=20.0,   # veh/km - light baseline traffic
        default_velocity=50.0,  # km/h - moderate speed
        inflow_density=30.0,    # veh/km - entry traffic
        inflow_velocity=40.0,   # km/h - entry speed
        t_final=120.0,          # 2 minutes (même que Phase 2.5 pour comparaison)
        output_dt=2.0,          # Output every 2 seconds
        cells_per_100m=10       # Grid resolution
    )
    
    print(f"✅ Configuration créée:")
    print(f"   - Simulation time: {config.time.t_final}s")
    print(f"   - Output interval: {config.time.output_dt}s")
    print(f"   - CFL factor: {config.time.cfl_factor}")
    print()
    
    print("=" * 80)
    print("PHASE 2: NETWORK BUILDING")
    print("=" * 80)
    
    network_grid = NetworkGrid.from_config(config)
    print(f"✅ Network built:")
    print(f"   - Segments: {len(network_grid.segments)}")
    print(f"   - Nodes: {len(network_grid.nodes)}")
    print()
    
    print("=" * 80)
    print("PHASE 3: SIMULATION RUNNER INITIALIZATION")
    print("=" * 80)
    
    runner = SimulationRunner(
        network_grid=network_grid,
        simulation_config=config,
        debug=False
    )
    print("✅ Runner initialized")
    print()
    
    print("=" * 80)
    print("PHASE 4: RUNNING SIMULATION WITH BATCHED GPU ARCHITECTURE")
    print("=" * 80)
    print()
    print("🚀 Starting simulation...")
    print(f"   Simulation time: {config.time.t_final}s")
    print(f"   Expected frames: {int(config.time.t_final / config.time.output_dt)}")
    print()
    
    # Clear warning capture
    warning_capture.clear()
    
    # Mesure du temps
    wall_start = time.time()
    
    try:
        results = runner.run(timeout=None)
        wall_end = time.time()
        wall_elapsed = wall_end - wall_start
        
        print()
        print("=" * 80)
        print("✅ SIMULATION COMPLETE")
        print("=" * 80)
        
        # Analyse des résultats
        sim_time = config.time.t_final
        time_per_sim_second = wall_elapsed / sim_time
        
        print()
        print("📊 PERFORMANCE METRICS:")
        print(f"   Simulation time: {sim_time:.1f}s")
        print(f"   Wall clock time: {wall_elapsed:.1f}s")
        print(f"   Time per sim-second: {time_per_sim_second:.3f} s/sim-s")
        print()
        
        # Comparaison avec baseline
        baseline_time_per_s = 1.72  # 413s / 240s
        phase25_time_per_s = 4.34   # 520s / 120s
        
        speedup_vs_baseline = baseline_time_per_s / time_per_sim_second
        speedup_vs_phase25 = phase25_time_per_s / time_per_sim_second
        
        print("🎯 SPEEDUP ANALYSIS:")
        print(f"   vs Baseline CPU (1.72 s/sim-s): {speedup_vs_baseline:.2f}×")
        print(f"   vs Phase 2.5 GPU (4.34 s/sim-s): {speedup_vs_phase25:.2f}×")
        print()
        
        # Analyse des warnings
        numba_warning_count = count_numba_warnings()
        total_warnings = len(warning_capture)
        
        print("⚠️  WARNING ANALYSIS:")
        print(f"   Total warnings: {total_warnings}")
        print(f"   NumbaPerformanceWarning: {numba_warning_count}")
        
        if numba_warning_count == 0:
            print("   ✅ ZERO NumbaPerformanceWarning - OBJECTIF ATTEINT!")
        else:
            print(f"   ❌ Still {numba_warning_count} NumbaPerformanceWarning")
            print("   First 5 warnings:")
            for w in warning_capture[:5]:
                print(f"      - {w}")
        print()
        
        # Verdict final
        print("=" * 80)
        print("📈 FINAL VERDICT:")
        print("=" * 80)
        
        success_criteria = []
        
        # Critère 1: Warnings éliminés
        if numba_warning_count == 0:
            success_criteria.append("✅ NumbaPerformanceWarning eliminated (100% → 0%)")
        else:
            success_criteria.append(f"❌ Still {numba_warning_count} warnings")
        
        # Critère 2: Speedup vs baseline
        if speedup_vs_baseline >= 2.0:
            success_criteria.append(f"✅ Speedup vs baseline: {speedup_vs_baseline:.2f}× (target: 2-3×)")
        else:
            success_criteria.append(f"⚠️  Speedup vs baseline: {speedup_vs_baseline:.2f}× (target: 2-3×)")
        
        # Critère 3: Speedup vs Phase 2.5
        if speedup_vs_phase25 >= 4.0:
            success_criteria.append(f"✅ Speedup vs Phase 2.5: {speedup_vs_phase25:.2f}× (target: 4-6×)")
        else:
            success_criteria.append(f"⚠️  Speedup vs Phase 2.5: {speedup_vs_phase25:.2f}× (target: 4-6×)")
        
        # Critère 4: Temps absolu
        if wall_elapsed < 200:
            success_criteria.append(f"✅ Wall time: {wall_elapsed:.1f}s < 200s target")
        else:
            success_criteria.append(f"⚠️  Wall time: {wall_elapsed:.1f}s (target: <200s for 240s sim)")
        
        print()
        for criterion in success_criteria:
            print(f"   {criterion}")
        print()
        
        # Sauvegarde des résultats
        output_path = '/kaggle/working/batched_gpu_benchmark_results.pkl'
        benchmark_data = {
            'wall_time': wall_elapsed,
            'sim_time': sim_time,
            'time_per_sim_s': time_per_sim_second,
            'speedup_vs_baseline': speedup_vs_baseline,
            'speedup_vs_phase25': speedup_vs_phase25,
            'numba_warnings': numba_warning_count,
            'total_warnings': total_warnings,
            'warning_list': warning_capture[:20],  # First 20 warnings
            'success_criteria': success_criteria,
            'results': results
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(benchmark_data, f)
        
        print(f"💾 Benchmark data saved to: {output_path}")
        print()
        
        # Return code basé sur le succès
        all_success = all('✅' in c for c in success_criteria)
        
        if all_success:
            print("🎉 ALL SUCCESS CRITERIA MET!")
            return 0
        else:
            print("⚠️  Some criteria not met - see analysis above")
            return 1
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ SIMULATION FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
