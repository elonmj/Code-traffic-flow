#!/usr/bin/env python3
"""
Script de diagnostic CPU vs GPU avec schémas numériques simples
Test pour isoler si le problème vient de WENO5/SSP-RK3 ou d'ailleurs
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Configuration du projet
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_simple_cpu_gpu_test():
    """Exécute un test simple CPU vs GPU pour diagnostic"""
    
    print("🔍 DIAGNOSTIC CPU vs GPU - Schémas Numériques Simples")
    print("=" * 60)
    
    try:
        # Import modules
        from code.simulation.runner import SimulationRunner
        from code.core.parameters import ModelParameters
        
        # Configuration du scénario simple
        scenario_file = "config/scenario_simple_cpu_gpu_test.yml"
        
        if not os.path.exists(scenario_file):
            print(f"❌ Fichier de scénario non trouvé: {scenario_file}")
            return
        
        print(f"📋 Scénario utilisé: {scenario_file}")
        print("   - Schéma spatial: first_order (PAS WENO5)")
        print("   - Schéma temporel: euler (PAS SSP-RK3)")
        print("   - Grille: 100 points, domaine 500m")
        print("   - Temps: 5s, CFL=0.3")
        print()
        
        # === TEST CPU ===
        print("🖥️  TEST CPU")
        print("-" * 20)
        
        runner_cpu = SimulationRunner(
            scenario_config_path=scenario_file,
            device='cpu',
            quiet=True
        )
        
        start_time = time.time()
        times_cpu, states_cpu = runner_cpu.run()
        cpu_duration = time.time() - start_time
        
        print(f"✅ Simulation CPU terminée en {cpu_duration:.2f}s")
        
        # Sauvegarde résultats CPU
        output_dir = Path("output_simple_test")
        output_dir.mkdir(exist_ok=True)
        
        cpu_file = output_dir / "simple_cpu_results.npz"
        
        # Convertir les résultats en format analysable
        states_array_cpu = np.stack(states_cpu, axis=0)  # Shape: (nt, 4, nx)
        np.savez_compressed(cpu_file, 
                           times=np.array(times_cpu),
                           rho_m=states_array_cpu[:, 0, :],
                           w_m=states_array_cpu[:, 1, :],
                           rho_c=states_array_cpu[:, 2, :],
                           w_c=states_array_cpu[:, 3, :])
        print(f"💾 Résultats CPU sauvés: {cpu_file}")
        
        # === TEST GPU ===
        print()
        print("🚀 TEST GPU")
        print("-" * 20)
        
        runner_gpu = SimulationRunner(
            scenario_config_path=scenario_file,
            device='gpu',
            quiet=True
        )
        
        start_time = time.time()
        times_gpu, states_gpu = runner_gpu.run()
        gpu_duration = time.time() - start_time
        
        print(f"✅ Simulation GPU terminée en {gpu_duration:.2f}s")
        
        # Sauvegarde résultats GPU
        gpu_file = output_dir / "simple_gpu_results.npz"
        
        # Convertir les résultats en format analysable
        states_array_gpu = np.stack(states_gpu, axis=0)  # Shape: (nt, 4, nx)
        np.savez_compressed(gpu_file, 
                           times=np.array(times_gpu),
                           rho_m=states_array_gpu[:, 0, :],
                           w_m=states_array_gpu[:, 1, :],
                           rho_c=states_array_gpu[:, 2, :],
                           w_c=states_array_gpu[:, 3, :])
        print(f"💾 Résultats GPU sauvés: {gpu_file}")
        
        # === COMPARAISON RAPIDE ===
        print()
        print("📊 COMPARAISON RAPIDE")
        print("-" * 30)
        
        # Chargement des données
        with np.load(cpu_file) as data_cpu:
            rho_m_cpu = data_cpu['rho_m']
            w_m_cpu = data_cpu['w_m']
            rho_c_cpu = data_cpu['rho_c']
            w_c_cpu = data_cpu['w_c']
            
        with np.load(gpu_file) as data_gpu:
            rho_m_gpu = data_gpu['rho_m']
            w_m_gpu = data_gpu['w_m']
            rho_c_gpu = data_gpu['rho_c']
            w_c_gpu = data_gpu['w_c']
        
        # Calcul des erreurs (état final uniquement)
        rho_m_final_cpu = rho_m_cpu[-1, :]
        rho_m_final_gpu = rho_m_gpu[-1, :]
        w_m_final_cpu = w_m_cpu[-1, :]
        w_m_final_gpu = w_m_gpu[-1, :]
        
        # Erreurs absolues
        error_rho_m = np.abs(rho_m_final_cpu - rho_m_final_gpu)
        error_w_m = np.abs(w_m_final_cpu - w_m_final_gpu)
        
        max_error_rho_m = np.max(error_rho_m)
        mean_error_rho_m = np.mean(error_rho_m)
        max_error_w_m = np.max(error_w_m)
        mean_error_w_m = np.mean(error_w_m)
        
        print(f"Performance:")
        print(f"  - Temps CPU: {cpu_duration:.2f}s")
        print(f"  - Temps GPU: {gpu_duration:.2f}s")
        print(f"  - Speedup: {cpu_duration/gpu_duration:.2f}x")
        print()
        print(f"Précision (état final):")
        print(f"  - Erreur max rho_m: {max_error_rho_m:.2e}")
        print(f"  - Erreur moy rho_m: {mean_error_rho_m:.2e}")
        print(f"  - Erreur max w_m: {max_error_w_m:.2e}")
        print(f"  - Erreur moy w_m: {mean_error_w_m:.2e}")
        
        # Test de proximité
        tolerance = 1e-10
        is_close_rho = np.allclose(rho_m_final_cpu, rho_m_final_gpu, atol=tolerance, rtol=tolerance)
        is_close_w = np.allclose(w_m_final_cpu, w_m_final_gpu, atol=tolerance, rtol=tolerance)
        
        print(f"  - Test proximité (tol={tolerance:.0e}): {'✅ PASS' if (is_close_rho and is_close_w) else '❌ FAIL'}")
        
        # === DIAGNOSTIC ===
        print()
        print("🔍 DIAGNOSTIC")
        print("-" * 20)
        
        if max_error_rho_m > 1e-6 or max_error_w_m > 1e-6:
            print("⚠️  ERREURS IMPORTANTES DÉTECTÉES avec schémas simples!")
            print("   Cela suggère un problème fondamental dans:")
            print("   - L'implémentation CPU/GPU de base")
            print("   - Les conditions aux limites")
            print("   - La gestion mémoire GPU")
            print("   - Les conversions de types de données")
        else:
            print("✅ Erreurs faibles avec schémas simples")
            print("   Le problème semble spécifique à WENO5/SSP-RK3")
            
        print()
        print(f"📁 Fichiers de résultats disponibles dans: {output_dir}")
        print("   - simple_cpu_results.npz")
        print("   - simple_gpu_results.npz")
        
        return {
            'cpu_duration': cpu_duration,
            'gpu_duration': gpu_duration,
            'max_error_rho_m': max_error_rho_m,
            'mean_error_rho_m': mean_error_rho_m,
            'max_error_w_m': max_error_w_m,
            'mean_error_w_m': mean_error_w_m,
            'is_close': is_close_rho and is_close_w
        }
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_simple_cpu_gpu_test()
    
    if results:
        print()
        print("=" * 60)
        print("🎯 CONCLUSION:")
        if results['max_error_rho_m'] > 1e-6:
            print("   Problème fondamental CPU/GPU détecté")
            print("   → Investiguer l'implémentation de base")
        else:
            print("   Schémas simples fonctionnent bien")
            print("   → Le problème est spécifique à WENO5/SSP-RK3")
