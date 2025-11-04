#!/usr/bin/env python3
"""
GPU Stability Test - Section Test Rapid

Test l'instabilité du modèle ARZ avec inflow BC à haute vitesse sur GPU.
But: Vérifier si GPU + petits pas de temps peuvent résoudre l'instabilité v→367 m/s.

CONTEXTE:
    - CPU: dt=0.001s → instabilité à v_m≥6.5 m/s (explosion v→367 m/s)
    - Solution proposée: GPU + dt=0.0001s (10x plus petit)
    - Littérature: GPU permet dt plus petits grâce au speedup (voir recherche web)

Configuration:
    - BC inflow: v_m = 10.0 m/s (très haute vitesse, instable sur CPU)
    - Timestep: dt = 0.0001s (10x plus petit que standard)
    - Duration: 15 secondes simulées
    - Expected: Vitesse stable (v_max < 20 m/s), congestion se forme (rho > 0.08)
    
Usage:
    python run_gpu_stability_test.py           # Lance sur Kaggle GPU
    python run_gpu_stability_test.py --quick   # Test rapide (5s au lieu de 15s)
"""

import sys
import subprocess
from pathlib import Path

def main():
    # Check for quick test mode
    quick_test = '--quick' in sys.argv
    
    print("=" * 80)
    if quick_test:
        print("GPU STABILITY TEST - QUICK MODE")
        print("=" * 80)
        print("\n[QUICK TEST]")
        print("  - Duration: 5 seconds simulated")
        print("  - Timestep: dt=0.0001s")
        print("  - BC inflow: v_m=10.0 m/s")
        print("  - Expected runtime: ~5 minutes on Kaggle GPU")
    else:
        print("GPU STABILITY TEST - FULL MODE")
        print("=" * 80)
        print("\n[FULL TEST]")
        print("  - Duration: 15 seconds simulated (150,000 timesteps!)")
        print("  - Timestep: dt=0.0001s (10x plus petit)")
        print("  - BC inflow: v_m=10.0 m/s (instable sur CPU)")
        print("  - Expected runtime: ~15 minutes on Kaggle GPU")
    
    print(f"\n[INFO] Configuration:")
    print(f"  - Test: Instabilité inflow BC à haute vitesse")
    print(f"  - Problème CPU: v→367 m/s avec dt=0.001s")
    print(f"  - Solution GPU: dt=0.0001s (10x réduction)")
    print(f"  - Durée: {'5s' if quick_test else '15s'} simulées")
    print(f"  - Succès attendu: v_max < 20 m/s, rho > 0.08")
    
    print("\n[RESEARCH] Littérature GPU + petits pas de temps:")
    print("  ✅ GPU permet dt plus petits grâce au speedup (Springer, ETH Zurich)")
    print("  ✅ Techniques WENO5-GPU optimisées pour équations hyperboliques")
    print("  ✅ Stabilité numérique améliorée avec dt réduit sur GPU")
    print("  Ref: arXiv:2211.13295, Springer s42967-022-00235-9")
    
    # Delegate to validation_cli.py
    print("\n[DELEGATE] Lancement via validation_cli.py...")
    
    cli_path = Path(__file__).parent / "validation_cli.py"
    
    # Generate commit message
    if quick_test:
        commit_msg = "Quick GPU stability test: v_m=10 m/s, dt=0.0001s (5s)"
    else:
        commit_msg = "GPU stability test: v_m=10 m/s, dt=0.0001s (15s)"
    
    # Build command
    cmd = [
        sys.executable,
        str(cli_path),
        "--section", "gpu_stability_test",
        "--commit-message", commit_msg
    ]
    
    if quick_test:
        cmd.append("--quick-test")
    
    # Execute
    try:
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("[SUCCESS] GPU STABILITY TEST TERMINÉ")
            print("=" * 80)
            print("\n[RESULTS] Téléchargés dans validation_output/")
            print("\n[ANALYSE] Vérifier:")
            print("  - v_max final < 20 m/s → STABLE ✅")
            print("  - v_max final > 100 m/s → INSTABLE ❌")
            print("  - rho_max > 0.08 → Congestion formée ✅")
            return 0
        else:
            print("\n[ERROR] Test échoué - vérifier logs Kaggle")
            return result.returncode
            
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test interrompu")
        return 130
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
