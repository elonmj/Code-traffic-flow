#!/usr/bin/env python3
"""
Section 7.6 RL Performance Validation - Quick Launcher

Tests Revendication R5 (Performance superieure des agents RL)

CONTEXTE BÉNINOIS:
    La baseline fixed-time reflète le seul système de contrôle déployé au Bénin.
    Cette validation démontre l'amélioration apportée par le RL dans ce contexte local.
    L'absence de systèmes actuated/adaptatifs reflète la réalité de l'infrastructure béninoise.

ARCHITECTURE:
    - This is a WRAPPER that delegates to validation_cli.py
    - Provides section-specific help and quick test support
    - NO duplication of orchestration logic (DRY principle)
    - Single source of truth: validation_cli.py

Usage:
    python run_kaggle_validation_section_7_6.py           # Full test (4 hours)
    python run_kaggle_validation_section_7_6.py --quick   # Quick test (15 min)
"""

import sys
import subprocess
from pathlib import Path

def main():
    # Check for quick test mode (preserved from original)
    quick_test = '--quick' in sys.argv or '--quick-test' in sys.argv
    
    # ✅ NEW: Check for single scenario mode
    # Expected format: --scenario=traffic_light_control or --scenario traffic_light_control
    scenario = None
    valid_scenarios = ['traffic_light_control', 'ramp_metering', 'adaptive_speed_control']
    
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--scenario='):
            scenario = arg.split('=')[1]
        elif arg == '--scenario' and i + 1 < len(sys.argv):
            scenario = sys.argv[i + 1]
    
    # Validate scenario if provided
    if scenario and scenario not in valid_scenarios:
        print(f"[ERROR] Invalid scenario: {scenario}")
        print(f"[ERROR] Valid scenarios: {', '.join(valid_scenarios)}")
        sys.exit(1)
    
    # ============================================================================
    # SECTION-SPECIFIC INFORMATION (preserved from original for user experience)
    # ============================================================================
    print("=" * 80)
    if quick_test:
        print("LANCEMENT VALIDATION KAGGLE GPU - SECTION 7.6 RL (QUICK TEST)")
        print("=" * 80)
        print("\n[QUICK TEST MODE]")
        print("  - Training: 100 timesteps (realistic quick test)")
        print("  - Duration: 2 minutes simulated time per episode")
        print("  - Scenarios: 1 scenario (traffic_light_control)")
        print("  - Expected runtime: ~15 minutes on GPU")
        timeout = 1800  # 30 min
    else:
        print("LANCEMENT VALIDATION KAGGLE GPU - SECTION 7.6 RL PERFORMANCE")
        print("=" * 80)
        print("\n[FULL MODE]")
        print("  - Training: 5000 timesteps (quality training)")
        print("  - Duration: 1 hour simulated time per episode")
        print("  - Scenarios: 3 scenarios")
        print("  - Expected runtime: ~3-4 hours on GPU")
        print("  - NOTE: Utilisez --quick pour test rapide")
        timeout = 14400  # 4 hours
    
    # Display configuration (without manager instantiation - no duplication!)
    print(f"\n[INFO] Configuration:")
    print(f"  - Section: 7.6 RL Performance")
    print(f"  - Revendication: R5 (Performance RL > Baselines)")
    print(f"  - Mode: {'QUICK TEST (100 timesteps)' if quick_test else 'FULL TEST (5000 timesteps)'}")
    print(f"  - Durée estimée: {'15 minutes' if quick_test else '3-4 heures'} sur GPU")
    if scenario:
        print(f"  - Scenario: {scenario} (single scenario mode)")
    else:
        print(f"  - Scenario: Default (traffic_light_control)")
    
    # Test information (preserved from original)
    print("\n[2/3] Lancement de la validation section 7.6...")
    print("  Revendication testée: R5 (Performance RL > Baselines)")
    print("  Contexte: Infrastructure béninoise (baseline fixed-time appropriée)")
    print("\n  Tests inclus:")
    print("    - Comparaison RL vs. Baseline fixed-time pour 3 scénarios:")
    print("      1. Contrôle de feux de signalisation (reflète pratique béninoise)")
    print("      2. Ramp metering (dosage d'accès)")
    print("      3. Contrôle adaptatif de vitesse")
    print("  Note: Baseline reflète le seul système déployé au Bénin")
    print("\n  Outputs générés:")
    print("    - 2 figures PNG (comparaison performance, courbe d'apprentissage)")
    print("    - 1 CSV avec les métriques détaillées")
    print("    - Contenu LaTeX pour la section 7.6 de la thèse")
    
    # ============================================================================
    # DELEGATION TO validation_cli.py (no duplication!)
    # ============================================================================
    print("\n[ORCHESTRATE] Delegating to validation_cli.py...")
    
    cli_path = Path(__file__).parent / "validation_cli.py"
    
    # Generate section-specific commit message (preserved from original)
    if quick_test:
        commit_msg = "Quick test: RL-ARZ integration validation (100 steps)"
    else:
        commit_msg = "Validation 7.6: RL Performance (5000 timesteps)"
    
    # Build command with proper arguments
    cmd = [
        sys.executable,
        str(cli_path),
        "--section", "section_7_6_rl_performance",
        "--timeout", str(timeout),
        "--commit-message", commit_msg
    ]
    
    if quick_test:
        cmd.append("--quick-test")
    
    # ✅ NEW: Add scenario selection if specified
    if scenario:
        cmd.extend(["--scenario", scenario])
    
    # Execute validation_cli.py and capture result
    try:
        result = subprocess.run(cmd)
        
        # Success/failure messages (preserved from original)
        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("[SUCCESS] VALIDATION KAGGLE 7.6 TERMINÉE")
            print("=" * 80)
            print("\n[3/3] Résultats téléchargés et structurés.")
            
            print("\n[NEXT] Pour intégrer dans la thèse:")
            print("  Dans chapters/partie3/ch7_validation_entrainement.tex, ajouter:")
            print("  \\input{validation_output/results/.../section_7_6_rl_performance/latex/section_7_6_content.tex}")
            
            return 0
        else:
            print("\n[ERROR] Validation échouée - vérifier les logs Kaggle.")
            return result.returncode
            
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Validation interrompue par l'utilisateur.")
        print("Le kernel Kaggle continue de s'exécuter en arrière-plan.")
        return 130
        
    except Exception as e:
        print(f"\n[ERROR] Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())