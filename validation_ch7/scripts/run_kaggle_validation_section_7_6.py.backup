#!/usr/bin/env python3
"""
Script de lancement pour upload et monitoring Kaggle - Section 7.6 RL Performance

Tests Revendication R5 (Performance superieure des agents RL)

CONTEXTE BÉNINOIS:
    La baseline fixed-time reflète le seul système de contrôle déployé au Bénin.
    Cette validation démontre l'amélioration apportée par le RL dans ce contexte local.
    L'absence de systèmes actuated/adaptatifs reflète la réalité de l'infrastructure béninoise.

Usage:
    python run_kaggle_validation_section_7_6.py           # Full test (2 hours)
    python run_kaggle_validation_section_7_6.py --quick   # Quick test (15 min)
"""

import sys
import os
from pathlib import Path

# Ajout du chemin projet
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from validation_ch7.scripts.validation_kaggle_manager import ValidationKaggleManager

# Check for quick test mode
quick_test = '--quick' in sys.argv or '--quick-test' in sys.argv

print("=" * 80)
if quick_test:
    print("LANCEMENT VALIDATION KAGGLE GPU - SECTION 7.6 RL (QUICK TEST)")
    print("=" * 80)
    print("\n[QUICK TEST MODE]")
    print("  - Training: 100 timesteps (realistic quick test)")
    print("  - Duration: 2 minutes simulated time per episode")
    print("  - Scenarios: 1 scenario (traffic_light_control)")
    print("  - Expected runtime: ~15 minutes on GPU")
else:
    print("LANCEMENT VALIDATION KAGGLE GPU - SECTION 7.6 RL PERFORMANCE")
    print("=" * 80)
    print("\n[FULL MODE]")
    print("  - Training: 5000 timesteps (quality training)")
    print("  - Duration: 1 hour simulated time per episode")
    print("  - Scenarios: 3 scenarios")
    print("  - Expected runtime: ~3-4 hours on GPU")

# Initialiser le manager
print("\n[1/3] Initialisation du ValidationKaggleManager...")
manager = ValidationKaggleManager()

print(f"\n[INFO] Configuration:")
print(f"  - Repository: {manager.repo_url}")
print(f"  - Branch: {manager.branch}")
print(f"  - Username: {manager.username}")
if quick_test:
    print(f"  - Mode: QUICK TEST (100 timesteps)")
    print(f"  - Durée estimée: 15 minutes sur GPU")
else:
    print(f"  - Mode: FULL TEST (5000 timesteps)")
    print(f"  - Durée estimée: 3-4 heures sur GPU")
    print(f"  - NOTE: Utilisez --quick pour test rapide")

# Lancer la validation section 7.6
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

try:
    # Set environment variable for quick test mode (LOCAL execution only)
    if quick_test:
        os.environ['QUICK_TEST'] = 'true'
    
    timeout = 1800 if quick_test else 14400  # 30 min for quick, 4 hours for full (5000 steps takes time)
    commit_msg = "Quick test: RL-ARZ integration validation (100 steps)" if quick_test else "Validation 7.6: RL Performance (5000 timesteps)"
    
    # Pass quick_test flag to manager for Kaggle kernel environment
    success, kernel_slug = manager.run_validation_section(
        section_name="section_7_6_rl_performance",
        timeout=timeout,
        commit_message=commit_msg,
        quick_test=quick_test  # CRITICAL: Pass to Kaggle kernel
    )
    
    if success:
        print("\n" + "=" * 80)
        print("[SUCCESS] VALIDATION KAGGLE 7.6 TERMINÉE")
        print("=" * 80)
        print(f"\n[INFO] Kernel: {kernel_slug}")
        print(f"[URL] https://www.kaggle.com/code/{kernel_slug}")
        print("\n[3/3] Résultats téléchargés et structurés.")
        
        print("\n[NEXT] Pour intégrer dans la thèse:")
        print("  Dans chapters/partie3/ch7_validation_entrainement.tex, ajouter:")
        print("  \\input{validation_output/results/.../section_7_6_rl_performance/latex/section_7_6_content.tex}")
        
    else:
        print("\n[ERROR] Validation échouée - vérifier les logs Kaggle.")
        print(f"[URL] https://www.kaggle.com/code/{kernel_slug}")
        sys.exit(1)
        
except KeyboardInterrupt:
    print("\n\n[INTERRUPTED] Validation interrompue par l'utilisateur.")
    print("Le kernel Kaggle continue de s'exécuter en arrière-plan.")
    sys.exit(130)
    
except Exception as e:
    print(f"\n[ERROR] Erreur inattendue: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)