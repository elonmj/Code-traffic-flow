#!/usr/bin/env python3
"""
Script de lancement pour upload et monitoring Kaggle - Section 7.6 RL Performance

Tests Revendication R5 (Performance superieure des agents RL)

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
    print("  - Training: 10 timesteps only")
    print("  - Duration: 10 minutes simulated time")
    print("  - Scenarios: 1 scenario (traffic_light_control)")
    print("  - Expected runtime: ~15 minutes on GPU")
else:
    print("LANCEMENT VALIDATION KAGGLE GPU - SECTION 7.6 RL PERFORMANCE")
    print("=" * 80)

# Initialiser le manager
print("\n[1/3] Initialisation du ValidationKaggleManager...")
manager = ValidationKaggleManager()

print(f"\n[INFO] Configuration:")
print(f"  - Repository: {manager.repo_url}")
print(f"  - Branch: {manager.branch}")
print(f"  - Username: {manager.username}")
if quick_test:
    print(f"  - Mode: QUICK TEST")
    print(f"  - Durée estimée: 15 minutes sur GPU")
else:
    print(f"  - Mode: FULL TEST")
    print(f"  - Durée estimée: 30-45 minutes sur GPU")

# Lancer la validation section 7.6
print("\n[2/3] Lancement de la validation section 7.6...")
print("  Revendication testée: R5 (Performance RL > Baselines)")
print("\n  Tests inclus:")
print("    - Comparaison RL vs. Baseline pour 3 scénarios de contrôle:")
print("      1. Contrôle de feux de signalisation")
print("      2. Ramp metering (dosage d'accès)")
print("      3. Contrôle adaptatif de vitesse")
print("\n  Outputs générés:")
print("    - 2 figures PNG (comparaison performance, courbe d'apprentissage)")
print("    - 1 CSV avec les métriques détaillées")
print("    - Contenu LaTeX pour la section 7.6 de la thèse")

try:
    # Set environment variable for quick test mode
    if quick_test:
        os.environ['QUICK_TEST'] = 'true'
    
    timeout = 1800 if quick_test else 7200  # 30 min for quick, 2 hours for full
    commit_msg = "Quick test: RL-ARZ integration validation (10 steps)" if quick_test else None
    
    success, kernel_slug = manager.run_validation_section(
        section_name="section_7_6_rl_performance",
        timeout=timeout,
        commit_message=commit_msg
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