#!/usr/bin/env python3
"""
Script de lancement pour upload et monitoring Kaggle - Section 7.3
"""

import sys
from pathlib import Path

# Ajout du chemin projet
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from validation_kaggle_manager import ValidationKaggleManager

print("=" * 80)
print("LANCEMENT VALIDATION KAGGLE GPU - SECTION 7.3 ANALYTICAL")
print("=" * 80)

# Initialiser le manager (lit automatiquement kaggle.json)
print("\n[1/3] Initialisation du ValidationKaggleManager...")
manager = ValidationKaggleManager()

print(f"\n[INFO] Configuration:")
print(f"  - Repository: {manager.repo_url}")
print(f"  - Branch: {manager.branch}")
print(f"  - Username: {manager.username}")
print(f"  - Durée estimée: 45 minutes sur GPU")

# Lancer la validation section 7.3
print("\n[2/3] Lancement de la validation section 7.3...")
print("  Revendications testées: R1 (convergence WENO5), R3 (solutions analytiques)")
print("  Tests inclus:")
print("    - 5 problèmes de Riemann avec figures PNG")
print("    - Analyse de convergence ordre 5")
print("    - Génération LaTeX automatique")

try:
    success, kernel_slug = manager.run_validation_section(
        section_name="section_7_3_analytical",
        timeout=4000  # 45 min estimé + marge
    )
    
    if success:
        print("\n" + "=" * 80)
        print("[SUCCESS] VALIDATION KAGGLE TERMINÉE")
        print("=" * 80)
        print(f"\n[INFO] Kernel: {kernel_slug}")
        print(f"[URL] https://www.kaggle.com/code/{kernel_slug}")
        print("\n[3/3] Résultats téléchargés:")
        print(f"  - NPZ files: validation_ch7/results/section_7_3_analytical/validation_results/npz/")
        print(f"  - Figures PNG: validation_ch7/results/section_7_3_analytical/validation_results/figures/")
        print(f"  - LaTeX: validation_ch7/results/section_7_3_analytical/validation_results/section_7_3_content.tex")
        print(f"  - Session summary: validation_ch7/results/section_7_3_analytical/validation_results/session_summary.json")
        
        print("\n[NEXT] Pour intégrer dans la thèse:")
        print("  \\input{validation_ch7/results/section_7_3_analytical/validation_results/section_7_3_content.tex}")
        
    else:
        print("\n[ERROR] Validation échouée - vérifier les logs Kaggle")
        sys.exit(1)
        
except KeyboardInterrupt:
    print("\n\n[INTERRUPTED] Validation interrompue par l'utilisateur")
    print("[INFO] Le kernel Kaggle continue à s'exécuter en arrière-plan")
    print("[INFO] Utilisez manager.check_kernel_status(kernel_slug) pour vérifier l'état")
    sys.exit(0)
    
except Exception as e:
    print(f"\n[ERROR] Erreur inattendue: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
