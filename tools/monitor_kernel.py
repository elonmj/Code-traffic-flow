#!/usr/bin/env python3
"""
Script pour monitorer un kernel Kaggle existant
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from validation_ch7.scripts.validation_kaggle_manager import ValidationKaggleManager

# Le kernel qui vient d'être uploadé
kernel_slug = "elonmj/arz-validation-73-jpso"

print("=" * 80)
print(f"MONITORING KERNEL: {kernel_slug}")
print("=" * 80)
print(f"\n[URL] https://www.kaggle.com/code/{kernel_slug}")
print("\nInitialisation du monitoring...")

manager = ValidationKaggleManager()

try:
    # Utiliser directement la méthode de monitoring
    success = manager._monitor_kernel_with_session_detection(
        kernel_slug=kernel_slug,
        timeout=4000  # 66 minutes max
    )
    
    if success:
        print("\n" + "=" * 80)
        print("[SUCCESS] KERNEL TERMINÉ AVEC SUCCÈS")
        print("=" * 80)
        print("\nLes résultats ont été téléchargés automatiquement dans:")
        print("  validation_ch7/results/section_7_3_analytical/validation_results/")
    else:
        print("\n[ERROR] Kernel a échoué ou timeout")
        
except KeyboardInterrupt:
    print("\n\n[INTERRUPTED] Monitoring interrompu")
    print(f"[INFO] Le kernel continue en arrière-plan: https://www.kaggle.com/code/{kernel_slug}")
    
except Exception as e:
    print(f"\n[ERROR] Erreur: {e}")
    import traceback
    traceback.print_exc()
