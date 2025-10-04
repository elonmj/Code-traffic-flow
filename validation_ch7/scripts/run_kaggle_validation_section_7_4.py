#!/usr/bin/env python3
"""
Script de lancement pour upload et monitoring Kaggle - Section 7.4 Calibration

CRITICAL: Uses REAL TomTom data from donnees_trafic_75_segments.csv
NO synthetic data generation - calibration on real Victoria Island data only
"""

import sys
from pathlib import Path

# Ajout du chemin projet
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from validation_ch7.scripts.validation_kaggle_manager import ValidationKaggleManager

print("=" * 80)
print("LANCEMENT VALIDATION KAGGLE GPU - SECTION 7.4 CALIBRATION")
print("=" * 80)

# Initialiser le manager (lit automatiquement kaggle.json)
print("\n[1/3] Initialisation du ValidationKaggleManager...")
manager = ValidationKaggleManager()

print(f"\n[INFO] Configuration:")
print(f"  - Repository: {manager.repo_url}")
print(f"  - Branch: {manager.branch}")
print(f"  - Username: {manager.username}")
print(f"  - Durée estimée: 60 minutes sur GPU")

# CRITICAL: Verify real data files exist before upload
print("\n[CRITICAL] Vérification des fichiers de données réelles...")
csv_data = project_root / "donnees_trafic_75_segments.csv"
network_json = project_root / "arz_model" / "calibration" / "data" / "groups" / "victoria_island_corridor.json"

if not csv_data.exists():
    print(f"[ERROR] Données TomTom manquantes: {csv_data}")
    print("[ERROR] Impossible de lancer la calibration sans données réelles")
    sys.exit(1)

if not network_json.exists():
    print(f"[ERROR] Définition réseau manquante: {network_json}")
    print("[ERROR] Impossible de lancer la calibration sans définition réseau")
    sys.exit(1)

print(f"[SUCCESS] Données réelles trouvées:")
print(f"  ✅ CSV TomTom: {csv_data} ({csv_data.stat().st_size / 1024:.1f} KB)")
print(f"  ✅ Network JSON: {network_json}")

# Lancer la validation section 7.4
print("\n[2/3] Lancement de la validation section 7.4...")
print("  Revendication testée: R2 (calibration Victoria Island)")
print("  Tests inclus:")
print("    - Calibration avec vraies données TomTom")
print("    - Métriques MAPE < 25%, GEH < 8.0")
print("    - Validation croisée robustesse")
print("    - Génération figures PNG et LaTeX")

print("\n[DATA] Source de données:")
print("    - REAL TomTom data: donnees_trafic_75_segments.csv")
print("    - 70 segments Victoria Island corridor")
print("    - NO synthetic data generation")

try:
    success, kernel_slug = manager.run_validation_section(
        section_name="section_7_4_calibration",
        timeout=14400  # 4 heures max (60 min estimé + marge pour calibration)
    )
    
    if success:
        print("\n" + "=" * 80)
        print("[SUCCESS] VALIDATION KAGGLE TERMINÉE")
        print("=" * 80)
        print(f"\n[INFO] Kernel: {kernel_slug}")
        print(f"[URL] https://www.kaggle.com/code/{kernel_slug}")
        print("\n[3/3] Résultats téléchargés:")
        print(f"  - Figures PNG: validation_ch7/results/section_7_4_calibration/validation_results/figures/")
        print(f"  - LaTeX: validation_ch7/results/section_7_4_calibration/validation_results/section_7_4_content.tex")
        print(f"  - Metrics JSON: validation_ch7/results/section_7_4_calibration/validation_results/section_7_4_calibration_results.json")
        print(f"  - Session summary: validation_ch7/results/section_7_4_calibration/validation_results/session_summary.json")
        
        print("\n[METRICS] Attendus:")
        print("  - MAPE: < 25% (amélioration vs 123% synthétique)")
        print("  - GEH: < 8.0")
        print("  - Theil U: < 0.5")
        
        print("\n[NEXT] Pour intégrer dans la thèse:")
        print("  \\input{validation_ch7/results/section_7_4_calibration/validation_results/section_7_4_content.tex}")
        
        # Display calibration metrics if available
        import json
        metrics_file = Path("validation_ch7/results/section_7_4_calibration/validation_results/section_7_4_calibration_results.json")
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                results = json.load(f)
                r2_metrics = results.get('r2_calibration', {}).get('metrics', {})
                if r2_metrics:
                    print("\n[RESULTS] Métriques de calibration:")
                    print(f"  MAPE: {r2_metrics.get('mape', 0.0):.2f}%")
                    print(f"  GEH: {r2_metrics.get('geh', 0.0):.2f}")
                    print(f"  Theil U: {r2_metrics.get('theil_u', 0.0):.3f}")
                    print(f"  Vitesse simulée: {r2_metrics.get('simulated_mean', 0.0):.1f} km/h")
                    print(f"  Vitesse observée: {r2_metrics.get('observed_mean', 0.0):.1f} km/h")
        
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
