#!/usr/bin/env python3
"""
Script de lancement pour upload et monitoring Kaggle - Section 7.5 Digital Twin

Tests Revendications R4 (Behavioral Reproduction) et R6 (Robustness)
"""

import sys
from pathlib import Path

# Ajout du chemin projet
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from validation_ch7.scripts.validation_kaggle_manager import ValidationKaggleManager

print("=" * 80)
print("LANCEMENT VALIDATION KAGGLE GPU - SECTION 7.5 DIGITAL TWIN")
print("=" * 80)

# Initialiser le manager (lit automatiquement kaggle.json)
print("\n[1/3] Initialisation du ValidationKaggleManager...")
manager = ValidationKaggleManager()

print(f"\n[INFO] Configuration:")
print(f"  - Repository: {manager.repo_url}")
print(f"  - Branch: {manager.branch}")
print(f"  - Username: {manager.username}")
print(f"  - Durée estimée: 90-120 minutes sur GPU")

# Vérifier que le script de test existe
print("\n[INFO] Vérification du script de validation...")
test_script = project_root / "validation_ch7" / "scripts" / "test_section_7_5_digital_twin.py"

if not test_script.exists():
    print(f"[ERROR] Script de test manquant: {test_script}")
    sys.exit(1)

print(f"[SUCCESS] Script trouvé: {test_script}")

# Lancer la validation section 7.5
print("\n[2/3] Lancement de la validation section 7.5...")
print("  Revendications testées:")
print("    - R4: Reproduction des comportements de trafic observés")
print("    - R6: Robustesse sous conditions dégradées")
print("\n  Tests inclus:")
print("    1. Behavioral Reproduction (3 scenarios):")
print("       - Free flow (trafic fluide)")
print("       - Congestion (congestion modérée)")
print("       - Jam formation (formation de bouchon)")
print("    2. Robustness Tests (3 perturbations):")
print("       - Density increase +50%")
print("       - Velocity decrease -30%")
print("       - Road degradation (R=1)")
print("    3. Cross-scenario validation (fundamental diagram)")
print("\n  Outputs générés:")
print("    - 4 figures PNG publication-ready (300 DPI)")
print("    - 3 CSV metrics (behavioral, robustness, summary)")
print("    - LaTeX content enrichi avec méthodologie + discussion")
print("    - Session summary JSON")

try:
    success, kernel_slug = manager.run_validation_section(
        section_name="section_7_5_digital_twin",
        timeout=14400  # 4 heures max (2h estimé + marge pour simulations)
    )
    
    if success:
        print("\n" + "=" * 80)
        print("[SUCCESS] VALIDATION KAGGLE TERMINÉE")
        print("=" * 80)
        print(f"\n[INFO] Kernel: {kernel_slug}")
        print(f"[URL] https://www.kaggle.com/code/{kernel_slug}")
        print("\n[3/3] Résultats téléchargés:")
        
        # Chemins des résultats
        results_base = Path("validation_output/results") / kernel_slug / "section_7_5_digital_twin"
        
        print(f"\n[STRUCTURE] {results_base}/")
        print(f"  ├── figures/")
        print(f"  │   ├── fig_behavioral_patterns.png")
        print(f"  │   ├── fig_robustness_perturbations.png")
        print(f"  │   ├── fig_fundamental_diagram.png")
        print(f"  │   └── fig_digital_twin_metrics.png")
        print(f"  ├── data/")
        print(f"  │   ├── scenarios/ (YAML configs)")
        print(f"  │   └── metrics/ (CSV files)")
        print(f"  │       ├── behavioral_metrics.csv")
        print(f"  │       ├── robustness_metrics.csv")
        print(f"  │       └── summary_metrics.csv")
        print(f"  ├── latex/")
        print(f"  │   └── section_7_5_digital_twin_content.tex")
        print(f"  └── session_summary.json")
        
        print("\n[FIGURES] Copiées dans:")
        print("  chapters/partie3/images/")
        print("    - fig_behavioral_patterns.png")
        print("    - fig_robustness_perturbations.png")
        print("    - fig_fundamental_diagram.png")
        print("    - fig_digital_twin_metrics.png")
        
        print("\n[LATEX] Contenu généré:")
        print("  chapters/partie3/section_7_5_digital_twin_content.tex")
        
        print("\n[METRICS] Critères de validation:")
        print("  R4 (Behavioral Reproduction):")
        print("    - Densités dans plages attendues par scénario")
        print("    - Vitesses cohérentes avec régime de trafic")
        print("    - Conservation de masse < 1% erreur")
        print("  R6 (Robustness):")
        print("    - Stabilité numérique (pas de NaN/explosions)")
        print("    - Temps convergence < seuils (150-200s)")
        print("    - RMSE final acceptable")
        
        print("\n[NEXT] Pour intégrer dans la thèse:")
        print("  Dans chapters/partie3/ch7_validation_entrainement.tex, ajouter:")
        print("  \\input{chapters/partie3/section_7_5_digital_twin_content.tex}")
        
        # Display test results if available
        import json
        summary_file = results_base / "session_summary.json"
        if summary_file.exists():
            print("\n[RESULTS] Résumé de validation:")
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                test_status = summary.get('test_status', {})
                
                print(f"  R4 Behavioral Reproduction: {'✅ PASSED' if test_status.get('behavioral_reproduction') else '❌ FAILED'}")
                print(f"  R6 Robustness Degraded:     {'✅ PASSED' if test_status.get('robustness') else '❌ FAILED'}")
                print(f"  Cross-Scenario Validation:   {'✅ PASSED' if test_status.get('cross_scenario') else '❌ FAILED'}")
                
                overall = summary.get('overall_validation', False)
                print(f"\n  OVERALL STATUS: {'✅ PASSED' if overall else '❌ FAILED'}")
                
                # Artifacts count
                artifacts = summary.get('artifacts', {})
                print(f"\n[ARTIFACTS] Générés:")
                print(f"  - Figures: {artifacts.get('figures', 0)}")
                print(f"  - CSV files: {artifacts.get('csv_files', 0)}")
                print(f"  - Scenarios: {artifacts.get('scenarios', 0)}")
                print(f"  - LaTeX files: {artifacts.get('latex_files', 0)}")
        else:
            print(f"\n[WARNING] Session summary not found: {summary_file}")
            print("  Check kernel logs for detailed results")
        
        print("\n" + "=" * 80)
        print("VALIDATION SECTION 7.5 COMPLETE")
        print("=" * 80)
        
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("[FAILED] VALIDATION KAGGLE ÉCHOUÉE")
        print("=" * 80)
        print(f"\n[ERROR] Kernel: {kernel_slug}")
        print(f"[URL] https://www.kaggle.com/code/{kernel_slug}")
        print("\n[DEBUG] Vérifier les logs Kaggle pour diagnostiquer l'erreur")
        print("  Causes possibles:")
        print("    - Simulation timeout (augmenter timeout si nécessaire)")
        print("    - Erreur dans les configurations de scénarios")
        print("    - Problème d'initialisation SimulationRunner")
        print("    - Manque de mémoire GPU")
        
        sys.exit(1)
        
except KeyboardInterrupt:
    print("\n\n[INTERRUPTED] Validation interrompue par l'utilisateur")
    print("  Le kernel Kaggle continue de s'exécuter en arrière-plan")
    print("  Vous pouvez vérifier l'état sur: https://www.kaggle.com/code")
    sys.exit(130)
    
except Exception as e:
    print("\n" + "=" * 80)
    print("[EXCEPTION] Erreur durant la validation")
    print("=" * 80)
    print(f"\n[ERROR] {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
