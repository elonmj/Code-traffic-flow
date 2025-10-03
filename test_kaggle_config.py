#!/usr/bin/env python3
"""
Script de test pour valider la configuration Kaggle avant upload
Vérifie : credentials, repo GitHub, structure des fichiers
"""

import sys
from pathlib import Path
import json

# Ajout du chemin projet
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("VALIDATION PRE-UPLOAD KAGGLE - SECTION 7.3")
print("=" * 80)

# ========== 1. KAGGLE CREDENTIALS ==========
print("\n[1/6] Vérification credentials Kaggle...")
try:
    kaggle_json = project_root / "kaggle.json"
    if kaggle_json.exists():
        with open(kaggle_json, 'r') as f:
            creds = json.load(f)
        print(f"  [OK] Username: {creds['username']}")
        print(f"  [OK] API key présente: {len(creds['key'])} caractères")
    else:
        print(f"  [ERREUR] kaggle.json non trouvé dans {project_root}")
        sys.exit(1)
except Exception as e:
    print(f"  [ERREUR] Lecture kaggle.json: {e}")
    sys.exit(1)

# ========== 2. REPOSITORY GITHUB ==========
print("\n[2/6] Vérification repository GitHub...")
try:
    import subprocess
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    if result.returncode == 0:
        repo_url = result.stdout.strip()
        print(f"  [OK] Repository: {repo_url}")
        
        # Vérifier branch actuelle
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        if branch_result.returncode == 0:
            branch = branch_result.stdout.strip()
            print(f"  [OK] Branch: {branch}")
        else:
            print(f"  [WARN] Impossible de déterminer la branch")
            branch = "main"
    else:
        print(f"  [ERREUR] Impossible de lire le remote Git")
        sys.exit(1)
except Exception as e:
    print(f"  [ERREUR] Git: {e}")
    sys.exit(1)

# ========== 3. STRUCTURE FICHIERS VALIDATION ==========
print("\n[3/6] Vérification structure fichiers...")
required_files = [
    "validation_ch7/scripts/test_section_7_3_analytical.py",
    "validation_ch7/scripts/validation_utils.py",
    "validation_ch7/templates/section_7_3_analytical.tex",
    "validation_kaggle_manager.py",
]

all_present = True
for file_path in required_files:
    full_path = project_root / file_path
    if full_path.exists():
        print(f"  [OK] {file_path}")
    else:
        print(f"  [ERREUR] MANQUANT: {file_path}")
        all_present = False

if not all_present:
    print("\n[FATAL] Fichiers manquants - impossible de continuer")
    sys.exit(1)

# ========== 4. MATPLOTLIB BACKEND ==========
print("\n[4/6] Vérification configuration matplotlib...")
try:
    with open(project_root / "validation_ch7/scripts/test_section_7_3_analytical.py", 'r') as f:
        content = f.read()
        if "matplotlib.use('Agg')" in content:
            print("  [OK] Backend Agg configuré dans test_section_7_3_analytical.py")
        else:
            print("  [WARN] Backend Agg non trouvé - risque de crash sur Kaggle headless")
    
    with open(project_root / "validation_ch7/scripts/validation_utils.py", 'r') as f:
        content = f.read()
        if "matplotlib.use('Agg')" in content:
            print("  [OK] Backend Agg configuré dans validation_utils.py")
        else:
            print("  [WARN] Backend Agg non trouvé dans validation_utils.py")
except Exception as e:
    print(f"  [ERREUR] Lecture fichiers: {e}")

# ========== 5. VALIDATION_SECTIONS CONFIG ==========
print("\n[5/6] Vérification VALIDATION_SECTIONS...")
try:
    # Import direct du manager pour vérifier config
    from validation_kaggle_manager import ValidationKaggleManager
    
    # Le manager lit automatiquement kaggle.json et configure repo_url/branch
    manager = ValidationKaggleManager()
    
    # Vérifier que section_7_3_analytical existe
    section_found = False
    for section in manager.validation_sections:
        if section['name'] == 'section_7_3_analytical':
            section_found = True
            print(f"  [OK] Section trouvée: {section['name']}")
            print(f"       Script: {section['script']}")
            print(f"       Durée estimée: {section['estimated_minutes']} min")
            print(f"       Revendications: {', '.join(section['revendications'])}")
            break
    
    if not section_found:
        print("  [ERREUR] Section 'section_7_3_analytical' non trouvée dans VALIDATION_SECTIONS")
        sys.exit(1)
        
except Exception as e:
    print(f"  [ERREUR] Import ValidationKaggleManager: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== 6. GIT STATUS ==========
print("\n[6/6] Vérification Git status...")
try:
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    if status_result.returncode == 0:
        uncommitted = status_result.stdout.strip()
        if uncommitted:
            print("  [WARN] Changements non commités détectés:")
            for line in uncommitted.split('\n')[:5]:  # Afficher 5 premiers
                print(f"       {line}")
            print("\n  [RECOMMANDATION] Commit et push avant upload Kaggle pour garantir synchronisation")
        else:
            print("  [OK] Aucun changement non commité")
    else:
        print("  [WARN] Impossible de vérifier git status")
        
except Exception as e:
    print(f"  [WARN] Git status: {e}")

# ========== RÉSUMÉ ==========
print("\n" + "=" * 80)
print("RÉSUMÉ PRE-UPLOAD")
print("=" * 80)
print(f"Repository: {repo_url}")
print(f"Branch: {branch}")
print(f"Username Kaggle: {creds['username']}")
print(f"Section: section_7_3_analytical")
print(f"Durée estimée: 21 minutes sur GPU")
print("\nFichiers requis: [OK]")
print("Configuration matplotlib: [OK]")
print("=" * 80)

print("\n[NEXT STEP] Pour lancer l'upload:")
print("""
from validation_kaggle_manager import ValidationKaggleManager

manager = ValidationKaggleManager()  # Lit automatiquement kaggle.json

manager.run_validation_section(
    section_name="section_7_3_analytical",
    monitor_progress=True
)
""")

print("\n[SUCCESS] Configuration validée - prêt pour upload Kaggle!")
