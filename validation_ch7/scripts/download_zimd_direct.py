"""
Téléchargement direct des résultats zimd avec contournement complet de l'encodage.
Utilise directement la CLI Kaggle avec redirection de sortie supprimée.
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path

def download_results_direct():
    """Téléchargement avec suppression complète de la sortie log."""
    
    kernel_slug = "elonmj/arz-validation-75digitaltwin-zimd"
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "validation_output" / "results" / "elonmj_arz-validation-75digitaltwin-zimd"
    
    print(f"[DOWNLOAD] Téléchargement: {kernel_slug}")
    print(f"[OUTPUT] Destination: {output_dir}")
    
    # Créer le répertoire de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Commande avec redirection vers NUL pour ignorer complètement les logs
    cmd = f'kaggle kernels output {kernel_slug} -p "{output_dir}" > nul 2>&1'
    
    print(f"[CMD] Exécution: kaggle kernels output (sortie supprimée)")
    print("[INFO] Ceci peut prendre 1-2 minutes...")
    
    # Exécution avec shell=True pour supporter la redirection
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(output_dir.parent)
    )
    
    # Vérifier si des fichiers ont été téléchargés
    if result.returncode == 0:
        files = list(output_dir.glob("*"))
        print(f"[OK] Téléchargement réussi: {len(files)} fichiers")
        
        # Lister les fichiers téléchargés
        for f in files:
            print(f"  - {f.name} ({f.stat().st_size:,} bytes)")
        
        # Lire session_summary.json s'il existe
        summary_path = output_dir / "session_summary.json"
        if summary_path.exists():
            print(f"\n[SUMMARY] Lecture de session_summary.json...")
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            print(f"[VALIDATION] Overall: {summary.get('overall_validation', 'N/A')}")
            
            # Afficher les résultats des tests
            if 'test_status' in summary:
                print(f"\n[TEST_STATUS]")
                for test_name, test_result in summary['test_status'].items():
                    status_symbol = "✅" if test_result.get('success', False) else "❌"
                    print(f"  {status_symbol} {test_name}: {test_result.get('success', False)}")
            
            return True
        else:
            print(f"[WARNING] session_summary.json non trouvé dans {output_dir}")
            return False
    else:
        print(f"[ERROR] Échec du téléchargement (code: {result.returncode})")
        return False

if __name__ == "__main__":
    try:
        success = download_results_direct()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
