#!/usr/bin/env python3
"""
Script pour télécharger les résultats du kernel zimd avec gestion Unicode.
"""
import os
import sys
from pathlib import Path
from kaggle import KaggleApi
import shutil
import tempfile
import json

def download_kernel_results(kernel_slug: str, output_dir: str):
    """
    Télécharge les résultats d'un kernel avec gestion Unicode appropriée.
    
    Args:
        kernel_slug: Slug du kernel (e.g., 'elonmj/arz-validation-75digitaltwin-zimd')
        output_dir: Répertoire de destination
    """
    print(f"[DOWNLOAD] Téléchargement des résultats: {kernel_slug}")
    
    # Initialiser l'API Kaggle
    api = KaggleApi()
    api.authenticate()
    print("[OK] API Kaggle authentifiée")
    
    # Créer un répertoire temporaire
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"[TEMP] Répertoire temporaire: {temp_dir}")
        
        try:
            # Télécharger avec gestion d'erreur Unicode
            print("[API] Appel kernels_output...")
            
            # SOLUTION: Patcher temporairement l'environnement système
            import locale
            import codecs
            
            # Sauvegarder l'encodage original
            original_encoding = sys.stdout.encoding
            
            # Forcer UTF-8 au niveau système
            if sys.platform == 'win32':
                # Sur Windows, utiliser un writer UTF-8 custom
                import io
                
                # Créer un nouveau stdout avec UTF-8
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer,
                    encoding='utf-8',
                    errors='replace',
                    line_buffering=True
                )
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer,
                    encoding='utf-8',
                    errors='replace',
                    line_buffering=True
                )
            
            try:
                # L'API devrait maintenant fonctionner
                api.kernels_output(kernel_slug, temp_dir)
                print("[OK] Téléchargement réussi")
            except UnicodeEncodeError as unicode_err:
                # Si l'erreur persiste, utiliser une méthode alternative
                print(f"[WARNING] Erreur Unicode persistante: {unicode_err}")
                print("[WORKAROUND] Utilisation de la méthode alternative...")
                
                # Méthode alternative: télécharger via subprocess avec encodage forcé
                import subprocess
                cmd = f'kaggle kernels output {kernel_slug} -p "{temp_dir}"'
                
                # Exécuter avec encodage UTF-8 forcé
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env
                )
                
                if result.returncode == 0:
                    print("[OK] Téléchargement réussi (méthode alternative)")
                else:
                    print(f"[ERROR] Échec: {result.stderr}")
                    raise Exception(f"Téléchargement échoué: {result.stderr}")
            
            
            # Lister les fichiers téléchargés
            temp_path = Path(temp_dir)
            downloaded_files = list(temp_path.rglob('*'))
            print(f"[FILES] {len(downloaded_files)} fichiers téléchargés")
            
            # Copier vers le répertoire de destination
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for file_path in downloaded_files:
                if file_path.is_file():
                    relative_path = file_path.relative_to(temp_path)
                    dest_path = output_path / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copier avec gestion Unicode pour les fichiers texte
                    if file_path.suffix in ['.txt', '.log', '.json', '.csv', '.tex']:
                        try:
                            # Lire avec UTF-8, remplacer les caractères problématiques
                            content = file_path.read_text(encoding='utf-8', errors='replace')
                            dest_path.write_text(content, encoding='utf-8')
                            print(f"  [TEXT] {relative_path}")
                        except Exception as e:
                            print(f"  [WARNING] Erreur lecture texte {relative_path}: {e}")
                            # Fallback: copie binaire
                            shutil.copy2(file_path, dest_path)
                    else:
                        # Copie binaire pour les autres fichiers
                        shutil.copy2(file_path, dest_path)
                        print(f"  [BIN] {relative_path}")
            
            print(f"[SUCCESS] Résultats copiés dans: {output_dir}")
            
            # Vérifier la présence de session_summary.json
            summary_path = output_path / "results" / "session_summary.json"
            if summary_path.exists():
                print("[FOUND] session_summary.json")
                try:
                    summary = json.loads(summary_path.read_text(encoding='utf-8'))
                    print(f"[STATUS] Test status: {summary.get('test_status', 'N/A')}")
                    print(f"[VALIDATION] Overall: {summary.get('overall_validation', 'N/A')}")
                    return summary
                except Exception as e:
                    print(f"[ERROR] Lecture session_summary.json: {e}")
            else:
                print("[WARNING] session_summary.json non trouvé")
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Erreur téléchargement: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    kernel_slug = "elonmj/arz-validation-75digitaltwin-zimd"
    output_dir = "d:/Projets/Alibi/Code project/validation_output/results/elonmj_arz-validation-75digitaltwin-zimd"
    
    summary = download_kernel_results(kernel_slug, output_dir)
    
    if summary:
        print("\n" + "="*80)
        print("RÉSULTATS SECTION 7.5 - KERNEL ZIMD")
        print("="*80)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("\n[WARNING] Impossible de récupérer le résumé complet")
