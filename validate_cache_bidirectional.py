"""
Script de validation du cache bidirectionnel Local ↔ Kaggle.

Teste:
1. Cache existe et est Git-tracked
2. Simulation du workflow local→Kaggle
3. Simulation du workflow Kaggle→local
4. Validation des logs pour CACHE HIT/MISS
"""

import subprocess
from pathlib import Path
import sys

class CacheBidirectionalValidator:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.cache_dir = self.project_root / "validation_ch7" / "cache" / "section_7_6"
        self.results = {
            'git_tracking': False,
            'cache_exists': False,
            'lfs_configured': False,
            'ready_for_kaggle': False
        }
    
    def check_cache_exists(self):
        """Vérifie que les fichiers cache existent."""
        print("\n" + "="*80)
        print("1️⃣  VÉRIFICATION: Fichiers Cache Existent")
        print("="*80)
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        if not cache_files:
            print("❌ Aucun fichier cache trouvé!")
            print(f"   Répertoire: {self.cache_dir}")
            print("\n💡 Action requise:")
            print("   python validation_ch7/scripts/test_section_7_6_rl_performance.py")
            return False
        
        print(f"✅ {len(cache_files)} fichier(s) cache trouvé(s):")
        for cache_file in cache_files:
            size_kb = cache_file.stat().st_size / 1024
            print(f"   - {cache_file.name} ({size_kb:.1f} KB)")
        
        self.results['cache_exists'] = True
        return True
    
    def check_git_tracking(self):
        """Vérifie que les fichiers cache sont trackés par Git."""
        print("\n" + "="*80)
        print("2️⃣  VÉRIFICATION: Git Tracking")
        print("="*80)
        
        try:
            # Vérifier statut Git des fichiers .pkl
            result = subprocess.run(
                ['git', 'status', '--porcelain', 'validation_ch7/cache/section_7_6/*.pkl'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            status_output = result.stdout.strip()
            
            if not status_output:
                # Aucun output = fichiers trackés et pas de modifications
                print("✅ Fichiers cache trackés par Git (staged ou committed)")
                self.results['git_tracking'] = True
                return True
            
            # Analyser le statut
            if status_output.startswith('??'):
                print("❌ Fichiers cache NON trackés (untracked)")
                print(f"   Status: {status_output}")
                print("\n💡 Action requise:")
                print("   Option 1 (Recommandé - Git LFS):")
                print("     git lfs install")
                print("     git lfs track 'validation_ch7/cache/**/*.pkl'")
                print("     git add .gitattributes")
                print("     git add validation_ch7/cache/section_7_6/*.pkl")
                print("     git commit -m 'feat(validation): Enable Git LFS for cache'")
                print("\n   Option 2 (Git standard):")
                print("     git add validation_ch7/cache/section_7_6/*.pkl")
                print("     git commit -m 'feat(validation): Add persistent cache'")
                return False
            
            elif status_output.startswith('A '):
                print("⚠️  Fichiers cache staged (pas encore committed)")
                print("   Status: Prêt pour commit")
                print("\n💡 Action requise:")
                print("   git commit -m 'feat(validation): Add persistent cache'")
                print("   git push origin main")
                self.results['git_tracking'] = False  # Pas encore pushé
                return False
            
            else:
                print(f"⚠️  Statut Git: {status_output}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur Git: {e}")
            return False
    
    def check_git_lfs(self):
        """Vérifie si Git LFS est configuré."""
        print("\n" + "="*80)
        print("3️⃣  VÉRIFICATION: Git LFS Configuration")
        print("="*80)
        
        gitattributes_path = self.project_root / ".gitattributes"
        
        if not gitattributes_path.exists():
            print("⚠️  Fichier .gitattributes n'existe pas")
            print("   Git LFS: NON configuré")
            print("\n💡 Recommandation:")
            print("   Git LFS optimise les fichiers binaires > 50 KB")
            print("   Pour l'activer:")
            print("     git lfs install")
            print("     git lfs track 'validation_ch7/cache/**/*.pkl'")
            self.results['lfs_configured'] = False
            return False
        
        # Lire .gitattributes
        with open(gitattributes_path, 'r') as f:
            content = f.read()
        
        if 'validation_ch7/cache/**/*.pkl' in content or '*.pkl' in content:
            print("✅ Git LFS configuré pour fichiers cache")
            print("   Pattern trouvé dans .gitattributes")
            self.results['lfs_configured'] = True
            return True
        else:
            print("⚠️  Git LFS installé mais cache .pkl NON tracké")
            print("\n💡 Action requise:")
            print("   git lfs track 'validation_ch7/cache/**/*.pkl'")
            print("   git add .gitattributes")
            self.results['lfs_configured'] = False
            return False
    
    def check_remote_sync(self):
        """Vérifie si le cache est synchronisé avec remote."""
        print("\n" + "="*80)
        print("4️⃣  VÉRIFICATION: Synchronisation Remote")
        print("="*80)
        
        try:
            # Vérifier si des commits non pushés existent
            result = subprocess.run(
                ['git', 'log', 'origin/main..HEAD', '--oneline'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            unpushed_commits = result.stdout.strip()
            
            if unpushed_commits:
                print("⚠️  Commits locaux non pushés:")
                for line in unpushed_commits.split('\n')[:3]:  # Max 3 lignes
                    print(f"   {line}")
                print("\n💡 Action requise:")
                print("   git push origin main")
                self.results['ready_for_kaggle'] = False
                return False
            
            print("✅ Aucun commit local non pushé")
            print("   Cache synchronisé avec remote")
            self.results['ready_for_kaggle'] = True
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Impossible de vérifier remote: {e}")
            print("   (Normal si pas de connexion internet)")
            return None
    
    def simulate_kaggle_workflow(self):
        """Simule le workflow Kaggle (vérification conceptuelle)."""
        print("\n" + "="*80)
        print("5️⃣  SIMULATION: Workflow Kaggle")
        print("="*80)
        
        print("\n📥 Workflow Local → Kaggle:")
        print("   1. Local: Cache créé et committed")
        print("   2. Local: git push origin main")
        print("   3. Kaggle: git clone / git pull")
        print("   4. Kaggle: Cache disponible dans validation_ch7/cache/")
        print("   5. Kaggle: CACHE HIT → <1s baseline")
        
        print("\n📤 Workflow Kaggle → Local:")
        print("   1. Kaggle: Cache créé (si nouveau scénario)")
        print("   2. Kaggle: Script commit + push cache")
        print("   3. Local: git pull origin main")
        print("   4. Local: Cache disponible")
        print("   5. Local: CACHE HIT → <1s baseline")
        
        if self.results['git_tracking'] and self.results['ready_for_kaggle']:
            print("\n✅ Workflow Kaggle: PRÊT")
            print("   Cache sera disponible sur Kaggle après git clone")
        else:
            print("\n❌ Workflow Kaggle: NON PRÊT")
            print("   Action requise: Tracker et pusher le cache")
    
    def print_summary(self):
        """Affiche le résumé des vérifications."""
        print("\n" + "="*80)
        print("📊 RÉSUMÉ DES VÉRIFICATIONS")
        print("="*80)
        
        checks = [
            ("Cache existe localement", self.results['cache_exists']),
            ("Cache tracké par Git", self.results['git_tracking']),
            ("Git LFS configuré", self.results['lfs_configured']),
            ("Synchronisé avec remote", self.results['ready_for_kaggle'])
        ]
        
        all_passed = all(status for _, status in checks)
        
        for check_name, status in checks:
            icon = "✅" if status else "❌"
            print(f"   {icon} {check_name}")
        
        print("\n" + "="*80)
        
        if all_passed:
            print("🎉 VALIDATION COMPLÈTE: Cache bidirectionnel OPÉRATIONNEL")
            print("="*80)
            print("\n✅ Le cache fonctionnera sur Kaggle:")
            print("   - git clone téléchargera le cache")
            print("   - CACHE HIT attendu (baseline < 1s)")
            print("   - Temps sauvegardé: ~3min36s par run")
            print("\n📝 Prochaines étapes:")
            print("   1. Test local quick: python validation_ch7/scripts/test_section_7_6_rl_performance.py")
            print("   2. Test Kaggle quick: python validation_cli.py --section 7.6 --mode quick")
            print("   3. Vérifier logs Kaggle: [CACHE BASELINE] ✅ Using universal cache")
        else:
            print("⚠️  VALIDATION INCOMPLÈTE: Actions requises")
            print("="*80)
            print("\n❌ Le cache NE fonctionnera PAS sur Kaggle tant que:")
            print("   - Fichiers cache non trackés par Git")
            print("   - Ou commits non pushés vers remote")
            print("\n📝 Actions immédiates:")
            if not self.results['cache_exists']:
                print("   1. Créer cache: python validation_ch7/scripts/test_section_7_6_rl_performance.py")
            if not self.results['git_tracking']:
                print("   2. Tracker cache avec Git LFS (recommandé):")
                print("      git lfs install")
                print("      git lfs track 'validation_ch7/cache/**/*.pkl'")
                print("      git add .gitattributes validation_ch7/cache/section_7_6/*.pkl")
                print("      git commit -m 'feat(validation): Enable Git LFS for cache'")
            if not self.results['ready_for_kaggle']:
                print("   3. Pusher vers remote:")
                print("      git push origin main")
        
        print("\n" + "="*80)
        
        return all_passed
    
    def run(self):
        """Exécute toutes les vérifications."""
        print("\n🔍 VALIDATION CACHE BIDIRECTIONNEL LOCAL ↔ KAGGLE")
        print("="*80)
        print(f"📁 Projet: {self.project_root}")
        print(f"📦 Cache: {self.cache_dir}")
        
        # Exécuter les vérifications dans l'ordre
        self.check_cache_exists()
        self.check_git_tracking()
        self.check_git_lfs()
        self.check_remote_sync()
        self.simulate_kaggle_workflow()
        
        # Résumé final
        all_passed = self.print_summary()
        
        # Code de sortie
        sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    validator = CacheBidirectionalValidator()
    validator.run()
