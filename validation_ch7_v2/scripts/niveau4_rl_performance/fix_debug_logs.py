#!/usr/bin/env python3
"""
🔧 SCRIPT DE FIX: Désactive les logs debug excessifs

PROBLÈME: Les logs [DEBUG_BC_GPU] et [DEBUG_BC_DISPATCHER] ralentissent
         massivement l'exécution (16k lignes en 3 secondes!)

SOLUTION: Désactive temporairement ces logs pour gagner 10-100x en vitesse
"""

import os
from pathlib import Path

def disable_debug_logs():
    """Désactive les logs debug dans le code ARZ."""
    
    print("=" * 80)
    print("🔧 DÉSACTIVATION DES LOGS DEBUG EXCESSIFS")
    print("=" * 80)
    
    # Trouver les fichiers avec logs debug
    arz_model_dir = Path(__file__).parent.parent.parent.parent / "arz_model"
    
    files_to_check = [
        arz_model_dir / "simulation" / "boundary_conditions_gpu.py",
        arz_model_dir / "simulation" / "boundary_conditions.py",
        arz_model_dir / "simulation" / "runner.py"
    ]
    
    changes_made = 0
    
    for file_path in files_to_check:
        if not file_path.exists():
            continue
        
        print(f"\n📄 Checking: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Remplacer les print debug par pass
        replacements = [
            ('print("[DEBUG_BC_GPU]', '# print("[DEBUG_BC_GPU]'),
            ('print("[DEBUG_BC_DISPATCHER]', '# print("[DEBUG_BC_DISPATCHER]'),
            ('print(f"[DEBUG_BC_GPU]', '# print(f"[DEBUG_BC_GPU]'),
            ('print(f"[DEBUG_BC_DISPATCHER]', '# print(f"[DEBUG_BC_DISPATCHER]'),
        ]
        
        for old, new in replacements:
            if old in content:
                count = content.count(old)
                content = content.replace(old, new)
                print(f"   ✅ Commented out {count} debug print statements")
                changes_made += count
        
        if content != original_content:
            # Backup original
            backup_path = file_path.with_suffix('.py.backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            print(f"   💾 Backup saved: {backup_path.name}")
            
            # Write modified
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   💾 Modified file saved")
    
    print("\n" + "=" * 80)
    if changes_made > 0:
        print(f"✅ SUCCÈS: {changes_made} debug statements commentés")
        print("   Vitesse d'exécution devrait être 10-100x plus rapide!")
    else:
        print("ℹ️  Aucun debug statement trouvé (peut-être déjà désactivés)")
    print("=" * 80)
    
    return changes_made


def restore_debug_logs():
    """Restaure les logs debug depuis les backups."""
    
    print("=" * 80)
    print("🔄 RESTAURATION DES LOGS DEBUG")
    print("=" * 80)
    
    arz_model_dir = Path(__file__).parent.parent.parent.parent / "arz_model"
    
    backups = list(arz_model_dir.rglob("*.py.backup"))
    
    if not backups:
        print("ℹ️  Aucun backup trouvé")
        return
    
    for backup_path in backups:
        original_path = backup_path.with_suffix('')
        
        print(f"\n📄 Restoring: {original_path.name}")
        
        with open(backup_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        with open(original_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        backup_path.unlink()
        print(f"   ✅ Restored from backup")
    
    print("\n✅ Tous les fichiers restaurés")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        restore_debug_logs()
    else:
        disable_debug_logs()
        print("\n💡 Pour restaurer les logs: python fix_debug_logs.py --restore")
