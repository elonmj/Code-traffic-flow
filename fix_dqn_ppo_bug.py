"""
Script pour corriger automatiquement le bug DQN/PPO dans test_section_7_6_rl_performance.py

Ce bug empêche le chargement du modèle PPO entraîné, causant un CSV vide.

ERREUR ACTUELLE:
    AttributeError: 'ActorCriticPolicy' object has no attribute 'q_net'
    
CAUSE:
    Le modèle a été entraîné avec PPO (ActorCriticPolicy)
    Mais le code essaie de le charger avec DQN.load() (Q-network policy)

SOLUTION:
    Remplacer DQN.load() par PPO.load()
"""

import os
import re

# Fichier à corriger
FILE_PATH = "validation_ch7/scripts/test_section_7_6_rl_performance.py"

print("=" * 80)
print("CORRECTION AUTOMATIQUE - BUG DQN/PPO")
print("=" * 80)

# Vérifier que le fichier existe
if not os.path.exists(FILE_PATH):
    print(f"\n❌ ERROR: File not found: {FILE_PATH}")
    print("\nVeuillez vérifier le chemin du fichier.")
    exit(1)

print(f"\n📄 Fichier: {FILE_PATH}")

# Lire le contenu actuel
with open(FILE_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"\n📊 Taille fichier: {len(content)} caractères")

# Rechercher les patterns à corriger
patterns_to_fix = [
    {
        'name': 'Import DQN',
        'old': r'from stable_baselines3 import DQN',
        'new': 'from stable_baselines3 import PPO',
        'line_pattern': r'from stable_baselines3 import.*DQN'
    },
    {
        'name': 'DQN.load()',
        'old': r'DQN\.load\(',
        'new': 'PPO.load(',
        'line_pattern': r'.*DQN\.load\('
    }
]

# Créer une backup
backup_path = FILE_PATH + ".backup"
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"\n💾 Backup créé: {backup_path}")

# Appliquer les corrections
modified = False
new_content = content

for pattern_info in patterns_to_fix:
    print(f"\n🔍 Recherche: {pattern_info['name']}")
    
    # Trouver toutes les occurrences
    matches = list(re.finditer(pattern_info['line_pattern'], new_content, re.MULTILINE))
    
    if matches:
        print(f"   ✅ Trouvé {len(matches)} occurrence(s)")
        
        # Afficher les lignes concernées
        for i, match in enumerate(matches):
            line_start = new_content.rfind('\n', 0, match.start()) + 1
            line_end = new_content.find('\n', match.end())
            if line_end == -1:
                line_end = len(new_content)
            line = new_content[line_start:line_end]
            
            # Calculer le numéro de ligne
            line_num = new_content[:match.start()].count('\n') + 1
            
            print(f"   📍 Ligne {line_num}: {line.strip()}")
        
        # Appliquer le remplacement
        new_content = re.sub(pattern_info['old'], pattern_info['new'], new_content)
        modified = True
    else:
        print(f"   ⚠️  Aucune occurrence trouvée")

# Sauvegarder si modifié
if modified:
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"\n{'=' * 80}")
    print("✅ CORRECTION APPLIQUÉE AVEC SUCCÈS")
    print(f"{'=' * 80}")
    
    print(f"\n📝 Modifications:")
    print(f"   - Import: DQN → PPO")
    print(f"   - Chargement: DQN.load() → PPO.load()")
    
    print(f"\n💾 Fichiers:")
    print(f"   - Original (backup): {backup_path}")
    print(f"   - Corrigé: {FILE_PATH}")
    
    print(f"\n🧪 TEST RECOMMANDÉ:")
    print(f"   python {FILE_PATH}")
    
    print(f"\n📊 RÉSULTAT ATTENDU:")
    print(f"   ✅ Modèle chargé sans erreur")
    print(f"   ✅ CSV rempli avec métriques de comparaison")
    print(f"   ✅ rl_performance_comparison.csv non vide")
    
else:
    print(f"\n{'=' * 80}")
    print("⚠️  AUCUNE MODIFICATION NÉCESSAIRE")
    print(f"{'=' * 80}")
    
    print(f"\n📌 Le fichier semble déjà corrigé ou ne contient pas les patterns recherchés.")
    print(f"\n🔍 Patterns recherchés:")
    for pattern_info in patterns_to_fix:
        print(f"   - {pattern_info['name']}: {pattern_info['old']}")
    
    # Nettoyer le backup si aucune modif
    os.remove(backup_path)
    print(f"\n🗑️  Backup supprimé (pas de modifications)")

print(f"\n{'=' * 80}")
print("ANALYSE COMPLÉMENTAIRE")
print(f"{'=' * 80}")

# Vérifier d'autres imports potentiels
print(f"\n📦 Imports Stable-Baselines3 détectés:")
import_pattern = r'from stable_baselines3 import (\w+(?:, \w+)*)'
imports = re.findall(import_pattern, content)

if imports:
    for imp in imports:
        print(f"   - {imp}")
else:
    print(f"   ⚠️  Aucun import SB3 trouvé")

# Vérifier les appels .load()
print(f"\n🔧 Appels .load() détectés:")
load_pattern = r'(\w+)\.load\('
loads = re.findall(load_pattern, content)

if loads:
    load_counts = {}
    for load in loads:
        load_counts[load] = load_counts.get(load, 0) + 1
    
    for algo, count in load_counts.items():
        print(f"   - {algo}.load(): {count} occurrence(s)")
else:
    print(f"   ⚠️  Aucun appel .load() trouvé")

print(f"\n✅ Analyse terminée!\n")
